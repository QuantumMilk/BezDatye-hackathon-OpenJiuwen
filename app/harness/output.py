import os
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from .contracts import Task
from .errors import ToolPolicyError
from .execution import run_isolated
from .history import AttemptHistory
from .tools import MAX_FILE_WRITE, resolve_workspace_path, source_path_allowed


@dataclass(frozen=True)
class FileChange:
    path: str
    before: bytes | None
    after: bytes


@dataclass(frozen=True)
class ValidatedAnswer:
    text: str
    changes: tuple[FileChange, ...] = ()


def collect_changes(task: Task, workspace: Path) -> tuple[FileChange, ...]:
    changes = []
    original = {
        path.relative_to(task.workspace_dir).as_posix()
        for path in task.workspace_dir.rglob("*") if path.is_file()
    }
    candidate = {
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*") if path.is_file()
    }
    if original - candidate:
        raise ToolPolicyError("Deleting workspace files is unsupported")
    for relative in sorted(candidate):
        source = resolve_workspace_path(task.workspace_dir, relative)
        target = resolve_workspace_path(workspace, relative)
        before = source.read_bytes() if relative in original else None
        after = target.read_bytes()
        if before == after:
            continue
        if not source_path_allowed(relative) or len(after) > MAX_FILE_WRITE:
            raise ToolPolicyError("Candidate modified a protected file")
        changes.append(FileChange(relative, before, after))
    return tuple(changes)


async def validate_candidate(
    task: Task, workspace: Path, answer: str, *, evidence: bool, timeout: float,
    history: AttemptHistory | None = None,
) -> ValidatedAnswer:
    answer = answer.strip()
    if task.task_type == "coding":
        if answer != "DONE":
            raise ValueError("Coding requires a DONE candidate")
        changes = collect_changes(task, workspace)
        if not changes:
            raise ValueError("No coding changes to publish")
        test = resolve_workspace_path(workspace, "test_mylib.py")
        if not test.is_file():
            raise ValueError("No supported public tests")
        if history:
            history.record("candidate_changes", paths=[history.history.clean(change.path) for change in changes])
            history.record("validation_tests_started", timeout=timeout)
        try:
            result = await run_isolated(workspace, "tests", timeout=timeout)
        except (OSError, ValueError) as error:
            if history:
                history.record("validation_tests_finished", status="error", error_type=type(error).__name__)
            raise
        if history:
            history.record(
                "validation_tests_finished", status=result.status, returncode=result.returncode,
                output=history.fragment(
                    result.output, truncated=result.status == "output_limit", payload=True,
                    tail=result.status not in {"passed", "output_limit"},
                ),
            )
        if result.status != "passed":
            raise ValueError(f"Public tests {result.status}")
        return ValidatedAnswer(answer, changes)

    if not answer or len(answer.encode("utf-8")) > 4096 or any(c in answer for c in "\n\r\x00"):
        raise ValueError("Expected a non-empty single-line answer")
    if task.check == "exact_int" and not re.fullmatch(r"[+-]?[0-9]+", answer):
        raise ValueError("Expected an integer answer")
    if not evidence:
        raise ValueError("A supporting tool result is required")
    return ValidatedAnswer(answer)


def atomic_write(path: Path, data: bytes) -> None:
    with NamedTemporaryFile(dir=path.parent, prefix=".agent-", delete=False) as temporary:
        staging = Path(temporary.name)
        try:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        except BaseException:
            temporary.close()
            staging.unlink(missing_ok=True)
            raise
    try:
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def publish(task: Task, answer: ValidatedAnswer) -> Path:
    marker = resolve_workspace_path(task.task_dir, "answer.txt")
    if marker.exists():
        raise FileExistsError("Task already has an answer")
    targets = []
    for change in answer.changes:
        if not source_path_allowed(change.path):
            raise ToolPolicyError("Cannot publish a protected file")
        path = resolve_workspace_path(task.workspace_dir, change.path)
        current = path.read_bytes() if path.exists() else None
        if current != change.before:
            raise ToolPolicyError("Task files changed before publication")
        targets.append((path, change))

    applied = []
    try:
        for path, change in targets:
            atomic_write(path, change.after)
            applied.append((path, change))
        atomic_write(marker, (answer.text + "\n").encode("utf-8"))
    except BaseException:
        if marker.exists():
            raise
        for path, change in reversed(applied):
            try:
                if change.before is None:
                    path.unlink(missing_ok=True)
                else:
                    atomic_write(path, change.before)
            except OSError:
                # A failed rollback leaves source files without a DONE marker.
                continue
        raise
    return marker
