import json
import stat
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from os import stat_result
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from .contracts import Task
from .errors import TaskFormatError

MAX_METADATA = 256 * 1024
MAX_WORKSPACE_BYTES = 128 * 1024 * 1024
MAX_WORKSPACE_ENTRIES = 4096


@dataclass(frozen=True)
class TaskLoadError:
    directory: str
    reason: str


def _safe_stat(path: Path) -> stat_result:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or (
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise TaskFormatError("Task paths cannot contain links or reparse points")
    return info


def load_task(task_dir: Path) -> Task:
    try:
        if not stat.S_ISDIR(_safe_stat(task_dir).st_mode):
            raise TaskFormatError("Task path must be a directory")
        metadata = task_dir / "task.json"
        info = _safe_stat(metadata)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_METADATA:
            raise TaskFormatError("Invalid task metadata file")
        with metadata.open("rb") as source:
            raw = source.read(MAX_METADATA + 1)
        if len(raw) > MAX_METADATA:
            raise TaskFormatError("Task metadata exceeds size limit")
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise TaskFormatError("Task metadata must be a JSON object")
        workspace = task_dir / "workspace"
        try:
            if not stat.S_ISDIR(_safe_stat(workspace).st_mode):
                raise TaskFormatError("Workspace must be a directory")
        except FileNotFoundError:
            pass
        return Task(
            task_id=data.get("task_id"),
            task_type=data.get("task_type"),
            instructions=data.get("instruction"),
            reward=data.get("reward"),
            check=data.get("check"),
            task_dir=task_dir,
            workspace_dir=workspace,
        )
    except (UnicodeError, json.JSONDecodeError, ValidationError, RecursionError):
        raise TaskFormatError("Invalid task metadata") from None
    except OSError:
        raise TaskFormatError("Could not read task files") from None


def load_tasks(root: Path) -> tuple[list[Task], list[TaskLoadError]]:
    tasks = []
    errors = []
    for directory in sorted(root.iterdir()):
        try:
            if not stat.S_ISDIR(_safe_stat(directory).st_mode):
                continue
            metadata = directory / "task.json"
            if not metadata.exists() and not metadata.is_symlink():
                continue
            tasks.append(load_task(directory))
        except (OSError, TaskFormatError) as error:
            reason = str(error) if isinstance(error, TaskFormatError) else "Could not inspect task path"
            errors.append(TaskLoadError(directory.name, reason))

    counts = Counter(task.task_id for task in tasks)
    for task in tasks:
        if counts[task.task_id] > 1:
            errors.append(TaskLoadError(task.task_dir.name, "Duplicate task ID"))
    return [task for task in tasks if counts[task.task_id] == 1], errors


@contextmanager
def working_copy(task: Task, *, temp_root: Path | None = None) -> Iterator[Path]:
    with TemporaryDirectory(prefix="contest-task-", dir=temp_root) as temp:
        destination = Path(temp) / "workspace"
        destination.mkdir()
        remaining = MAX_WORKSPACE_BYTES
        entries = 0
        try:
            mode = _safe_stat(task.workspace_dir).st_mode
        except FileNotFoundError:
            mode = None
        if mode is not None:
            if not stat.S_ISDIR(mode):
                raise TaskFormatError("Workspace must be a directory")
            directories = [(task.workspace_dir, destination)]
            while directories:
                source_dir, target_dir = directories.pop()
                for source in source_dir.iterdir():
                    entries += 1
                    if entries > MAX_WORKSPACE_ENTRIES:
                        raise TaskFormatError("Workspace exceeds entry limit")
                    info = _safe_stat(source)
                    target = target_dir / source.name
                    if stat.S_ISDIR(info.st_mode):
                        target.mkdir()
                        directories.append((source, target))
                    elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                        if info.st_size > remaining:
                            raise TaskFormatError("Workspace exceeds size limit")
                        with source.open("rb") as reader, target.open("xb") as writer:
                            while chunk := reader.read(min(64 * 1024, remaining + 1)):
                                remaining -= len(chunk)
                                if remaining < 0:
                                    raise TaskFormatError("Workspace exceeds size limit")
                                writer.write(chunk)
                    else:
                        raise TaskFormatError("Workspace contains an unsupported file")
        yield destination
