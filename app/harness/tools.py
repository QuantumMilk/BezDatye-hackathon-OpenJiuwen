import json
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath

from .errors import TaskFormatError, ToolPolicyError
from .execution import run_isolated
from .history import AttemptHistory
from .loader import MAX_WORKSPACE_BYTES, MAX_WORKSPACE_ENTRIES, _safe_stat

MAX_TEXT_BYTES = 256 * 1024
MAX_FILE_WRITE = 64 * 1024


def resolve_workspace_path(root: Path, relative_path: str) -> Path:
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute() or PureWindowsPath(relative_path).drive
        or "\\" in relative_path or ":" in relative_path or "\x00" in relative_path
        or ".." in path.parts
    ):
        raise ToolPolicyError("Path must stay inside the workspace")
    target = root
    try:
        for part in path.parts:
            target /= part
            try:
                info = _safe_stat(target)
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(info.st_mode) and (
                not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            ):
                raise ToolPolicyError("Unsupported workspace entry")
    except (OSError, TaskFormatError):
        raise ToolPolicyError("Could not safely inspect workspace path") from None
    return target


def source_path_allowed(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    parts = tuple(part.casefold() for part in path.parts)
    return (
        path.suffix.lower() == ".py"
        and all(not part.startswith((".", "test")) for part in parts)
        and path.name.casefold() != "conftest.py"
    )


class WorkspaceTools:
    def __init__(self, root: Path, *, coding: bool, can_execute: bool = False):
        self.root = root
        self.coding = coding
        self.can_execute = can_execute

    def list_files(self, relative_path: str = ".", limit: int = 100) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ToolPolicyError("File count must be between 1 and 200")
        start = resolve_workspace_path(self.root, relative_path)
        if not start.is_dir():
            raise ToolPolicyError("Expected a workspace directory")
        files = []
        directories = [start]
        while directories and len(files) < limit:
            for item in sorted(directories.pop().iterdir()):
                if len(files) >= limit:
                    break
                relative = item.relative_to(self.root).as_posix()
                safe = resolve_workspace_path(self.root, relative)
                if safe.is_dir():
                    directories.append(safe)
                else:
                    files.append({"path": relative, "bytes": safe.stat().st_size})
        return files

    def read_file(self, relative_path: str, start_line: int = 1, max_lines: int = 80) -> str:
        if type(start_line) is not int or type(max_lines) is not int:
            raise ToolPolicyError("Line numbers must be integers")
        if start_line < 1 or not 1 <= max_lines <= 200:
            raise ToolPolicyError("Invalid line range")
        path = resolve_workspace_path(self.root, relative_path)
        with path.open("rb") as reader:
            data = reader.read(MAX_TEXT_BYTES)
        text = data.decode("utf-8")
        if "\x00" in text:
            raise ToolPolicyError("File must contain UTF-8 text")
        return "\n".join(text.splitlines()[start_line - 1:start_line - 1 + max_lines])[:8192]

    def search_text(self, literal_query: str, relative_path: str = ".") -> list[dict]:
        if not literal_query or len(literal_query) > 256:
            raise ToolPolicyError("Invalid search query")
        target = resolve_workspace_path(self.root, relative_path)
        files = (
            self.list_files(relative_path, 100) if target.is_dir()
            else [{"path": relative_path}]
        )
        matches = []
        for file in files:
            path = resolve_workspace_path(self.root, file["path"])
            with path.open("rb") as reader:
                data = reader.read(MAX_TEXT_BYTES)
            if b"\x00" in data:
                continue
            for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if literal_query in line:
                    matches.append({"path": file["path"], "line": number, "text": line[:300]})
                    if len(matches) == 20:
                        return matches
        return matches

    def write_file(self, relative_path: str, content: str, *, history: AttemptHistory | None = None,
                   tool_call_id: str | None = None) -> str:
        if not self.coding or not source_path_allowed(relative_path):
            raise ToolPolicyError("Only coding source files can be edited")
        path = resolve_workspace_path(self.root, relative_path)
        payload = content.encode("utf-8")
        if len(payload) > MAX_FILE_WRITE:
            raise ToolPolicyError("Source file exceeds the write limit")
        entries = list(self.root.rglob("*"))
        previous_size = path.stat().st_size if path.is_file() else 0
        total = sum(item.stat().st_size for item in entries if item.is_file())
        if total - previous_size + len(payload) > MAX_WORKSPACE_BYTES or (
            not path.exists() and len(entries) >= MAX_WORKSPACE_ENTRIES
        ):
            raise ToolPolicyError("Workspace limit exceeded")
        if history and history.detailed:
            created = not path.is_file()
            changed = created or previous_size != len(payload)
            if not changed:
                try:
                    changed = path.read_bytes() != payload
                except OSError:
                    changed = None
        path.write_bytes(payload)
        if history and history.detailed:
            history.protect_file(payload)
            history.record(
                "workspace_write", tool_call_id=tool_call_id,
                path=history.fragment(relative_path), bytes=len(payload),
                created=created, changed=changed,
            )
        return f"Wrote {len(payload)} bytes"

    async def execute(self, mode: str, code: str = "", *, timeout: float = 10,
                      history: AttemptHistory | None = None, tool_call_id: str | None = None) -> str:
        if not self.can_execute:
            raise ToolPolicyError("Isolated execution is unavailable")
        if mode == "tests":
            if not self.coding:
                raise ToolPolicyError("Public test execution is available for coding tasks")
            test = resolve_workspace_path(self.root, "test_mylib.py")
            if not test.is_file():
                raise ToolPolicyError("No supported public tests in this task")
        elif mode != "python":
            raise ToolPolicyError("Unsupported execution mode")
        result = await run_isolated(self.root, mode, code, timeout=timeout)
        if history:
            history.record("execution_finished", tool_call_id=tool_call_id, mode=mode,
                           status=result.status, returncode=result.returncode)
        if result.status == "unavailable":
            self.can_execute = False
        return json.dumps({"status": result.status, "output": result.output}, ensure_ascii=False)
