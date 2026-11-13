import asyncio
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from .errors import ToolPolicyError

WORKER = Path(__file__).with_name("sandbox_child.py").resolve()


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    output: str = ""
    returncode: int | None = None


async def run_isolated(
    workspace: Path, mode: str, code: str = "", *, timeout: float = 10, output_limit: int = 8192
) -> ExecutionResult:
    if sys.platform != "linux":
        return ExecutionResult("unavailable")
    if mode not in {"probe", "tests", "python"} or timeout <= 0 or output_limit <= 0:
        raise ValueError("Invalid isolated execution options")
    source = code.encode("utf-8")
    if len(source) > 65536:
        raise ToolPolicyError("Python source exceeds the size limit")
    workspace = workspace.resolve(strict=True)
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-ISBu", str(WORKER), str(workspace), mode,
        cwd=workspace, env={"LANG": "C.UTF-8"}, start_new_session=True,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output = bytearray()

    async def send_source():
        try:
            process.stdin.write(source)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()

    async def read_output():
        while chunk := await process.stdout.read(4096):
            remaining = output_limit - len(output)
            output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                return "output_limit"
        return "finished"

    send = asyncio.create_task(send_source())
    try:
        try:
            async with asyncio.timeout(timeout):
                status = await read_output()
                if status == "finished":
                    await process.wait()
        except TimeoutError:
            status = "timeout"
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            # Drain the pipe so asyncio can reap a child killed during output.
            await process.stdout.read()
            await process.wait()
        await send

    if status == "finished":
        status = "unavailable" if process.returncode == 125 else (
            "passed" if process.returncode == 0 else "failed"
        )
    return ExecutionResult(status, output.decode("utf-8", errors="replace"), process.returncode)
