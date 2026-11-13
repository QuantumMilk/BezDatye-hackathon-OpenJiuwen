import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.contracts import Task
from harness.execution import ExecutionResult
from harness.output import atomic_write, publish, validate_candidate


class OutputTests(unittest.IsolatedAsyncioTestCase):
    def test_answer_survives_process_interruption(self):
        with tempfile.TemporaryDirectory() as temp:
            code = """
import sys
from pathlib import Path
from time import sleep
from harness.contracts import Task
from harness.output import ValidatedAnswer, publish

root = Path(sys.argv[1])
task = Task(task_id="first", task_type="math", instructions="Solve it.",
            reward=1, task_dir=root, workspace_dir=root / "workspace")
publish(task, ValidatedAnswer("42"))
print("published", flush=True)
sleep(10)
"""
            with subprocess.Popen(
                [sys.executable, "-B", "-c", code, temp],
                stdout=subprocess.PIPE, text=True,
            ) as process:
                try:
                    self.assertEqual(process.stdout.readline().strip(), "published")
                finally:
                    process.kill()
                    process.wait(timeout=5)
            self.assertEqual((Path(temp) / "answer.txt").read_bytes(), b"42\n")

    async def test_utf8_and_skip_leave_prior_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.make_task(root, "first", "search")
            second = self.make_task(root, "second", "math")
            answer = await validate_candidate(first, first.workspace_dir, "ответ", evidence=True, timeout=1)
            publish(first, answer)
            with self.assertRaises(ValueError):
                await validate_candidate(second, second.workspace_dir, "x", evidence=False, timeout=1)
            self.assertFalse((second.task_dir / "answer.txt").exists())
            self.assertEqual((first.task_dir / "answer.txt").read_bytes(), "ответ\n".encode("utf-8"))

    async def test_code_before_done_and_rollback_on_failed_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = self.make_task(root, "code", "coding")
            original = task.workspace_dir / "solution.py"
            original.write_bytes(b"x = 0\n")
            (task.workspace_dir / "test_mylib.py").write_bytes(b"assert True\n")
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "solution.py").write_bytes(b"x = 1\n")
            (candidate / "test_mylib.py").write_bytes(b"assert True\n")
            with patch("harness.output.run_isolated", return_value=ExecutionResult("passed")):
                answer = await validate_candidate(task, candidate, "DONE", evidence=False, timeout=1)
            marker = task.task_dir / "answer.txt"

            def fail_marker(path, data):
                if path == marker:
                    self.assertEqual(original.read_bytes(), b"x = 1\n")
                    raise OSError("fixture")
                atomic_write(path, data)

            with patch("harness.output.atomic_write", side_effect=fail_marker), self.assertRaises(OSError):
                publish(task, answer)
            self.assertEqual(original.read_bytes(), b"x = 0\n")
            self.assertFalse(marker.exists())
            publish(task, answer)
            self.assertEqual(original.read_bytes(), b"x = 1\n")
            self.assertEqual(marker.read_bytes(), b"DONE\n")

    def make_task(self, root, name, kind):
        directory = root / name
        workspace = directory / "workspace"
        workspace.mkdir(parents=True)
        return Task(
            task_id=name, task_type=kind, instructions="Solve it.", reward=1,
            task_dir=directory, workspace_dir=workspace,
        )
