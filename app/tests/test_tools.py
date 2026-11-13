import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.errors import ToolPolicyError
from harness.execution import ExecutionResult, run_isolated
from harness.tools import WorkspaceTools


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_paths_tests_and_disabled_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = WorkspaceTools(root, coding=True)
            tests = root / "test_mylib.py"
            tests.write_text("public tests", encoding="utf-8")
            tools.write_file("solution.py", "print(42)")
            self.assertEqual(tools.read_file("solution.py"), "print(42)")
            self.assertEqual(tools.search_text("42")[0]["path"], "solution.py")
            for path in ("../other.py", "C:/other.py", "/tmp/other.py", "test_mylib.py"):
                with self.subTest(path=path), self.assertRaises(ToolPolicyError):
                    tools.write_file(path, "modified")
            self.assertEqual(tests.read_text(encoding="utf-8"), "public tests")
            with self.assertRaises(ToolPolicyError):
                await tools.execute("python", "print(42)")
            tools.can_execute = True
            with patch("harness.tools.run_isolated", return_value=ExecutionResult("unavailable")):
                await tools.execute("python", "print(42)")
            self.assertFalse(tools.can_execute)

    @unittest.skipUnless(sys.platform == "linux", "Linux isolation required")
    async def test_isolated_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            work = root / "workspace"
            work.mkdir()
            (root / "other.txt").write_text("secret", encoding="utf-8")
            (work / "linked.txt").symlink_to(root / "other.txt")
            (work / "test_mylib.py").write_text(
                "import unittest\n"
                "class Public(unittest.TestCase):\n"
                "    def test_math(self): self.assertEqual(2 + 2, 4)\n"
                "unittest.main()\n",
                encoding="utf-8",
            )
            self.assertEqual((await run_isolated(work, "probe")).status, "passed")
            tools = WorkspaceTools(work, coding=True, can_execute=True)
            with self.assertRaises(ToolPolicyError):
                tools.read_file("linked.txt")
            self.assertEqual((await run_isolated(work, "tests")).status, "passed")
            with patch.dict(os.environ, {"LLM_API_KEY": "sentinel"}):
                result = await run_isolated(
                    work, "python", "import os; print(os.getenv('LLM_API_KEY'))"
                )
            self.assertEqual(result.output.strip(), "None")
            for source in (
                "open('../other.txt').read()",
                "open('linked.txt').read()",
                "open('/proc/1/environ').read()",
                "open('test_mylib.py', 'w')",
                "import os; os.fork()",
                "import socket; socket.socket()",
            ):
                with self.subTest(source=source):
                    result = await run_isolated(work, "python", source)
                    self.assertEqual(result.status, "failed")
                    self.assertIn("PermissionError", result.output)
            output = await run_isolated(work, "python", "print('x' * 10000)", output_limit=128)
            self.assertEqual((output.status, len(output.output)), ("output_limit", 128))
            timed = await run_isolated(work, "python", "while True: pass", timeout=0.3)
            self.assertEqual(timed.status, "timeout")
