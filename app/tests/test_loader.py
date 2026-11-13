import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.errors import TaskFormatError
from harness.loader import load_tasks, working_copy


class LoaderTests(unittest.TestCase):
    def test_task_types_malformed_task_and_private_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kinds = {"math", "search", "data_analysis", "coding", "unfamiliar"}
            for kind in kinds:
                directory = root / kind
                directory.mkdir()
                (directory / "task.json").write_text(json.dumps({
                    "task_id": kind, "task_type": kind, "instruction": "Solve it.",
                    "reward": 3, "expected": "SECRET", "private_solution": "SECRET",
                }), encoding="utf-8")
            broken = root / "broken"
            broken.mkdir()
            (broken / "task.json").write_text("{", encoding="utf-8")
            workspace = root / "coding" / "workspace"
            workspace.mkdir()
            (workspace / "input.txt").write_text("input", encoding="utf-8")
            (workspace.parent / "solution.txt").write_text("SECRET", encoding="utf-8")

            tasks, errors = load_tasks(root)
            self.assertEqual({task.task_type for task in tasks}, kinds)
            self.assertEqual([error.directory for error in errors], ["broken"])
            for task in tasks:
                self.assertNotIn("SECRET", task.model_dump_json())
                with working_copy(task) as copy:
                    expected = ["input.txt"] if task.task_type == "coding" else []
                    self.assertEqual(sorted(path.name for path in copy.iterdir()), expected)
                self.assertFalse(copy.exists())
            coding = next(task for task in tasks if task.task_type == "coding")
            with patch("harness.loader.MAX_WORKSPACE_BYTES", 4):
                with self.assertRaises(TaskFormatError), working_copy(coding):
                    self.fail("Oversized workspace was copied")
