import unittest
from pathlib import Path

from harness.contracts import Task
from harness.planner import Candidate, Estimate, calibrate_coding, select


class PlannerTests(unittest.TestCase):
    def test_small_coding_estimate_uses_observed_cost_with_headroom(self):
        initial = Estimate(0.6, 96_000, 163)
        self.assertEqual(calibrate_coding(initial, [8_000, 10_000]), initial)
        revised = calibrate_coding(initial, [8_000, 10_000, 12_000])
        self.assertEqual(revised, Estimate(0.6, 40_000, 163))
        self.assertEqual(calibrate_coding(initial, [60_000, 70_000, 80_000]), initial)

    def test_rank_and_readmit_after_spend(self):
        def candidate(name, reward, cost, seconds):
            task = Task(task_id=name, task_type="math", instructions="Solve", reward=reward,
                        task_dir=Path(name), workspace_dir=Path(name) / "workspace")
            return Candidate(task, Estimate(0.8, cost, seconds))

        cheap = candidate("cheap", 4, 20_000, 30)
        expensive = candidate("expensive", 6, 45_000, 50)
        pending = [expensive, cheap]
        first, allowance = select(pending, 70_000, 100)
        self.assertEqual(first, cheap)
        self.assertEqual(select([expensive], 70_000 - allowance.units, 70)[0], expensive)
        self.assertIsNone(select([expensive], 44_999, 70))
        self.assertIsNone(select([expensive], 50_000, 49))
        self.assertEqual(select(pending, 70_000, 100, sequential=True)[0], expensive)

    def test_small_coding_can_try_with_remaining_budget(self):
        task = Task(task_id="small", task_type="coding", instructions="Repair", reward=5,
                    task_dir=Path("small"), workspace_dir=Path("small/workspace"))
        candidate = Candidate(task, Estimate(0.6, 96_000, 163))
        self.assertIsNone(select([candidate], 60_000, 200))
        self.assertIsNone(select([candidate], 39_999, 200, small_coding_ids={"small"}))
        selected, allocation = select([candidate], 60_000, 200, small_coding_ids={"small"})
        self.assertEqual((selected, allocation.units), (candidate, 60_000))
