from collections.abc import Collection
from dataclasses import dataclass
from statistics import median

from .contracts import Task

MIN_CODING_TRIAL_UNITS = 40_000

@dataclass(frozen=True)
class Estimate:
    probability: float
    units: int
    seconds: float


@dataclass(frozen=True)
class Candidate:
    task: Task
    estimate: Estimate

    @property
    def priority(self) -> float:
        return self.task.reward * self.estimate.probability / max(1, self.estimate.units)


@dataclass(frozen=True)
class Allocation:
    units: int
    seconds: float


def calibrate_coding(initial: Estimate, successful_costs: list[int]) -> Estimate:
    if len(successful_costs) < 3:
        return initial
    # Leave room for an output reservation and for reviewing the full contract.
    units = min(initial.units, max(MIN_CODING_TRIAL_UNITS, 3 * int(median(successful_costs))))
    return Estimate(initial.probability, units, initial.seconds)


def estimate(task: Task, *, files: int, file_bytes: int, isolated: bool) -> Estimate:
    probability, units, seconds = {
        "math": (0.8, 30_000, 50),
        "search": (0.75, 32_000, 60),
        "data_analysis": (0.7, 45_000, 90),
        "coding": (0.6 if isolated else 0.0, 95_000, 160),
    }.get(task.task_type, (0.4, 45_000, 90))
    instruction_bytes = len(task.instructions.encode("utf-8"))
    size_penalty = min(45_000, instruction_bytes + 100 * files + file_bytes // 100)
    return Estimate(probability, units + size_penalty, seconds + min(30, files))


def select(
    candidates: list[Candidate], units: int | None, seconds: float, *, sequential: bool = False,
    small_coding_ids: Collection[str] = (),
) -> tuple[Candidate, Allocation] | None:
    feasible = [
        candidate for candidate in candidates
        if candidate.estimate.probability > 0
        and (units is None or candidate.estimate.units <= units)
        and candidate.estimate.seconds <= seconds
    ]
    if not feasible:
        if units is None or units < MIN_CODING_TRIAL_UNITS:
            return None
        feasible = [
            candidate for candidate in candidates
            if candidate.task.task_id in small_coding_ids
            and candidate.estimate.probability > 0
            and candidate.estimate.seconds <= seconds
        ]
        if not feasible:
            return None
    candidate = feasible[0] if sequential else max(feasible, key=lambda item: item.priority)
    allowance = candidate.estimate.units if units is None else min(candidate.estimate.units, units)
    return candidate, Allocation(allowance, candidate.estimate.seconds)
