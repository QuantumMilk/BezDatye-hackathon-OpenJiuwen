from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from uuid import uuid4

from .contracts import Usage
from .errors import BudgetExceeded, DeadlineExceeded


@dataclass(frozen=True)
class Reservation:
    id: str
    estimated_input: int
    max_output: int

    @property
    def units(self) -> int:
        return self.estimated_input + 10 * self.max_output


class BudgetManager:
    """Track reservations and settlements in the runner's event loop."""

    def __init__(
        self,
        total_units: int | None,
        deadline_monotonic: float,
        *,
        on_usage: Callable[[dict[str, int]], None] | None = None,
    ) -> None:
        self.total_units = total_units
        self.deadline = deadline_monotonic
        self.spent_units = 0
        self.estimated_units = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.estimated_input_tokens = 0
        self.estimated_output_tokens = 0
        self._on_usage = on_usage
        self.exhausted = False
        self._active: dict[str, int] = {}

    def available_units(self) -> int | None:
        if self.total_units is None:
            return None
        return self.total_units - self.spent_units - sum(self._active.values())

    def remaining_time(self) -> float:
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise DeadlineExceeded("Run deadline reached")
        return remaining

    def reserve(self, estimated_input: int, max_output: int, task_remaining: int | None) -> Reservation:
        self.remaining_time()
        if estimated_input < 0 or max_output <= 0:
            raise ValueError("Invalid token reservation")
        units = estimated_input + 10 * max_output
        limits = (task_remaining, self.available_units())
        if self.exhausted or any(limit is not None and units > limit for limit in limits):
            raise BudgetExceeded("Model request exceeds the remaining budget")
        reservation = Reservation(uuid4().hex, estimated_input, max_output)
        self._active[reservation.id] = units
        return reservation

    def settle(self, reservation: Reservation, usage: Usage | None) -> int:
        reserved = self._active.pop(reservation.id)
        units = usage.cost_units if usage is not None else reserved
        if usage is None:
            input_tokens = reservation.estimated_input
            output_tokens = reservation.max_output
            self.estimated_units += units
            self.estimated_input_tokens += input_tokens
            self.estimated_output_tokens += output_tokens
        else:
            input_tokens = usage.input_tokens
            output_tokens = usage.total_output_tokens
        self.spent_units += units
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if self._on_usage is not None:
            self._on_usage(self.usage_totals())
        return units

    def usage_totals(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cost_units": self.spent_units,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_units": self.estimated_units,
        }

    def release(self, reservation: Reservation) -> None:
        del self._active[reservation.id]
