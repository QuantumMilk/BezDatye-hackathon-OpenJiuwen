import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Literal

import httpx
from pydantic import ValidationError

from .budget import BudgetManager
from .config import MODELS
from .contracts import Usage
from .errors import BudgetExceeded, DeadlineExceeded, ProviderError, UsageUnknownError
from .history import AttemptHistory

UsageProfile = Literal["included", "separate"]
QUOTA_CODES = {"insufficient_quota", "quota_exceeded", "budget_exceeded"}
RATE_LIMIT_CODES = {"rate_limit_exceeded", "too_many_requests"}


def normalize_usage(raw: object, profile: UsageProfile) -> Usage:
    if not isinstance(raw, Mapping):
        raise UsageUnknownError("Missing model usage")
    details = raw.get("completion_tokens_details")
    if details is not None and not isinstance(details, Mapping):
        raise UsageUnknownError("Invalid completion token details")
    nested = details.get("reasoning_tokens") if details is not None else None
    separate = raw.get("reasoning_tokens")
    if nested is not None and separate is not None and nested != separate:
        raise UsageUnknownError("Conflicting reasoning counts")
    reasoning = separate if separate is not None else nested
    try:
        usage = Usage(
            input_tokens=raw.get("prompt_tokens"),
            total_output_tokens=raw.get("completion_tokens"),
            reasoning_tokens=reasoning,
        )
    except ValidationError:
        raise UsageUnknownError("Invalid model usage") from None

    if profile == "separate":
        if usage.reasoning_tokens is None:
            raise UsageUnknownError("Missing separate reasoning count")
        usage.total_output_tokens += usage.reasoning_tokens
    elif usage.reasoning_tokens is not None and usage.reasoning_tokens > usage.total_output_tokens:
        raise UsageUnknownError("Reasoning exceeds completion token count")
    total = raw.get("total_tokens")
    if total is not None and (
        type(total) is not int or total != usage.input_tokens + usage.total_output_tokens
    ):
        raise UsageUnknownError("Inconsistent total token count")
    return usage


@dataclass(frozen=True)
class ModelReply:
    message: dict
    usage: Usage | None
    cost_units: int
    requests: int


class ModelGateway:
    def __init__(
        self,
        client: httpx.AsyncClient,
        budget: BudgetManager,
        model: str,
        usage_profile: UsageProfile = "included",
    ) -> None:
        if model not in MODELS:
            raise ValueError("Unsupported contest model")
        self.client = client
        self.budget = budget
        self.model = model
        self.usage_profile = usage_profile

    async def generate(
        self,
        messages: list[dict],
        max_output: int,
        task_remaining: int | None,
        tools: list[dict] | None = None,
        *,
        history: AttemptHistory | None = None,
    ) -> ModelReply:
        payload = {"model": self.model, "messages": messages, "max_tokens": max_output}
        if tools:
            payload["tools"] = tools
        # UTF-8 bytes plus framing are a conservative local estimate.
        estimated_input = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 64
        for attempt in (1, 2):
            try:
                timeout = min(60, self.budget.remaining_time())
                reservation = self.budget.reserve(estimated_input, max_output, task_remaining)
            except (BudgetExceeded, DeadlineExceeded) as error:
                if history:
                    history.record("model_request_skipped", model=self.model, request=attempt,
                                   reason=type(error).__name__)
                raise
            started = monotonic()
            if history:
                history.record("model_request_started", model=self.model, request=attempt, timeout=timeout)
            try:
                async with asyncio.timeout(timeout):
                    response = await self.client.post(
                        "chat/completions",
                        json=payload,
                        timeout=httpx.Timeout(timeout, connect=min(10, timeout)),
                    )
            except (TimeoutError, httpx.TransportError, asyncio.CancelledError) as error:
                cost = self.budget.settle(reservation, None)
                if history:
                    history.record(
                        "model_request_finished", model=self.model, request=attempt, status="error",
                        seconds=round(monotonic() - started, 6), reason=type(error).__name__,
                        input_tokens=reservation.estimated_input, output_tokens=reservation.max_output,
                        cost_units=cost, estimated=True,
                    )
                raise

            try:
                body = response.json()
            except ValueError:
                body = {}
            if not isinstance(body, dict):
                body = {}
            try:
                usage = normalize_usage(body.get("usage"), self.usage_profile)
            except UsageUnknownError:
                usage = None
            cost = self.budget.settle(reservation, usage)
            error = body.get("error")
            choices = body.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else None
            message = choice.get("message") if isinstance(choice, dict) else None
            if history:
                finish = choice.get("finish_reason") if isinstance(choice, dict) else None
                history.record(
                    "model_request_finished", model=self.model, request=attempt,
                    status="ok" if response.is_success and not error and isinstance(message, dict) else "error",
                    http_status=response.status_code, seconds=round(monotonic() - started, 6),
                    finish_reason=history.fragment(finish)["text"] if isinstance(finish, str) else None,
                    input_tokens=usage.input_tokens if usage else reservation.estimated_input,
                    output_tokens=usage.total_output_tokens if usage else reservation.max_output,
                    reasoning_tokens=usage.reasoning_tokens if usage else None,
                    cost_units=cost, estimated=usage is None,
                )
            if not response.is_success or error:
                code = error.get("code") if isinstance(error, dict) else None
                if response.status_code == 402 or (isinstance(code, str) and code in QUOTA_CODES):
                    self.budget.exhausted = True
                    raise BudgetExceeded("Proxy quota exhausted")
                if response.status_code == 429 and isinstance(code, str) and code in RATE_LIMIT_CODES:
                    if attempt == 1 and self.budget.remaining_time() > 1:
                        if task_remaining is not None:
                            task_remaining -= cost
                        await asyncio.sleep(1)
                        continue
                elif response.status_code in (401, 403, 429):
                    self.budget.exhausted = True
                raise ProviderError(f"Proxy request failed (HTTP {response.status_code})")

            if not isinstance(choice, dict) or "message" not in choice:
                raise ProviderError("Proxy returned no model message")
            if not isinstance(message, dict):
                raise ProviderError("Proxy returned an invalid model message")
            return ModelReply(message, usage, cost, attempt)

        raise AssertionError("Proxy attempts exhausted")
