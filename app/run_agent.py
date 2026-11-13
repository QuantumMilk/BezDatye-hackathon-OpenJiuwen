import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from time import monotonic

import httpx

from harness.agent import SolveResult, local_search, solve
from harness.budget import BudgetManager
from harness.config import DEFAULT_MODEL, MODELS, budget_units
from harness.errors import DeadlineExceeded
from harness.execution import run_isolated
from harness.history import History
from harness.llm import ModelGateway
from harness.loader import load_tasks, working_copy
from harness.output import publish, validate_candidate
from harness.planner import Candidate, calibrate_coding, estimate, select
from harness.tools import WorkspaceTools, resolve_workspace_path


def event(name: str, **fields) -> None:
    print(json.dumps({"event": name, **fields}, ensure_ascii=False), flush=True)


async def run(args: argparse.Namespace) -> int:
    started = monotonic()
    history = History(event, detailed=args.attempt_history, secret=os.getenv("LLM_API_KEY", ""))
    budget = BudgetManager(
        budget_units(args.budget), started + 3590,
        on_usage=lambda totals: event("model_usage", **totals),
    )
    tasks, errors = load_tasks(args.tasks_dir.resolve(strict=True))
    for error in errors:
        history.scope(None).record("task_decision", status="skipped", reason=error.reason,
                                   directory=history.fragment(error.directory))
        event("task_finished", directory=history.clean(error.directory), status="error", reason=error.reason)
    os.environ.setdefault("OPENJIUWEN_RUNTIME_LOG_DIR", gettempdir())

    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            handler.setStream(sys.stderr)

    with TemporaryDirectory(prefix="contest-probe-") as temp:
        isolation = (await run_isolated(Path(temp), "probe", timeout=3)).status == "passed"
    candidates = []
    small_coding = {}
    coding_costs = []
    for task in tasks:
        try:
            if resolve_workspace_path(task.task_dir, "answer.txt").exists():
                history.scope(task.task_id).record("task_decision", status="skipped", reason="Answer already present")
                event("task_finished", task_id=task.task_id, status="skipped", reason="Answer already present")
                continue
            tools = WorkspaceTools(task.workspace_dir, coding=task.task_type == "coding")
            if task.task_type == "search":
                local_started = monotonic()
                answer = local_search(task, tools)
                if answer is not None:
                    try:
                        validated = await validate_candidate(
                            task, task.workspace_dir, answer, evidence=True, timeout=0,
                        )
                    except ValueError:
                        pass
                    else:
                        publish(task, validated)
                        history.scope(task.task_id).record("solve_route", status="saved", reason="local search")
                        event(
                            "task_finished", task_id=task.task_id, status="saved", reason="local search",
                            cost_units=0, estimated_units=0, seconds=round(monotonic() - local_started, 3),
                        )
                        continue
            files = tools.list_files(limit=200) if task.workspace_dir.exists() else []
            file_bytes = sum(file["bytes"] for file in files)
            prediction = estimate(task, files=len(files), file_bytes=file_bytes, isolated=isolation)
            if task.task_type == "coding" and len(files) <= 5 and file_bytes <= 16_384:
                small_coding[task.task_id] = prediction
            candidates.append(Candidate(task, prediction))
            history.scope(task.task_id).record(
                "task_decision", status="estimated", reward=task.reward, probability=prediction.probability,
                estimated_units=prediction.units, estimated_seconds=prediction.seconds, isolation=isolation,
            )
            event(
                "task_estimated", task_id=task.task_id, probability=prediction.probability,
                units=prediction.units, seconds=prediction.seconds, reward=task.reward,
            )
        except (OSError, ValueError) as error:
            history.scope(task.task_id).record("task_decision", status="skipped", reason=type(error).__name__)
            event("task_finished", task_id=task.task_id, status="error", reason=type(error).__name__)

    base = os.environ["LLM_BASE_URL"].rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    async with httpx.AsyncClient(
        base_url=base + "/",
        headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"]},
        trust_env=False,
    ) as client:
        gateway = ModelGateway(client, budget, args.model, args.usage_profile)
        while candidates:
            try:
                remaining = budget.remaining_time()
            except DeadlineExceeded:
                break
            selection = select(
                candidates, budget.available_units(), remaining, sequential=args.sequential,
                small_coding_ids=small_coding,
            )
            if selection is None:
                break
            candidate, allowance = selection
            candidates.remove(candidate)
            task = candidate.task
            attempt_history = history.scope(task.task_id, selected=True)
            attempt_history.record(
                "task_decision", status="selected", reason="sequential" if args.sequential else "expected_reward",
                priority=candidate.priority, units=allowance.units, seconds=allowance.seconds,
                run_remaining_units=budget.available_units(), run_remaining_seconds=remaining,
            )
            attempt_started = monotonic()
            previous_spend = budget.spent_units
            previous_estimated = budget.estimated_units
            event("task_selected", task_id=task.task_id, priority=candidate.priority,
                  units=allowance.units, seconds=allowance.seconds)
            try:
                with working_copy(task) as workspace:
                    from harness.runtime import AgentRuntime

                    tools = WorkspaceTools(
                        workspace, coding=task.task_type == "coding", can_execute=isolation
                    )
                    result = None

                    async def attempt(messages, cancel):
                        nonlocal result
                        try:
                            result = await solve(task, tools, gateway, allowance, attempt_history)
                        except Exception as error:
                            attempt_history.record("solve_error", error_type=type(error).__name__,
                                                   exception=attempt_history.fragment(str(error)))
                            result = SolveResult("error", type(error).__name__)
                        yield {"status": result.status}

                    await AgentRuntime(attempt).run([{"role": "user", "content": task.instructions}])
                    if result is None:
                        raise RuntimeError("Runtime returned no result")
                    if result.answer is not None:
                        attempt_history.record("publication_started", changes=len(result.answer.changes))
                        try:
                            publish(task, result.answer)
                        except (OSError, ValueError, RuntimeError) as error:
                            attempt_history.record(
                                "publication_finished", status="error", answer_written=False,
                                error_type=type(error).__name__, exception=attempt_history.fragment(str(error)),
                            )
                            raise
                        attempt_history.record("publication_finished", status="saved", answer_written=True)
                        result = SolveResult("saved", result.reason)
                    else:
                        attempt_history.record("publication_skipped", reason="No accepted answer")
            except (OSError, ValueError, RuntimeError) as error:
                result = SolveResult("error", type(error).__name__)
            reason = attempt_history.fragment(result.reason)["text"]
            cost = budget.spent_units - previous_spend
            if result.status == "saved" and task.task_id in small_coding and cost > 0:
                coding_costs.append(cost)
                candidates = [
                    Candidate(item.task, calibrate_coding(small_coding[item.task.task_id], coding_costs))
                    if item.task.task_id in small_coding else item
                    for item in candidates
                ]
            attempt_history.record(
                "attempt_finished", status=result.status, reason=reason,
                cost_units=cost,
                estimated_units=budget.estimated_units - previous_estimated,
                seconds=round(monotonic() - attempt_started, 3),
            )
            event(
                "task_finished", task_id=task.task_id, status=result.status, reason=reason,
                cost_units=cost,
                estimated_units=budget.estimated_units - previous_estimated,
                seconds=round(monotonic() - attempt_started, 3),
            )

    for candidate in candidates:
        prediction = candidate.estimate
        reasons = []
        remaining_units = budget.available_units()
        if prediction.probability == 0:
            reasons.append("isolation_unavailable")
        if remaining_units is not None and prediction.units > remaining_units:
            reasons.append("budget_limit")
        if prediction.seconds > budget.deadline - monotonic():
            reasons.append("time_limit")
        history.scope(candidate.task.task_id).record("task_decision", status="skipped", reasons=reasons)
        event(
            "task_finished", task_id=candidate.task.task_id, status="skipped",
            reason="Isolation unavailable" if candidate.estimate.probability == 0 else "Budget or time limit",
        )
    event("run_finished", **budget.usage_totals(), seconds=round(monotonic() - started, 3))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the contest agent")
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--budget")
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--usage-profile", choices=("included", "separate"), default="included")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--attempt-history", action="store_true", help="Stream detailed local JSONL to stdout")
    args = parser.parse_args()
    if args.model is None:
        args.model = os.getenv("MODEL_NAME") or DEFAULT_MODEL
        if args.model not in MODELS:
            parser.error(f"Unsupported MODEL_NAME: {args.model!r}")
    try:
        raise SystemExit(asyncio.run(run(args)))
    except (KeyError, OSError, ValueError) as error:
        event("run_failed", reason=type(error).__name__)
        raise SystemExit(1) from None
