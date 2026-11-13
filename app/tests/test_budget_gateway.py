import asyncio
import os
import unittest
from time import monotonic
from unittest.mock import patch

import httpx

from harness.budget import BudgetManager
from harness.config import DEFAULT_MODEL, budget_units
from harness.errors import BudgetExceeded, DeadlineExceeded, UsageUnknownError
from harness.history import History
from harness.llm import ModelGateway, normalize_usage


class BudgetTests(unittest.TestCase):
    def test_configuration(self):
        with patch.dict(os.environ, {"BUDGET": "120"}):
            self.assertEqual(budget_units("25"), 250)
            self.assertEqual(budget_units(), 1200)
        with patch.dict(os.environ, clear=True):
            self.assertIsNone(budget_units())
            budget = BudgetManager(budget_units(), monotonic() + 10)
            reservation = budget.reserve(1_000_001, 1, task_remaining=None)
            budget.settle(reservation, None)
            self.assertIsNone(budget.available_units())

    def test_reservation_settlement_and_deadline(self):
        budget = BudgetManager(900, monotonic() + 10)
        with self.assertRaises(BudgetExceeded):
            budget.reserve(100, 81, 900)
        reservation = budget.reserve(100, 80, 900)
        self.assertEqual(budget.available_units(), 0)
        usage = normalize_usage({"prompt_tokens": 100, "completion_tokens": 60}, "included")
        self.assertEqual(budget.settle(reservation, usage), 700)
        self.assertEqual(budget.available_units(), 200)
        with self.assertRaises(DeadlineExceeded):
            BudgetManager(900, monotonic() - 1).reserve(100, 60, 900)

    def test_usage_profiles(self):
        included = {
            "prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160,
            "completion_tokens_details": {"reasoning_tokens": 20},
        }
        separate = {
            "prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 160,
            "reasoning_tokens": 20,
        }
        self.assertEqual(normalize_usage(included, "included").cost_units, 700)
        self.assertEqual(normalize_usage(separate, "separate").cost_units, 700)
        with self.assertRaises(UsageUnknownError):
            normalize_usage(separate, "included")

    def test_contest_proxy_usage(self):
        # Anonymized token counts from the two live Qwen 3.8 probes.
        for prompt, completion, reasoning, cost in ((18, 2, 0, 38), (46, 24, 20, 286)):
            with self.subTest(reasoning=reasoning):
                usage = normalize_usage({
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "completion_tokens_details": {"reasoning_tokens": reasoning},
                    "total_tokens": prompt + completion,
                }, "included")
                self.assertEqual(usage.cost_units, cost)


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_attempt_timeout_charges_reservation_and_allows_next_request(self):
        sent = asyncio.Event()

        async def respond(request):
            if not sent.is_set():
                sent.set()
                await asyncio.Future()
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "4"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            })

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="https://proxy.invalid/v1/"
        ) as client:
            budget = BudgetManager(2000, monotonic() + 10)
            gateway = ModelGateway(client, budget, DEFAULT_MODEL)
            with self.assertRaises(TimeoutError):
                async with asyncio.timeout(0.1):
                    await gateway.generate([], 20, 2000)
            self.assertTrue(sent.is_set())
            charged = budget.estimated_units
            self.assertGreater(charged, 200)
            self.assertEqual(budget.available_units(), 2000 - charged)
            result = await gateway.generate([], 20, 2000)
            self.assertEqual(result.message["content"], "4")
            self.assertEqual(budget.spent_units, charged + 20)
            self.assertEqual(budget.estimated_units, charged)

    async def test_deadline_cancels_in_flight_request(self):
        sent = asyncio.Event()

        async def respond(request):
            sent.set()
            await asyncio.sleep(10)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="https://proxy.invalid/v1/"
        ) as client:
            budget = BudgetManager(2000, monotonic() + 0.25)
            gateway = ModelGateway(client, budget, DEFAULT_MODEL)
            with self.assertRaises(TimeoutError):
                await gateway.generate([], 20, 2000)
            self.assertTrue(sent.is_set())
            self.assertGreater(budget.estimated_units, 0)
            with self.assertRaises(DeadlineExceeded):
                await gateway.generate([], 20, 2000)

    async def test_retry_estimate_and_quota(self):
        responses = iter([
            httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}}),
            httpx.Response(200, json={"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]}),
            httpx.Response(429, json={"error": {"code": "insufficient_quota"}}),
        ])
        requests = []
        totals = []
        events = []
        history = History(lambda event, **fields: events.append((event, fields)), detailed=True)
        attempt = history.scope("task", selected=True)
        attempt.step = 2

        def respond(request):
            requests.append(request.url.path)
            return next(responses)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="https://proxy.invalid/v1/"
        ) as client:
            budget = BudgetManager(2000, monotonic() + 10, on_usage=totals.append)
            gateway = ModelGateway(client, budget, DEFAULT_MODEL)
            result = await gateway.generate([], 20, 2000, history=attempt)
            self.assertEqual(result.requests, 2)
            self.assertIsNone(result.usage)
            self.assertEqual(budget.estimated_units, 2 * result.cost_units)
            self.assertEqual(len(totals), 2)
            self.assertEqual(totals[1]["estimated_output_tokens"], 40)
            self.assertEqual(totals[1]["estimated_input_tokens"], totals[1]["input_tokens"])
            self.assertEqual(totals[1]["total_tokens"], totals[1]["input_tokens"] + 40)
            with self.assertRaises(BudgetExceeded):
                await gateway.generate([], 20, 2000, history=attempt)
            with self.assertRaises(BudgetExceeded):
                await gateway.generate([], 20, 2000, history=attempt)
            self.assertEqual(requests, ["/v1/chat/completions"] * 3)
            self.assertEqual(budget.spent_units, 3 * result.cost_units)
            self.assertEqual(len(totals), 3)
            self.assertEqual(totals[-1], budget.usage_totals())
        finished = [fields for event, fields in events if event == "model_request_finished"]
        self.assertEqual([row["request"] for row in finished], [1, 2, 1])
        self.assertEqual([row["http_status"] for row in finished], [429, 200, 429])
        self.assertEqual(finished[1]["finish_reason"], "stop")
        self.assertTrue(all(row["estimated"] and row["seconds"] >= 0 for row in finished))
        self.assertEqual(events[-1][0], "model_request_skipped")
        self.assertTrue(all(fields["step"] == 2 and fields["attempt_id"] == attempt.attempt_id
                            for _, fields in events))
