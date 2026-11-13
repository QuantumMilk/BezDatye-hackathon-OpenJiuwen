"""Exercise AgentApp, a Core tool, and a measured model request."""

import asyncio
import json
import os
from tempfile import gettempdir
from time import monotonic

import httpx

from harness.budget import BudgetManager
from harness.config import DEFAULT_MODEL
from harness.llm import ModelGateway


async def main() -> None:
    os.environ.setdefault("OPENJIUWEN_RUNTIME_LOG_DIR", gettempdir())
    from harness.runtime import AgentRuntime
    from openjiuwen.core.common.logging.log_config import configure_log_config

    configure_log_config({
        "output": ["console"],
        "interface_output": ["console"],
        "performance_output": ["console"],
    })
    from openjiuwen.core.foundation.tool import tool

    @tool
    def add(left: int, right: int) -> int:
        """Add two integers."""
        return left + right

    requests = []

    def mock_proxy(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "42"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2},
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(mock_proxy), base_url="https://proxy.invalid/v1/"
    ) as proxy:
        budget = BudgetManager(100_000, monotonic() + 10)
        gateway = ModelGateway(proxy, budget, DEFAULT_MODEL)

        async def solve(messages, cancel):
            assert not cancel.is_set()
            result = await add.invoke({"left": 19, "right": 23})
            reply = await gateway.generate(
                [*messages, {"role": "user", "content": str(result)}],
                max_output=64,
                task_remaining=budget.available_units(),
            )
            yield {"tool_result": result, "usage": reply.usage.model_dump()}

        events = await AgentRuntime(solve).run([{"role": "user", "content": "19 + 23?"}])
    assert len(requests) == 1
    assert requests[0]["messages"][-1]["content"] == "42"
    assert budget.spent_units == 32
    assert events == [{
        "tool_result": 42,
        "usage": {
            "input_tokens": 12,
            "total_output_tokens": 2,
            "reasoning_tokens": None,
            "cost_units": 32,
        },
    }]
    print(json.dumps(events[0]))


if __name__ == "__main__":
    asyncio.run(main())
