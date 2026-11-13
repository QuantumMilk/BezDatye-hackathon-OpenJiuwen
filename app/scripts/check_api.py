import argparse
import asyncio
import json
import os
from time import monotonic

import httpx

from harness.budget import BudgetManager
from harness.config import DEFAULT_MODEL, MODELS, budget_units
from harness.errors import BudgetExceeded, DeadlineExceeded, ProviderError
from harness.llm import ModelGateway


async def main(args: argparse.Namespace) -> None:
    total = budget_units(args.budget)
    base = os.environ["LLM_BASE_URL"].rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    budget = BudgetManager(total, monotonic() + 60)
    async with httpx.AsyncClient(
        base_url=base + "/",
        headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"]},
        trust_env=False,
    ) as client:
        gateway = ModelGateway(client, budget, args.model, args.usage_profile)
        result = await gateway.generate(
            [{"role": "user", "content": "Reply with just 4."}],
            max_output=64,
            task_remaining=total,
        )
    print(json.dumps({
        "usage": result.usage.model_dump() if result.usage is not None else None,
        "cost_units": result.cost_units,
        "source": "reported" if result.usage is not None else "estimated",
        "requests": result.requests,
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Qwen proxy usage")
    parser.add_argument("--budget")
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--usage-profile", choices=["included", "separate"], default="included")
    try:
        asyncio.run(main(parser.parse_args()))
    except (KeyError, ValueError, BudgetExceeded, DeadlineExceeded, ProviderError) as error:
        parser.exit(1, f"{error}\n")
    except (httpx.TransportError, TimeoutError):
        parser.exit(1, "Proxy transport failure\n")
