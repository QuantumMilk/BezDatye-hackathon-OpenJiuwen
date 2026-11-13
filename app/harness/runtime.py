import asyncio
import json
from collections.abc import AsyncIterator, Callable
from uuid import uuid4

import httpx
from openjiuwen_runtime.service import AgentApp, QueryRequest

Message = dict[str, object]
Solve = Callable[[list[Message], asyncio.Event], AsyncIterator[Message]]


class AgentRuntime:
    """Run a solver through AgentApp's query lifecycle in the same process."""

    def __init__(self, solve: Solve) -> None:
        app = AgentApp("ContestAgent")

        @app.query
        async def query(messages: list[Message], request: QueryRequest, cancel: asyncio.Event):
            async for event in solve(messages, cancel):
                yield event, False

        self._transport = httpx.ASGITransport(app=app.app)

    async def run(self, messages: list[Message]) -> list[Message]:
        async with httpx.AsyncClient(
            transport=self._transport, base_url="http://agent-runtime"
        ) as client:
            response = await client.post(
                "/query",
                json={"conversation_id": uuid4().hex, "messages": messages},
            )
        response.raise_for_status()
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        if any(event.get("type") == "error" for event in events):
            raise RuntimeError("AgentApp query failed")
        return events
