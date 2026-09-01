import asyncio
import contextvars
import json
from collections.abc import AsyncIterable, Collection, Sequence
from typing import Any, Self

import ai


def recover_partial_messages(
    input_messages: list[ai.messages.Message],
    events: Sequence[ai.events.Event],
) -> list[ai.messages.Message]:
    """Merge streamed messages and close tool calls interrupted mid-turn."""
    hydrator = ai.events.MessageHydrator()
    for event in events:
        hydrator.feed(event)

    messages = list(input_messages) + [
        message for message in hydrator.messages if message.role != "internal"
    ]

    answered = {
        result.tool_call_id
        for message in messages
        if message.role == "tool"
        for result in message.tool_results
    }
    pending: list[ai.messages.ToolCallPart] = []
    normalized: list[ai.messages.Message] = []
    for message in messages:
        parts: list[ai.messages.Part] = []
        for part in message.parts:
            if isinstance(part, ai.messages.ToolCallPart):
                try:
                    json.loads(part.tool_args)
                except (json.JSONDecodeError, TypeError):
                    part = part.model_copy(update={"tool_args": "{}"})
                if part.tool_call_id not in answered:
                    pending.append(part)
            parts.append(part)
        normalized.append(message.model_copy(update={"parts": parts}))
    if pending:
        normalized.append(
            ai.tool_message(
                *(
                    ai.tool_result_part(
                        part.tool_call_id,
                        tool_name=part.tool_name,
                        result="Interrupted by user",
                        is_error=True,
                    )
                    for part in pending
                )
            )
        )
    return normalized


current_tool_call_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_tool_call_id", default=None
)


class TrackingToolRunner(ai.ToolRunner):
    def __init__(self) -> None:
        super().__init__()
        self.id_to_task: dict[str, asyncio.Task[ai.events.ToolCallResult]] = {}
        self.task_to_id: dict[asyncio.Task[ai.events.ToolCallResult], str] = {}

    def discard(self, task: asyncio.Task[ai.events.ToolCallResult]) -> None:
        if task in self.task_to_id:
            del self.id_to_task[self.task_to_id[task]]
            del self.task_to_id[task]
        super().discard(task)

    def discard_except(self, ok_calls: Collection[ai.ToolCall]) -> None:
        ok_ids = {tc.id for tc in ok_calls}
        for id, task in list(self.id_to_task.items()):
            if id not in ok_ids:
                self.discard(task)

    def schedule(
        self, tc: ai.agents.agent.ToolCallCallable
    ) -> asyncio.Task[ai.events.ToolCallResult]:
        id = getattr(tc, "id", None)
        if id and id in self.id_to_task:
            return self.id_to_task[id]

        async def _run() -> ai.events.ToolCallResult:
            if id:
                current_tool_call_id.set(id)
            return await tc()

        task = super().schedule(_run)
        if id:
            self.id_to_task[id] = task
            self.task_to_id[task] = id
        return task


class SpeculativeToolRunner(TrackingToolRunner):
    def __init__(
        self,
        *,
        tool_stream: AsyncIterable[ai.ToolCall],
    ) -> None:
        super().__init__()
        self.tool_stream = tool_stream

    async def __aenter__(self) -> Self:
        res = await super().__aenter__()
        self.worker = asyncio.create_task(self.watcher())
        return res

    async def __aexit__(self, *args: Any) -> None:
        self.worker.cancel()
        return await super().__aexit__(*args)

    async def watcher(self) -> None:
        async for tool_call in self.tool_stream:
            self.schedule(tool_call)
