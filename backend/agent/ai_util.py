import asyncio
import contextlib
import contextvars
from collections.abc import AsyncGenerator, Collection, Sequence
from typing import ClassVar, Protocol

import ai

current_tool_call_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_tool_call_id", default=None
)


class Streamer(Protocol):
    def __call__(
        self, *, context: ai.Context
    ) -> contextlib.AbstractAsyncContextManager[
        AsyncGenerator[ai.events.AgentEvent]
    ]: ...


class DurableAgent(ai.Agent):
    # We require the loop to run in lockstep with the client code, so that
    # tool streams are always sent before the next llm_step invocation.
    LOOP_BUFFER = 0

    TOOLS: ClassVar[list[ai.AgentTool]] = []

    def __init__(
        self,
        *,
        tools: Sequence[ai.AgentTool | ai.Tool] | None = None,
        streamer: Streamer,
        eager_tools: Collection[str] = (),
    ) -> None:
        super().__init__(tools=tools)
        self.streamer = streamer
        self.eager_tools = frozenset(eager_tools)

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        while context.keep_running():
            async with (
                self.streamer(context=context) as stream,
                TrackingToolRunner() as tr,
            ):
                final = False
                async for event in stream:
                    yield event

                    if isinstance(event, ai.events.ToolEnd):
                        if final or event.tool_call.tool_name in self.eager_tools:
                            tool = context.resolve(event.tool_call)
                            tr.schedule(tool)
                    elif isinstance(event, ai.events.StreamEnd):
                        tr.discard_except(context.resolve(event.message.tool_calls))
                        context.add(event.message)
                    elif isinstance(event, ai.events.StreamStart):
                        final = True

                async for event in tr.events():
                    yield event

                context.add(tr.get_tool_message())


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
