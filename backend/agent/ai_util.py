import contextlib
import json
from collections.abc import AsyncGenerator, Collection, Sequence
from typing import ClassVar, Protocol

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
                ai.ToolRunner() as tr,
            ):
                final = False
                async for event in stream:
                    yield event

                    if isinstance(event, ai.events.ToolEnd):
                        if final or event.tool_call.tool_name in self.eager_tools:
                            tool = context.resolve(event.tool_call)
                            tr.schedule(tool)
                    elif isinstance(event, ai.events.StreamEnd):
                        tr.discard_all(
                            except_=context.resolve(event.message.tool_calls)
                        )
                        context.add(event.message)
                    elif isinstance(event, ai.events.StreamStart):
                        final = True

                async for event in tr.events():
                    yield event

                context.add(tr.get_tool_message())
