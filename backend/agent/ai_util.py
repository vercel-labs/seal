import json
from collections.abc import Sequence

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
