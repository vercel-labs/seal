"""Round-trips for hidden ai SDK message fields used during replay."""

from __future__ import annotations

import ai
import ai.types.messages as messages_

from agent import turn


def _assistant_with_tool_calls() -> messages_.Message:
    return messages_.Message(
        role="assistant",
        parts=[
            messages_.TextPart(text="running it"),
            messages_.ToolCallPart(
                tool_call_id="tc-1", tool_name="bash", tool_args='{"command": "ls"}'
            ),
            messages_.ToolCallPart(
                tool_call_id="tc-2",
                tool_name="web_fetch",
                tool_args='{"url": "https://example.com"}',
            ),
        ],
    )


def test_round_trip_preserves_replay_and_cached_result() -> None:
    cached = ai.tool_result_part("tc-1", tool_name="bash", result="file.txt")
    message = _assistant_with_tool_calls()
    message = message.model_copy(
        update={
            "replay": True,
            "parts": [
                part.model_copy(update={"cached_result": cached})
                if isinstance(part, messages_.ToolCallPart)
                and part.tool_call_id == "tc-1"
                else part
                for part in message.parts
            ],
        }
    )

    restored = turn._load_message(turn._dump_message(message))

    assert restored.replay is True
    by_id = {part.tool_call_id: part for part in restored.tool_calls}
    assert by_id["tc-1"].cached_result is not None
    assert by_id["tc-1"].cached_result.result == "file.txt"
    assert by_id["tc-2"].cached_result is None
    assert restored.model_dump(mode="json") == message.model_dump(mode="json")


def test_round_trip_of_plain_message_stays_plain() -> None:
    message = _assistant_with_tool_calls()

    restored = turn._load_message(turn._dump_message(message))

    assert restored.replay is False
    assert all(part.cached_result is None for part in restored.tool_calls)
    assert restored.model_dump(mode="json") == message.model_dump(mode="json")


def test_plain_model_dump_drops_the_hidden_fields() -> None:
    cached = ai.tool_result_part("tc-1", tool_name="bash", result="file.txt")
    message = _assistant_with_tool_calls().model_copy(update={"replay": True})
    message.parts[1] = message.parts[1].model_copy(update={"cached_result": cached})

    data = message.model_dump(mode="json")
    assert "replay" not in data
    assert "cached_result" not in data["parts"][1]

    restored = messages_.Message.model_validate(data)
    assert restored.replay is False
    assert restored.tool_calls[0].cached_result is None
