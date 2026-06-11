"""The replay-smuggling boundary (``turn._dump_message`` / ``_load_message``)
and the direct ``run_turn`` seam.

``Message.replay`` and ``ToolCallPart.cached_result`` are ``exclude=True``
pydantic fields in the ai SDK, so they vanish through a plain
``model_dump``/``model_validate`` round trip. The turn module smuggles them
through the llm_step JSON boundary by hand; replay-after-approval depends on
them surviving.
"""

from __future__ import annotations

import ai
import ai.types.messages as messages_
import turn_capture
import vercel.workflow
from conftest import MockProvider, text_msg
from harness import InProcessWorld, wait_run

from agent import proto, turn


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
    # the visible content is untouched
    assert restored.model_dump(mode="json") == message.model_dump(mode="json")


def test_round_trip_of_plain_message_stays_plain() -> None:
    message = _assistant_with_tool_calls()

    restored = turn._load_message(turn._dump_message(message))

    assert restored.replay is False
    assert all(part.cached_result is None for part in restored.tool_calls)
    assert restored.model_dump(mode="json") == message.model_dump(mode="json")


def test_plain_model_dump_drops_the_hidden_fields() -> None:
    """Documents why the hack exists.

    If this fails, the SDK started serializing ``replay``/``cached_result``
    and ``_dump_message``/``_load_message`` can be deleted.
    """
    cached = ai.tool_result_part("tc-1", tool_name="bash", result="file.txt")
    message = _assistant_with_tool_calls().model_copy(update={"replay": True})
    message.parts[1] = message.parts[1].model_copy(update={"cached_result": cached})

    data = message.model_dump(mode="json")
    assert "replay" not in data
    assert "cached_result" not in data["parts"][1]

    restored = messages_.Message.model_validate(data)
    assert restored.replay is False
    assert restored.tool_calls[0].cached_result is None


async def test_capture_turn_runs_a_single_turn_directly(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    """The turn-level seam: run one ``run_turn`` without a session around it.

    This is the fast iteration loop for changes to ``DurableAgent.loop`` —
    a failure here is a turn bug, not a driver bug.
    """
    scripted_model.responses = [[text_msg("solo answer")]]
    turn_input = proto.TurnInput(
        session_id="s1",
        messages=[
            ai.system_message(turn.SYSTEM_PROMPT),
            ai.user_message("hi"),
        ],
        turn_hook_token="seal-turn:s1:0",
    )

    run = await vercel.workflow.start(
        turn_capture.capture_turn, turn_input.model_dump(mode="json")
    )
    output = proto.TurnOutput.model_validate(await wait_run(run))

    assert output.kind == "suspend"
    assert output.messages[-1].text == "solo answer"
    assert output.pending_requests == []
