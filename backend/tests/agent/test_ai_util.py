import ai

from agent import ai_util


def test_recover_partial_messages_closes_pending_tool_calls() -> None:
    user = ai.user_message("run it")
    partial = ai.messages.Message(
        role="assistant",
        parts=[
            ai.messages.ToolCallPart(
                tool_call_id="tc-1",
                tool_name="bash",
                tool_args='{"command":',
            )
        ],
    )

    messages = ai_util.recover_partial_messages(
        [user],
        [
            ai.events.StreamStart(message=partial),
            ai.events.ToolStart(message=partial, tool_call_id="tc-1", tool_name="bash"),
            ai.events.ToolDelta(
                message=partial, tool_call_id="tc-1", chunk='{"command":'
            ),
        ],
    )

    assert [message.role for message in messages] == ["user", "assistant", "tool"]
    assert messages[1].tool_calls[0].tool_args == "{}"
    [result] = messages[2].tool_results
    assert result.tool_call_id == "tc-1"
    assert result.result == "Interrupted by user"
    assert result.result_kind == "error"


def test_recover_partial_messages_without_events_returns_input() -> None:
    user = ai.user_message("run it")

    assert ai_util.recover_partial_messages([user], []) == [user]
