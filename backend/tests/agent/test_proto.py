"""Round-trips of every shape the app persists or sends across a hook."""

from __future__ import annotations

from typing import Any

import ai
import ai.types.events as events_
import ai.types.messages as messages_

from agent import proto


def _hook_event() -> events_.HookEvent:
    hook: messages_.HookPart[Any] = messages_.HookPart(
        hook_id="approve_tc-1",
        hook_type="ToolApproval",
        status="pending",
        metadata={"tool": "bash", "kwargs": {"command": "ls"}},
    )
    return events_.HookEvent(
        message=ai.messages.Message(role="internal", parts=[hook]), hook=hook
    )


def test_resume_payloads_round_trip() -> None:
    bundle = ai.agents.MessageBundle(
        messages=(
            ai.user_message("delegated task"),
            ai.messages.Message(
                role="assistant", parts=[messages_.TextPart(text="child answer")]
            ),
        )
    )
    payloads: list[proto.ResumePayload] = [
        proto.SubagentResult(
            output=proto.SessionOutput(
                tool_call_id="tc-1", session_id="s:child:tc-1", output=bundle
            )
        ),
        proto.ToolApprovals(
            tool_approvals=[
                proto.ToolApprovalResponse(
                    tool_call_id="tc-1", granted=False, reason="nope"
                )
            ]
        ),
        proto.NewUserMessage(prompt="hi", close=False),
    ]
    for payload in payloads:
        restored = proto.RESUME_PAYLOAD_ADAPTER.validate_python(
            payload.model_dump(mode="json")
        )
        assert type(restored) is type(payload)
        assert restored.model_dump(mode="json") == payload.model_dump(mode="json")


def test_subagent_result_restores_message_bundle() -> None:
    bundle = ai.agents.MessageBundle(messages=(ai.user_message("task"),))
    payload = proto.SubagentResult(
        output=proto.SessionOutput(
            tool_call_id="tc-1", session_id="child", output=bundle
        )
    )
    restored = proto.RESUME_PAYLOAD_ADAPTER.validate_python(
        payload.model_dump(mode="json")
    )
    assert isinstance(restored, proto.SubagentResult)
    assert isinstance(restored.output.output, ai.agents.MessageBundle)
    assert restored.output.output.messages[0].text == "task"


def test_stream_events_round_trip() -> None:
    tool_message = ai.tool_message(tool_call_id="tc-1", tool_name="bash", result="ok")
    stream_events: list[proto.StreamEvent] = [
        proto.LifecycleEvent(type=proto.TURN_STARTED, data={"turn_index": 0}),
        events_.TextDelta(block_id="text-0", chunk="hello"),
        events_.StreamEnd(
            message=ai.messages.Message(
                role="assistant",
                parts=[
                    messages_.TextPart(text="hi"),
                    messages_.ToolCallPart(
                        tool_call_id="tc-1", tool_name="bash", tool_args="{}"
                    ),
                ],
            )
        ),
        events_.ToolCallResult(message=tool_message, results=tool_message.tool_results),
        _hook_event(),
    ]
    for event in stream_events:
        restored = proto.STREAM_EVENT_ADAPTER.validate_python(
            event.model_dump(mode="json")
        )
        assert type(restored) is type(event), f"{type(event).__name__} changed type"
        assert restored.model_dump(mode="json") == event.model_dump(mode="json")


def test_hook_event_round_trip_keeps_approval_fields() -> None:
    restored = proto.STREAM_EVENT_ADAPTER.validate_python(
        _hook_event().model_dump(mode="json")
    )
    assert isinstance(restored, events_.HookEvent)
    assert restored.hook.hook_id == "approve_tc-1"
    assert restored.hook.status == "pending"
    assert restored.hook.metadata["tool"] == "bash"
    assert restored.hook.metadata["kwargs"] == {"command": "ls"}


def test_session_state_round_trip_is_dump_stable() -> None:
    bundle = ai.agents.MessageBundle(
        messages=(
            ai.user_message("task"),
            ai.messages.Message(
                role="assistant", parts=[messages_.TextPart(text="child answer")]
            ),
        )
    )
    messages = [
        ai.system_message("you are seal"),
        ai.user_message("do the thing"),
        ai.messages.Message(
            role="assistant",
            parts=[
                messages_.TextPart(text="delegating"),
                messages_.ToolCallPart(
                    tool_call_id="tc-1",
                    tool_name="subagent",
                    tool_args='{"prompt":"x"}',
                ),
            ],
        ),
        ai.messages.Message(
            role="tool",
            parts=[ai.tool_result_part("tc-1", tool_name="subagent", result=bundle)],
        ),
    ]
    state = proto.SessionState(
        session_id="s1",
        mode="infinite",
        messages=messages,
        tool_approvals=[proto.ToolApprovalResponse(tool_call_id="tc-2", granted=True)],
        pending=proto.PendingState(
            turn_index=3,
            subagents=[proto.SubagentRequest(tool_call_id="tc-9", prompt="go")],
            dispatched=True,
            subagent_outputs={
                "tc-9": proto.SessionOutput(
                    tool_call_id="tc-9", session_id="s1:child:tc-9", output=bundle
                )
            },
        ),
    )

    once = state.model_dump(mode="json")
    restored = proto.SessionState.model_validate(once)
    assert restored.model_dump(mode="json") == once

    roles = [message.role for message in restored.messages]
    assert roles == ["system", "user", "assistant", "tool"]
    assert restored.messages[2].tool_calls[0].tool_call_id == "tc-1"
    assert restored.messages[3].tool_results[0].tool_call_id == "tc-1"
