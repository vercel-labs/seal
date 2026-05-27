from __future__ import annotations

from typing import Any, cast

from ai import events as ai_events
from ai import messages as ai_messages
from ai.agents.hooks import TOOL_APPROVAL_HOOK_TYPE
from ai.agents.ui.ai_sdk import UIMessage, to_ui_messages

from routers import chat as chat_router


def test_prepare_chat_request_accepts_ui_only_step_start_shape() -> None:
    stored_messages = [
        ai_messages.Message(
            id="user-1",
            role="user",
            parts=[ai_messages.TextPart(id="user-text-1", text="message 1")],
        ),
        ai_messages.Message(
            id="assistant-1",
            turn_id="turn-1",
            role="assistant",
            parts=[ai_messages.TextPart(id="assistant-text-1", text="reply 1")],
        ),
    ]
    request_messages = [
        UIMessage.model_validate(m.model_dump(mode="json", by_alias=True))
        for m in to_ui_messages(stored_messages)
    ]
    assistant_data = request_messages[1].model_dump(mode="json", by_alias=True)
    assistant_data["parts"] = [{"type": "step-start"}, *assistant_data["parts"]]
    request_messages[1] = UIMessage.model_validate(assistant_data)
    request_messages.append(
        UIMessage.model_validate(
            {
                "id": "user-2",
                "role": "user",
                "parts": [{"type": "text", "text": "message 2"}],
            }
        )
    )

    prepared = chat_router.prepare_chat_request(
        request_messages=request_messages,
        stored_messages=stored_messages,
    )

    assert prepared.has_work is True
    assert prepared.changed is True
    assert [(m.id, m.role) for m in prepared.messages] == [
        ("user-1", "user"),
        ("assistant-1", "assistant"),
        ("user-2", "user"),
    ]


def test_framework_renders_persisted_pending_approval() -> None:
    messages = [
        ai_messages.Message(
            id="assistant-1",
            turn_id="turn-1",
            role="assistant",
            parts=[
                ai_messages.TextPart(id="text-1", text="Need approval"),
                ai_messages.ToolCallPart(
                    id="call-part-1",
                    tool_call_id="call-1",
                    tool_name="bash",
                    tool_args="{}",
                ),
            ],
        ),
        ai_messages.Message(
            id="tool-1",
            turn_id="turn-1",
            role="tool",
            parts=[
                ai_messages.ToolResultPart(
                    id="result-1",
                    tool_call_id="call-1",
                    tool_name="bash",
                    result="Pending",
                    is_error=True,
                    is_hook_pending=True,
                )
            ],
        ),
        ai_messages.Message(
            id="internal-1",
            turn_id="turn-1",
            role="internal",
            parts=[
                ai_messages.HookPart(
                    id="hook-part-1",
                    hook_id="approve_call-1",
                    hook_type=TOOL_APPROVAL_HOOK_TYPE,
                    status="pending",
                    metadata={"tool": "bash"},
                )
            ],
        ),
    ]

    ui_messages = to_ui_messages(messages)
    tool_part = next(
        part.model_dump(mode="json", by_alias=True)
        for part in ui_messages[0].parts
        if getattr(part, "tool_call_id", None) == "call-1"
    )

    assert tool_part["state"] == "approval-requested"
    assert tool_part["approval"]["id"] == "approve_call-1"


def test_load_agent_event_round_trips_model_event() -> None:
    event = ai_events.TextDelta(
        block_id="text-1",
        chunk="hello",
        message=ai_messages.Message(
            id="assistant-1",
            role="assistant",
            parts=[ai_messages.TextPart(id="text-1", text="hello")],
        ),
    )

    loaded = chat_router._load_agent_event(event.model_dump(mode="json"))

    assert isinstance(loaded, ai_events.TextDelta)
    assert loaded.chunk == "hello"


def test_load_agent_event_round_trips_tool_result_event() -> None:
    message = ai_messages.Message(
        id="tool-1",
        role="tool",
        parts=[
            ai_messages.ToolResultPart(
                id="result-1",
                tool_call_id="call-1",
                tool_name="bash",
                result="ok",
                is_error=False,
            )
        ],
    )
    event = ai_events.ToolCallResult(message=message, results=message.tool_results)

    loaded = chat_router._load_agent_event(event.model_dump(mode="json"))

    assert isinstance(loaded, ai_events.ToolCallResult)
    assert loaded.results[0].result == "ok"


def test_load_agent_event_round_trips_hook_event() -> None:
    message = ai_messages.Message(
        id="internal-1",
        role="internal",
        parts=[
            ai_messages.HookPart(
                id="hook-part-1",
                hook_id="approve_call-1",
                hook_type=TOOL_APPROVAL_HOOK_TYPE,
                status="pending",
                metadata={"tool": "bash"},
            )
        ],
    )
    hook = cast(ai_messages.HookPart[Any], message.parts[0])
    event = ai_events.HookEvent(message=message, hook=hook)

    loaded = chat_router._load_agent_event(event.model_dump(mode="json"))

    assert isinstance(loaded, ai_events.HookEvent)
    assert loaded.hook.hook_id == "approve_call-1"
