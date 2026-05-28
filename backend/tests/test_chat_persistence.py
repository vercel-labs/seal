from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

import fastapi
import pytest
from ai import events as ai_events
from ai import messages as ai_messages
from ai.agents.hooks import TOOL_APPROVAL_HOOK_TYPE
from ai.agents.ui.ai_sdk import UIMessage, to_ui_messages

import sessions
import stream_store
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


def _sse_event_types(chunks: list[str]) -> list[str]:
    types: list[str] = []
    for chunk in chunks:
        payload = chunk.removeprefix("data: ").strip()
        if payload == "[DONE]":
            continue
        types.append(str(json.loads(payload)["type"]))
    return types


def test_supported_sse_filters_approval_response_chunk() -> None:
    hook_message = ai_messages.Message(
        id="internal-1",
        turn_id="assistant-1",
        role="internal",
        parts=[
            ai_messages.HookPart(
                id="hook-part-1",
                hook_id="approve_call-1",
                hook_type=TOOL_APPROVAL_HOOK_TYPE,
                status="resolved",
                metadata={"tool": "bash"},
                resolution={"granted": True, "reason": None},
            )
        ],
    )
    result_message = ai_messages.Message(
        id="tool-1",
        turn_id="assistant-1",
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

    async def run() -> list[str]:
        async def events() -> AsyncGenerator[ai_events.AgentEvent]:
            hook = cast(ai_messages.HookPart[Any], hook_message.parts[0])
            yield ai_events.HookEvent(message=hook_message, hook=hook)
            yield ai_events.ToolCallResult(
                message=result_message,
                results=result_message.tool_results,
            )

        return [chunk async for chunk in chat_router._to_supported_sse(events())]

    types = _sse_event_types(asyncio.run(run()))

    assert "tool-approval-response" not in types
    assert "tool-output-available" in types


def test_supported_sse_keeps_denied_state_for_rejected_approval() -> None:
    hook_message = ai_messages.Message(
        id="internal-1",
        turn_id="assistant-1",
        role="internal",
        parts=[
            ai_messages.HookPart(
                id="hook-part-1",
                hook_id="approve_call-1",
                hook_type=TOOL_APPROVAL_HOOK_TYPE,
                status="resolved",
                metadata={"tool": "bash"},
                resolution={"granted": False, "reason": "no"},
            )
        ],
    )
    result_message = ai_messages.Message(
        id="tool-1",
        turn_id="assistant-1",
        role="tool",
        parts=[
            ai_messages.ToolResultPart(
                id="result-1",
                tool_call_id="call-1",
                tool_name="bash",
                result="Rejected: no",
                is_error=True,
            )
        ],
    )

    async def run() -> list[str]:
        async def events() -> AsyncGenerator[ai_events.AgentEvent]:
            hook = cast(ai_messages.HookPart[Any], hook_message.parts[0])
            yield ai_events.HookEvent(message=hook_message, hook=hook)
            yield ai_events.ToolCallResult(
                message=result_message,
                results=result_message.tool_results,
            )

        return [chunk async for chunk in chat_router._to_supported_sse(events())]

    types = _sse_event_types(asyncio.run(run()))

    assert "tool-approval-response" not in types
    assert "tool-output-denied" in types
    assert "tool-output-error" not in types


def test_resume_chat_stream_accepts_waiting_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SEAL_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("SEAL_STREAMS_DIR", str(tmp_path / "streams"))

    async def run() -> fastapi.responses.Response:
        await sessions.create_session("s1")
        await sessions.set_active_run("s1", "wrun_1")
        await stream_store.set_status("s1", "waiting")
        request = fastapi.Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/chat/s1/stream",
                "headers": [],
                "query_string": b"",
            }
        )
        return await chat_router.resume_chat_stream("s1", request)

    response = asyncio.run(run())

    assert response.status_code == 200
