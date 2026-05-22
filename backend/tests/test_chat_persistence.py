from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import ai
from ai import messages as ai_messages
from ai.agents.hooks import TOOL_APPROVAL_HOOK_TYPE
from ai.agents.ui.ai_sdk import UIMessage, to_ui_messages

import sessions as session_store


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def make_store(tmp_path: Path) -> session_store.JsonlSessionStore:
    return session_store.JsonlSessionStore(tmp_path / "sessions")


def test_jsonl_store_roundtrips_canonical_ai_messages(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    session_store.set_store_for_tests(store)
    try:
        messages = [
            ai_messages.Message(
                id="user-1",
                role="user",
                parts=[ai_messages.TextPart(id="part-1", text="hello")],
            ),
            ai_messages.Message(
                id="assistant-1",
                turn_id="turn-1",
                role="assistant",
                parts=[ai_messages.TextPart(id="part-2", text="hi")],
            ),
        ]

        run(session_store.create_session("session-1", "Title"))
        run(session_store.persist_ai_messages("session-1", messages))
        loaded = run(session_store.load_ai_messages("session-1"))

        assert [m.model_dump(mode="json") for m in loaded] == [
            m.model_dump(mode="json") for m in messages
        ]
        assert run(session_store.list_sessions())[0].title == "Title"
    finally:
        session_store.set_store_for_tests(None)


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

    prepared = session_store.prepare_chat_request(
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


def test_prepare_chat_request_accepts_approval_response() -> None:
    stored_messages = [
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
    request_data = [
        m.model_dump(mode="json", by_alias=True)
        for m in to_ui_messages(stored_messages)
    ]
    for part in request_data[0]["parts"]:
        if part.get("toolCallId") == "call-1":
            part["state"] = "approval-responded"
            part["approval"] = {"id": "approve_call-1", "approved": True}

    prepared = session_store.prepare_chat_request(
        request_messages=[UIMessage.model_validate(request_data[0])],
        stored_messages=stored_messages,
    )

    assert prepared.has_work is True
    assert prepared.changed is False
    assert prepared.messages is stored_messages
    assert [(a.hook_id, a.granted) for a in prepared.approvals] == [
        ("approve_call-1", True)
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


def test_branch_fork_label_and_compact_use_active_ai_messages(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    session_store.set_store_for_tests(store)
    try:
        original = [
            ai.user_message("one"),
            ai.assistant_message("two"),
        ]
        branch = [
            ai.user_message("one"),
            ai.assistant_message("alternate"),
        ]

        run(session_store.create_session("session-1"))
        run(session_store.persist_ai_messages("session-1", original))
        tree = run(session_store.get_tree("session-1"))
        assert tree is not None
        first_leaf = tree["leaf_id"]

        run(session_store.persist_ai_messages("session-1", branch))
        assert run(session_store.move_leaf("session-1", first_leaf)) is True
        assert [m.text for m in run(session_store.load_ai_messages("session-1"))] == [
            "one",
            "two",
        ]

        assert run(session_store.label_entry("session-1", first_leaf, "checkpoint"))
        labeled_tree = run(session_store.get_tree("session-1"))
        assert labeled_tree is not None
        labeled = next(e for e in labeled_tree["entries"] if e["id"] == first_leaf)
        assert labeled["label"] == "checkpoint"

        forked = run(session_store.fork_session("session-1", "session-2"))
        assert forked is not None
        assert [m.text for m in run(session_store.load_ai_messages("session-2"))] == [
            "one",
            "two",
        ]

        assert run(session_store.compact_session("session-2", "summary", keep_last=1))
        compacted = run(session_store.load_ai_messages("session-2"))
        assert compacted[0].text.startswith("Previous conversation summary")
        assert compacted[-1].text == "two"
    finally:
        session_store.set_store_for_tests(None)
