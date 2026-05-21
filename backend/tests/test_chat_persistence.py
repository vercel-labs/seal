from __future__ import annotations

import asyncio
from typing import Any

import ai
import fastapi
import pytest
from ai import messages as ai_messages
from ai.agents.hooks import TOOL_APPROVAL_HOOK_TYPE
from ai.agents.ui.ai_sdk import UIMessage, to_messages, to_ui_messages

import db
from routers import chat, sessions

StoredRow = tuple[str, str, int, str, str | None, list[dict[str, Any]]]


def test_persist_run_messages_assigns_dense_sequence_and_preserves_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[StoredRow] = []

    async def fake_save_messages_snapshot(
        session_id: str,
        rows: list[StoredRow],
    ) -> None:
        assert session_id == "session-1"
        captured.extend(rows)

    monkeypatch.setattr(db, "save_messages_snapshot", fake_save_messages_snapshot)

    messages = [
        ai.system_message("system"),
        ai_messages.Message(
            id="user-1",
            role="user",
            parts=[ai_messages.TextPart(id="user-text-1", text="hello")],
        ),
        ai_messages.Message(
            id="assistant-1",
            turn_id="turn-1",
            role="assistant",
            parts=[ai_messages.TextPart(id="assistant-text-1", text="hi")],
        ),
    ]

    asyncio.run(chat._persist_run_messages("session-1", messages))

    assert [(row[0], row[2], row[3], row[4]) for row in captured] == [
        ("user-1", 0, "user", None),
        ("assistant-1", 1, "assistant", "turn-1"),
    ]
    assert captured[0][5][0]["id"] == "user-text-1"
    assert captured[1][5][0]["id"] == "assistant-text-1"


def test_persist_run_messages_preserves_multi_step_assistant_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[StoredRow] = []

    async def fake_save_messages_snapshot(
        session_id: str,
        rows: list[StoredRow],
    ) -> None:
        assert session_id == "session-1"
        captured.extend(rows)

    monkeypatch.setattr(db, "save_messages_snapshot", fake_save_messages_snapshot)

    messages = [
        ai_messages.Message(
            id="assistant-live-id-1",
            turn_id="turn-1",
            role="assistant",
            parts=[
                ai_messages.TextPart(id="text-part-1", text="I'll use a tool."),
                ai_messages.ToolCallPart(
                    id="call-part-1",
                    tool_call_id="call-1",
                    tool_name="bash",
                    tool_args='{"command":"echo hi"}',
                ),
            ],
        ),
        ai_messages.Message(
            id="tool-live-id",
            turn_id="turn-1",
            role="tool",
            parts=[
                ai_messages.ToolResultPart(
                    id="result-part-1",
                    tool_call_id="call-1",
                    tool_name="bash",
                    result={"value": "hi\n"},
                )
            ],
        ),
        ai_messages.Message(
            id="assistant-live-id-2",
            turn_id="turn-1",
            role="assistant",
            parts=[
                ai_messages.TextPart(
                    id="text-part-2",
                    text="The tool returned hi.",
                )
            ],
        ),
    ]

    asyncio.run(chat._persist_run_messages("session-1", messages))

    assert [row[0] for row in captured] == [
        "assistant-live-id-1",
        "tool-live-id",
        "assistant-live-id-2",
    ]
    assert [row[4] for row in captured] == ["turn-1", "turn-1", "turn-1"]
    assert captured[0][5][0]["id"] == "text-part-1"
    assert captured[0][5][1]["id"] == "call-part-1"
    assert captured[1][5][0]["id"] == "result-part-1"
    assert captured[2][5][0]["id"] == "text-part-2"


def test_persist_run_messages_is_stable_after_metadata_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_batches: list[list[StoredRow]] = []

    async def fake_save_messages_snapshot(
        session_id: str,
        rows: list[StoredRow],
    ) -> None:
        assert session_id == "session-1"
        captured_batches.append(rows)

    monkeypatch.setattr(db, "save_messages_snapshot", fake_save_messages_snapshot)

    first_run = [
        ai_messages.Message(
            id="user-1",
            role="user",
            parts=[ai_messages.TextPart(id="user-text-1", text="message 1")],
        ),
        ai_messages.Message(
            id="assistant-1",
            turn_id="turn-1",
            role="assistant",
            parts=[
                ai_messages.TextPart(
                    id="assistant-text-1",
                    text="reply 1",
                )
            ],
        ),
    ]

    asyncio.run(chat._persist_run_messages("session-1", first_run))
    first_batch = captured_batches[-1]
    assert [row[0] for row in first_batch] == [
        "user-1",
        "assistant-1",
    ]

    stored = [
        ai_messages.Message.model_validate(
            {
                "id": row[0],
                "role": row[3],
                "turn_id": row[4],
                "parts": row[5],
            }
        )
        for row in first_batch
    ]
    ui_messages = [
        UIMessage.model_validate(m.model_dump(mode="json", by_alias=True))
        for m in to_ui_messages(stored)
    ]
    roundtripped, _ = to_messages(
        [
            *ui_messages,
            UIMessage.model_validate(
                {
                    "id": "user-2",
                    "role": "user",
                    "parts": [{"type": "text", "text": "message 2"}],
                }
            ),
        ]
    )

    asyncio.run(chat._persist_run_messages("session-1", roundtripped))
    second_batch = captured_batches[-1]

    assert [row[0] for row in second_batch] == [
        "user-1",
        "assistant-1",
        "user-2",
    ]
    assert second_batch[1][4] == "turn-1"
    assert second_batch[1][5][0]["id"] == "assistant-text-1"


def test_stored_to_ai_messages_preserves_canonical_turn_id() -> None:
    stored = [
        db.StoredMessage(
            id="user-1",
            seq=0,
            role="user",
            parts=[
                ai_messages.TextPart(
                    id="user-text-1",
                    text="message 1",
                ).model_dump(mode="json")
            ],
            created_at="2026-05-20T00:00:00+00:00",
        ),
        db.StoredMessage(
            id="assistant-1",
            seq=1,
            turn_id="turn-1",
            role="assistant",
            parts=[
                ai_messages.TextPart(
                    id="assistant-text-1",
                    text="reply 1",
                ).model_dump(mode="json")
            ],
            created_at="2026-05-20T00:00:01+00:00",
        ),
    ]

    messages = sessions._stored_to_ai_messages(stored)

    assert [(m.id, m.turn_id, m.role) for m in messages] == [
        ("user-1", None, "user"),
        ("assistant-1", "turn-1", "assistant"),
    ]
    assert sessions._extract_first_user_text(messages) == "message 1"


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


def test_history_assertion_uses_framework_normalization() -> None:
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
            parts=[
                ai_messages.TextPart(
                    id="assistant-text-1",
                    text="reply 1",
                )
            ],
        ),
    ]
    request_messages = [
        UIMessage.model_validate(m.model_dump(mode="json", by_alias=True))
        for m in to_ui_messages(stored_messages)
    ]
    assistant_data = request_messages[1].model_dump(mode="json", by_alias=True)
    assistant_data["parts"] = [
        {"type": "step-start"},
        *assistant_data["parts"],
    ]
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

    latest_user = chat._assert_matching_history(
        request_messages=request_messages,
        stored_messages=stored_messages,
    )

    assert latest_user is request_messages[-1]


def test_history_assertion_accepts_approval_response() -> None:
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
    request_messages = [UIMessage.model_validate(request_data[0])]

    latest_user = chat._assert_matching_history(
        request_messages=request_messages,
        stored_messages=stored_messages,
    )

    assert latest_user is None


def test_history_assertion_rejects_changed_stored_history() -> None:
    stored_messages = [
        ai_messages.Message(
            id="user-1",
            role="user",
            parts=[ai_messages.TextPart(id="user-text-1", text="message 1")],
        ),
    ]
    request_messages = [
        UIMessage.model_validate(
            {
                "id": "user-1",
                "role": "user",
                "parts": [{"type": "text", "text": "changed"}],
            }
        )
    ]

    with pytest.raises(fastapi.HTTPException) as exc_info:
        chat._assert_matching_history(
            request_messages=request_messages,
            stored_messages=stored_messages,
        )
    assert exc_info.value.status_code == 409


def test_save_messages_batch_rejects_duplicate_message_ids() -> None:
    rows: list[StoredRow] = [
        ("message-1", "session-1", 0, "user", None, []),
        ("message-1", "session-1", 1, "assistant", "turn-1", []),
    ]

    with pytest.raises(AssertionError, match="duplicate message IDs"):
        asyncio.run(db.save_messages_batch(rows))


def test_save_messages_snapshot_rejects_duplicate_message_ids() -> None:
    rows: list[StoredRow] = [
        ("message-1", "session-1", 0, "user", None, []),
        ("message-1", "session-1", 1, "assistant", "turn-1", []),
    ]

    with pytest.raises(AssertionError, match="duplicate message IDs"):
        asyncio.run(db.save_messages_snapshot("session-1", rows))


def test_save_messages_snapshot_deletes_stale_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[Any, ...]]] = []

    class FakeTransaction:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *args: object) -> None:
            return None

    class FakeConnection:
        def transaction(self) -> FakeTransaction:
            return FakeTransaction()

        async def execute(self, sql: str, *args: Any) -> None:
            calls.append((sql, args))

    class FakeAcquire:
        def __init__(self, conn: FakeConnection) -> None:
            self.conn = conn

        async def __aenter__(self) -> FakeConnection:
            return self.conn

        async def __aexit__(self, *args: object) -> None:
            return None

    class FakePool:
        def __init__(self) -> None:
            self.conn = FakeConnection()

        def acquire(self) -> FakeAcquire:
            return FakeAcquire(self.conn)

    async def fake_get_pool() -> FakePool:
        return FakePool()

    monkeypatch.setattr(db, "get_pool", fake_get_pool)

    rows: list[StoredRow] = [
        ("message-1", "session-1", 0, "user", None, []),
        ("message-2", "session-1", 1, "assistant", "turn-1", []),
    ]
    asyncio.run(db.save_messages_snapshot("session-1", rows))

    delete_call = calls[-1]
    assert delete_call[0].startswith("DELETE FROM messages WHERE session_id = $1")
    assert delete_call[1] == ("session-1", ["message-1", "message-2"])
