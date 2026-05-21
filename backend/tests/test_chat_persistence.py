from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any, cast

import ai
import pytest
from ai import events as ai_events
from ai import messages as ai_messages
from ai.agents.ui.ai_sdk import UIMessage, to_messages, to_ui_messages, ui_events

import db
from routers import chat, sessions

StoredRow = tuple[str, str, int, str, str | None, list[dict[str, Any]]]


def test_persist_run_messages_assigns_dense_sequence_and_preserves_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[StoredRow] = []

    async def fake_save_messages_batch(rows: list[StoredRow]) -> None:
        captured.extend(rows)

    monkeypatch.setattr(db, "save_messages_batch", fake_save_messages_batch)

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

    async def fake_save_messages_batch(rows: list[StoredRow]) -> None:
        captured.extend(rows)

    monkeypatch.setattr(db, "save_messages_batch", fake_save_messages_batch)

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

    async def fake_save_messages_batch(rows: list[StoredRow]) -> None:
        captured_batches.append(rows)

    monkeypatch.setattr(db, "save_messages_batch", fake_save_messages_batch)

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


def test_to_sse_with_roundtrip_metadata_injects_finish_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_to_stream(
        events: AsyncIterable[ai_events.AgentEvent],
    ) -> AsyncGenerator[ui_events.UIMessageStreamEvent]:
        _ = events
        yield ui_events.UIFinishEvent(finish_reason="stop")

    async def no_events() -> AsyncGenerator[ai_events.AgentEvent]:
        if False:
            yield cast(ai_events.AgentEvent, None)

    messages = [
        ai_messages.Message(
            id="assistant-1",
            turn_id="turn-1",
            role="assistant",
            parts=[ai_messages.TextPart(id="text-1", text="hi")],
        )
    ]

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in chat._to_sse_with_roundtrip_metadata(
                no_events(),
                lambda: messages,
            )
        ]

    monkeypatch.setattr(chat, "to_stream", fake_to_stream)

    chunks = asyncio.run(collect())

    assert chunks[0].startswith('data: {"type": "finish"')
    assert '"messageMetadata":' in chunks[0]
    assert '"sourceMessages":' in chunks[0]
    assert '"id": "assistant-1"' in chunks[0]
    assert chunks[1] == "data: [DONE]\n\n"
