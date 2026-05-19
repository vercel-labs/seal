from __future__ import annotations

import asyncio
from typing import Any

import ai
import pytest
from ai import messages as ai_messages
from ai.agents.ui.ai_sdk import UIMessage, to_messages, to_ui_messages

import db
from routers import chat

StoredRow = tuple[str, str, int, str, list[dict[str, Any]]]


def test_persist_run_messages_assigns_dense_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[StoredRow] = []

    async def fake_save_messages_batch(rows: list[StoredRow]) -> None:
        captured.extend(rows)

    monkeypatch.setattr(db, "save_messages_batch", fake_save_messages_batch)

    messages = [
        ai.system_message("system"),
        ai.user_message("hello"),
        ai.assistant_message("hi"),
    ]
    user_id = messages[1].id
    assistant_turn_id = messages[2].id

    asyncio.run(chat._persist_run_messages("session-1", messages))

    assert [(row[2], row[3]) for row in captured] == [
        (0, "user"),
        (1, "assistant"),
    ]
    assert [row[0] for row in captured] == [
        user_id,
        f"{assistant_turn_id}:assistant:0",
    ]
    assert captured[0][4][0]["id"] == f"{user_id}:text:0"
    assert captured[1][4][0]["id"] == f"{assistant_turn_id}:assistant:0:text:0"


def test_persist_run_messages_canonicalizes_multi_step_assistant_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[StoredRow] = []

    async def fake_save_messages_batch(rows: list[StoredRow]) -> None:
        captured.extend(rows)

    monkeypatch.setattr(db, "save_messages_batch", fake_save_messages_batch)

    messages = [
        ai_messages.Message(
            id="turn-1",
            role="assistant",
            parts=[
                ai_messages.TextPart(text="I'll use a tool."),
                ai_messages.ToolCallPart(
                    tool_call_id="call-1",
                    tool_name="bash",
                    tool_args='{"command":"echo hi"}',
                ),
            ],
        ),
        ai_messages.Message(
            id="tool-live-id",
            role="tool",
            parts=[
                ai_messages.ToolResultPart(
                    tool_call_id="call-1",
                    tool_name="bash",
                    result={"value": "hi\n"},
                )
            ],
        ),
        ai_messages.Message(
            id="assistant-live-id-2",
            role="assistant",
            parts=[ai_messages.TextPart(text="The tool returned hi.")],
        ),
    ]

    asyncio.run(chat._persist_run_messages("session-1", messages))

    assert [row[0] for row in captured] == [
        "turn-1:assistant:0",
        "turn-1:tool:0",
        "turn-1:assistant:1",
    ]
    assert captured[0][4][0]["id"] == "turn-1:assistant:0:text:0"
    assert captured[0][4][1]["id"] == "turn-1:assistant:0:call:call-1"
    assert captured[1][4][0]["id"] == "turn-1:tool:0:result:call-1"
    assert captured[2][4][0]["id"] == "turn-1:assistant:1:text:0"


def test_persist_run_messages_is_stable_after_ui_roundtrip(
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
            parts=[ai_messages.TextPart(text="message 1")],
        ),
        ai_messages.Message(
            id="assistant-1",
            role="assistant",
            parts=[ai_messages.TextPart(text="reply 1")],
        ),
    ]

    asyncio.run(chat._persist_run_messages("session-1", first_run))
    first_batch = captured_batches[-1]
    assert [row[0] for row in first_batch] == [
        "user-1",
        "assistant-1:assistant:0",
    ]

    stored = [
        ai_messages.Message.model_validate(
            {"id": row[0], "role": row[3], "parts": row[4]}
        )
        for row in first_batch
    ]
    ui_messages = to_ui_messages(stored)
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
        "assistant-1:assistant:0",
        "user-2",
    ]
    assert "assistant-1" not in {row[0] for row in second_batch}


def test_no_op_sse_returns_finish_chunk() -> None:
    async def collect() -> list[str]:
        return [chunk async for chunk in chat._no_op_sse()]

    assert asyncio.run(collect()) == [
        'data: {"type":"finish","finishReason":"stop"}\n\n'
    ]
