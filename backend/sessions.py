"""Session storage and AI message helpers."""

from __future__ import annotations

import json
from typing import Any

import asyncpg  # type: ignore[import-untyped]
import pydantic
from ai import messages as ai_messages
from ai.agents.hooks import TOOL_APPROVAL_HOOK_TYPE
from ai.agents.ui.ai_sdk import UIMessage, to_messages, to_ui_messages

import db

StoredRow = tuple[str, str, int, str, str | None, list[dict[str, Any]]]

_SESSION_COLS = "id, title, created_at, updated_at"
_HISTORY_MISMATCH_DETAIL = "Frontend message history does not match stored history"


class HistoryMismatchError(Exception):
    """Raised when client history does not match stored history."""


class Session(pydantic.BaseModel):
    """Serialisable session record."""

    id: str
    title: str | None = None
    created_at: str
    updated_at: str


class StoredMessage(pydantic.BaseModel):
    """A message as stored in the DB (parts already parsed)."""

    id: str
    seq: int
    turn_id: str | None = None
    role: str
    parts: list[dict[str, Any]]
    created_at: str


def _row_to_session(row: asyncpg.Record) -> Session:
    return Session(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
    )


def _parse_jsonb(val: Any) -> Any:
    if isinstance(val, str):
        return json.loads(val)
    return val


def _row_to_message(row: asyncpg.Record) -> StoredMessage:
    return StoredMessage(
        id=row["id"],
        seq=row["seq"],
        turn_id=row["turn_id"],
        role=row["role"],
        parts=_parse_jsonb(row["parts"]),
        created_at=row["created_at"].isoformat(),
    )


async def create_session(session_id: str, title: str | None = None) -> Session:
    """Insert a new session and return it. No-op if the ID exists."""
    pool = await db.get_pool()
    row = await pool.fetchrow(
        f"INSERT INTO sessions (id, title) VALUES ($1, $2) "
        f"ON CONFLICT (id) DO UPDATE SET id = EXCLUDED.id "
        f"RETURNING {_SESSION_COLS}",
        session_id,
        title,
    )
    return _row_to_session(row)


async def get_or_create_session(session_id: str) -> Session:
    """Return a session, creating it if needed."""
    return await create_session(session_id)


async def list_sessions() -> list[Session]:
    """Return all sessions ordered by most-recently-updated first."""
    pool = await db.get_pool()
    rows = await pool.fetch(
        f"SELECT {_SESSION_COLS} FROM sessions ORDER BY updated_at DESC",
    )
    return [_row_to_session(r) for r in rows]


async def get_session(session_id: str) -> Session | None:
    """Return a single session or None."""
    pool = await db.get_pool()
    row = await pool.fetchrow(
        f"SELECT {_SESSION_COLS} FROM sessions WHERE id = $1",
        session_id,
    )
    return _row_to_session(row) if row else None


async def update_session_title(session_id: str, title: str) -> Session | None:
    """Set the title and bump updated_at."""
    pool = await db.get_pool()
    row = await pool.fetchrow(
        f"UPDATE sessions SET title = $2, updated_at = now() WHERE id = $1 "
        f"RETURNING {_SESSION_COLS}",
        session_id,
        title,
    )
    return _row_to_session(row) if row else None


async def delete_session(session_id: str) -> bool:
    """Delete a session and its messages. Return True if found."""
    pool = await db.get_pool()
    result = await pool.execute("DELETE FROM sessions WHERE id = $1", session_id)
    return bool(result == "DELETE 1")


async def touch_session(session_id: str) -> None:
    """Bump updated_at without changing other fields."""
    pool = await db.get_pool()
    await pool.execute(
        "UPDATE sessions SET updated_at = now() WHERE id = $1", session_id
    )


async def get_messages(session_id: str) -> list[StoredMessage]:
    """Return all messages for a session in chronological order."""
    pool = await db.get_pool()
    rows = await pool.fetch(
        "SELECT id, seq, turn_id, role, parts, created_at "
        "FROM messages WHERE session_id = $1 ORDER BY seq, created_at, id",
        session_id,
    )
    return [_row_to_message(r) for r in rows]


async def save_message(
    message_id: str,
    session_id: str,
    role: str,
    parts: list[dict[str, Any]],
    seq: int | None = None,
    turn_id: str | None = None,
) -> None:
    """Insert or update a single message."""
    pool = await db.get_pool()
    await pool.execute(
        "WITH next_seq AS ("
        "  SELECT COALESCE($5::int, COALESCE(MAX(seq) + 1, 0)) AS seq "
        "  FROM messages WHERE session_id = $2"
        ") "
        "INSERT INTO messages (id, session_id, role, parts, seq, turn_id) "
        "SELECT $1, $2, $3, $4::jsonb, next_seq.seq, $6 FROM next_seq "
        "ON CONFLICT (id) DO UPDATE SET "
        "session_id = EXCLUDED.session_id, "
        "role = EXCLUDED.role, "
        "parts = EXCLUDED.parts, "
        "seq = EXCLUDED.seq, "
        "turn_id = EXCLUDED.turn_id",
        message_id,
        session_id,
        role,
        json.dumps(parts),
        seq,
        turn_id,
    )


async def save_messages_batch(messages: list[StoredRow]) -> None:
    """Batch-upsert messages."""
    if not messages:
        return

    message_ids = [row[0] for row in messages]
    assert len(message_ids) == len(set(message_ids)), (
        "save_messages_batch received duplicate message IDs"
    )

    pool = await db.get_pool()
    sql, args = _message_upsert_sql(messages)
    await pool.execute(sql, *args)


def _assert_valid_snapshot(session_id: str, messages: list[StoredRow]) -> None:
    message_ids = [row[0] for row in messages]
    assert len(message_ids) == len(set(message_ids)), (
        "save_messages_snapshot received duplicate message IDs"
    )

    session_ids = {row[1] for row in messages}
    assert session_ids <= {session_id}, (
        "save_messages_snapshot received rows for a different session"
    )

    seqs = [row[2] for row in messages]
    assert seqs == list(range(len(messages))), (
        "save_messages_snapshot requires dense sequence numbers"
    )


async def save_messages_snapshot(session_id: str, messages: list[StoredRow]) -> None:
    """Replace a session's persisted messages with the supplied snapshot."""
    _assert_valid_snapshot(session_id, messages)
    message_ids = [row[0] for row in messages]

    pool = await db.get_pool()
    async with pool.acquire() as conn, conn.transaction():
        if not messages:
            await conn.execute("DELETE FROM messages WHERE session_id = $1", session_id)
            return

        sql, args = _message_upsert_sql(messages)
        await conn.execute(sql, *args)
        await conn.execute(
            "DELETE FROM messages WHERE session_id = $1 AND NOT (id = ANY($2::text[]))",
            session_id,
            message_ids,
        )


def _message_upsert_sql(messages: list[StoredRow]) -> tuple[str, list[Any]]:
    args: list[Any] = []
    placeholders: list[str] = []
    for i, (mid, sid, seq, role, turn_id, parts) in enumerate(messages):
        base = i * 6
        placeholders.append(
            f"(${base + 1}, ${base + 2}, ${base + 3}, ${base + 4}, "
            f"${base + 5}, ${base + 6}::jsonb)"
        )
        args.extend([mid, sid, seq, role, turn_id, json.dumps(parts)])

    sql = (
        "INSERT INTO messages (id, session_id, seq, role, turn_id, parts) VALUES "
        + ", ".join(placeholders)
        + " ON CONFLICT (id) DO UPDATE SET "
        + "session_id = EXCLUDED.session_id, "
        + "seq = EXCLUDED.seq, "
        + "role = EXCLUDED.role, "
        + "turn_id = EXCLUDED.turn_id, "
        + "parts = EXCLUDED.parts"
    )
    return sql, args


def stored_to_ai_messages(rows: list[StoredMessage]) -> list[ai_messages.Message]:
    """Decode stored rows into framework messages."""
    return [
        ai_messages.Message.model_validate(
            {
                "id": row.id,
                "turn_id": row.turn_id,
                "role": row.role,
                "parts": row.parts,
            }
        )
        for row in rows
    ]


async def load_ai_messages(session_id: str) -> list[ai_messages.Message]:
    """Load a session's messages in framework shape."""
    return stored_to_ai_messages(await get_messages(session_id))


def ai_messages_to_rows(
    session_id: str,
    messages: list[ai_messages.Message],
) -> list[StoredRow]:
    """Encode framework messages for storage."""
    stored_messages = [m for m in messages if m.role != "system"]
    return [
        (
            m.id,
            session_id,
            seq,
            m.role,
            m.turn_id,
            [p.model_dump(mode="json") for p in m.parts],
        )
        for seq, m in enumerate(stored_messages)
    ]


async def persist_ai_messages(
    session_id: str,
    messages: list[ai_messages.Message],
) -> None:
    """Persist the latest canonical framework snapshot for a session."""
    await save_messages_snapshot(session_id, ai_messages_to_rows(session_id, messages))


def history_signature(messages: list[ai_messages.Message]) -> list[dict[str, Any]]:
    """Stable comparison shape for framework-normalized history."""
    result: list[dict[str, Any]] = []
    for message in messages:
        if message.role in ("system", "internal"):
            continue

        parts: list[dict[str, Any]] = []
        for part in message.parts:
            if isinstance(part, ai_messages.ToolResultPart) and part.is_hook_pending:
                continue
            parts.append(part.model_dump(mode="json"))

        if not parts and message.role == "tool":
            continue

        result.append(
            {
                "id": message.id,
                "turn_id": message.turn_id,
                "role": message.role,
                "parts": parts,
            }
        )
    return result


def validate_chat_history(
    *,
    request_messages: list[UIMessage],
    stored_messages: list[ai_messages.Message],
) -> UIMessage | None:
    """Validate frontend history and return the latest new user message."""
    stored_ui_messages = [
        UIMessage.model_validate(m.model_dump(mode="json", by_alias=True))
        for m in to_ui_messages(stored_messages)
    ]

    latest_user: UIMessage | None = None
    request_history = request_messages
    if (
        len(request_messages) == len(stored_ui_messages) + 1
        and request_messages[-1].role == "user"
    ):
        latest_user = request_messages[-1]
        request_history = request_messages[:-1]

    if len(request_history) != len(stored_ui_messages):
        raise HistoryMismatchError(_HISTORY_MISMATCH_DETAIL)

    request_ai_messages, _ = to_messages(request_history)
    stored_ai_messages, _ = to_messages(stored_ui_messages)
    if history_signature(request_ai_messages) != history_signature(stored_ai_messages):
        raise HistoryMismatchError(_HISTORY_MISMATCH_DETAIL)

    return latest_user


def first_user_text(messages: list[ai_messages.Message]) -> str | None:
    """Return the text of the first user message, or None."""
    for msg in messages:
        if msg.role != "user":
            continue
        if msg.text:
            return msg.text
    return None


def _tool_call_id_for_hook(hook: ai_messages.HookPart[Any]) -> str | None:
    if hook.hook_type != TOOL_APPROVAL_HOOK_TYPE:
        return None
    prefix = "approve_"
    if not hook.hook_id.startswith(prefix):
        return None
    return hook.hook_id[len(prefix) :]


def _turn_id_for_tool_call(
    messages: list[ai_messages.Message],
    tool_call_id: str,
) -> str | None:
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        if any(part.tool_call_id == tool_call_id for part in message.tool_calls):
            return message.turn_id or message.id
    return None


def with_pending_hook_messages(
    messages: list[ai_messages.Message],
    pending_hook_messages: list[ai_messages.Message],
) -> list[ai_messages.Message]:
    """Attach pending internal hook messages to the persisted snapshot."""
    result = list(messages)
    for message in pending_hook_messages:
        if len(message.parts) != 1 or not isinstance(
            message.parts[0], ai_messages.HookPart
        ):
            result.append(message)
            continue

        tool_call_id = _tool_call_id_for_hook(message.parts[0])
        turn_id = (
            _turn_id_for_tool_call(messages, tool_call_id)
            if tool_call_id is not None
            else None
        )
        result.append(message.model_copy(update={"turn_id": turn_id}))
    return result
