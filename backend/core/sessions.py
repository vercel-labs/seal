"""Small session store.

A session is metadata plus a JSON message history. Postgres is used when
``DATABASE_URL`` is set; otherwise local development uses one JSONL file per
session, with each line storing the latest full snapshot.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pydantic
from ai import messages as ai_messages

from core import db


class SessionMeta(pydantic.BaseModel):
    id: str
    title: str | None
    created_at: str
    updated_at: str


class SessionRecord(SessionMeta):
    messages: list[ai_messages.Message] = pydantic.Field(default_factory=list)
    active_run_id: str | None = None
    active_stream_id: str | None = None


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    messages    JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS messages JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS active_run_id TEXT;
ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS active_stream_id TEXT;
ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
"""

_DEFAULT_DIR = Path("data/sessions")
_locks: dict[str, asyncio.Lock] = {}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _use_postgres() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _root() -> Path:
    configured = os.environ.get("SEAL_SESSIONS_DIR")
    return Path(configured) if configured else _DEFAULT_DIR


def _path(session_id: str) -> Path:
    return _root() / f"{quote(session_id, safe='')}.jsonl"


def _lock_for(session_id: str) -> asyncio.Lock:
    lock = _locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[session_id] = lock
    return lock


def _dump_messages(messages: list[ai_messages.Message]) -> list[dict[str, Any]]:
    return [
        message.model_dump(mode="json")
        for message in messages
        if message.role != "system"
    ]


def _load_messages(value: Any) -> list[ai_messages.Message]:
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, list):
        return []
    return [ai_messages.Message.model_validate(item) for item in data]


def _meta(record: SessionRecord) -> SessionMeta:
    return SessionMeta(
        id=record.id,
        title=record.title,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _record_from_row(row: Any) -> SessionRecord:
    return SessionRecord(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
        active_run_id=row["active_run_id"],
        active_stream_id=row["active_stream_id"],
        messages=_load_messages(row["messages"]),
    )


def _read_local(session_id: str) -> SessionRecord | None:
    path = _path(session_id)
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        return None
    try:
        return SessionRecord.model_validate_json(lines[-1])
    except pydantic.ValidationError:
        return None


def _write_local(record: SessionRecord) -> None:
    _root().mkdir(parents=True, exist_ok=True)
    with _path(record.id).open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json())
        fh.write("\n")


async def ensure_schema() -> None:
    if not _use_postgres():
        _root().mkdir(parents=True, exist_ok=True)
        return
    pool = await db.get_pool()
    await pool.execute(_SCHEMA)


async def list_sessions() -> list[SessionMeta]:
    if _use_postgres():
        pool = await db.get_pool()
        rows = await pool.fetch(
            "SELECT id, title, created_at, updated_at, active_run_id, "
            "active_stream_id, messages "
            "FROM sessions ORDER BY updated_at DESC"
        )
        return [_meta(_record_from_row(row)) for row in rows]

    if not _root().exists():
        return []
    records = [
        record
        for path in _root().glob("*.jsonl")
        if (record := _read_local(path.stem)) is not None
    ]
    records.sort(key=lambda record: record.updated_at, reverse=True)
    return [_meta(record) for record in records]


async def create_session(session_id: str, title: str | None = None) -> SessionMeta:
    if _use_postgres():
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO sessions (id, title, messages) "
                "VALUES ($1, $2, '[]'::jsonb) "
                "ON CONFLICT (id) DO NOTHING "
                "RETURNING id, title, created_at, updated_at, active_run_id, "
                "active_stream_id, messages",
                session_id,
                title,
            )
            if row is None:
                row = await conn.fetchrow(
                    "SELECT id, title, created_at, updated_at, active_run_id, "
                    "active_stream_id, messages "
                    "FROM sessions WHERE id = $1",
                    session_id,
                )
        return _meta(_record_from_row(row))

    async with _lock_for(session_id):
        existing = _read_local(session_id)
        if existing is not None:
            return _meta(existing)
        now = _now()
        record = SessionRecord(
            id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
            active_run_id=None,
            messages=[],
        )
        _write_local(record)
        return _meta(record)


async def get_session(session_id: str) -> SessionRecord | None:
    if _use_postgres():
        pool = await db.get_pool()
        row = await pool.fetchrow(
            "SELECT id, title, created_at, updated_at, active_run_id, "
            "active_stream_id, messages "
            "FROM sessions WHERE id = $1",
            session_id,
        )
        return _record_from_row(row) if row is not None else None

    async with _lock_for(session_id):
        return _read_local(session_id)


async def save_messages(
    session_id: str,
    messages: list[ai_messages.Message],
) -> SessionMeta:
    if _use_postgres():
        pool = await db.get_pool()
        row = await pool.fetchrow(
            "INSERT INTO sessions (id, messages) "
            "VALUES ($1, $2::jsonb) "
            "ON CONFLICT (id) DO UPDATE SET messages = $2::jsonb, updated_at = now() "
            "RETURNING id, title, created_at, updated_at, active_run_id, "
            "active_stream_id, messages",
            session_id,
            json.dumps(_dump_messages(messages), separators=(",", ":")),
        )
        return _meta(_record_from_row(row))

    async with _lock_for(session_id):
        existing = _read_local(session_id)
        now = _now()
        record = SessionRecord(
            id=session_id,
            title=existing.title if existing else None,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            active_run_id=existing.active_run_id if existing else None,
            active_stream_id=existing.active_stream_id if existing else None,
            messages=_load_messages(_dump_messages(messages)),
        )
        _write_local(record)
        return _meta(record)


async def set_title(session_id: str, title: str) -> SessionMeta | None:
    if _use_postgres():
        pool = await db.get_pool()
        row = await pool.fetchrow(
            "UPDATE sessions SET title = $2, updated_at = now() "
            "WHERE id = $1 "
            "RETURNING id, title, created_at, updated_at, active_run_id, "
            "active_stream_id, messages",
            session_id,
            title,
        )
        return _meta(_record_from_row(row)) if row is not None else None

    async with _lock_for(session_id):
        existing = _read_local(session_id)
        if existing is None:
            return None
        record = existing.model_copy(update={"title": title, "updated_at": _now()})
        _write_local(record)
        return _meta(record)


async def set_active_run(
    session_id: str,
    run_id: str | None,
) -> SessionMeta | None:
    """Set or clear the workflow run currently driving a session."""
    if _use_postgres():
        pool = await db.get_pool()
        row = await pool.fetchrow(
            "UPDATE sessions SET active_run_id = $2, updated_at = now() "
            "WHERE id = $1 "
            "RETURNING id, title, created_at, updated_at, active_run_id, "
            "active_stream_id, messages",
            session_id,
            run_id,
        )
        return _meta(_record_from_row(row)) if row is not None else None

    async with _lock_for(session_id):
        existing = _read_local(session_id)
        if existing is None:
            return None
        record = existing.model_copy(
            update={"active_run_id": run_id, "updated_at": _now()}
        )
        _write_local(record)
        return _meta(record)


async def set_active_stream(
    session_id: str,
    stream_id: str | None,
) -> SessionMeta | None:
    """Set or clear the UI stream currently visible for a session."""
    if _use_postgres():
        pool = await db.get_pool()
        row = await pool.fetchrow(
            "UPDATE sessions SET active_stream_id = $2, updated_at = now() "
            "WHERE id = $1 "
            "RETURNING id, title, created_at, updated_at, active_run_id, "
            "active_stream_id, messages",
            session_id,
            stream_id,
        )
        return _meta(_record_from_row(row)) if row is not None else None

    async with _lock_for(session_id):
        existing = _read_local(session_id)
        if existing is None:
            return None
        record = existing.model_copy(
            update={"active_stream_id": stream_id, "updated_at": _now()}
        )
        _write_local(record)
        return _meta(record)


async def clear_active_stream(
    session_id: str,
    stream_id: str,
) -> SessionMeta | None:
    """Clear the active UI stream if it still points at ``stream_id``."""
    if _use_postgres():
        pool = await db.get_pool()
        row = await pool.fetchrow(
            "UPDATE sessions SET active_stream_id = NULL, updated_at = now() "
            "WHERE id = $1 AND active_stream_id = $2 "
            "RETURNING id, title, created_at, updated_at, active_run_id, "
            "active_stream_id, messages",
            session_id,
            stream_id,
        )
        return _meta(_record_from_row(row)) if row is not None else None

    async with _lock_for(session_id):
        existing = _read_local(session_id)
        if existing is None or existing.active_stream_id != stream_id:
            return None
        record = existing.model_copy(
            update={"active_stream_id": None, "updated_at": _now()}
        )
        _write_local(record)
        return _meta(record)


async def delete_session(session_id: str) -> bool:
    if _use_postgres():
        pool = await db.get_pool()
        result = await pool.execute("DELETE FROM sessions WHERE id = $1", session_id)
        return bool(result == "DELETE 1")

    async with _lock_for(session_id):
        path = _path(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True


def get_new_messages(
    request_messages: list[ai_messages.Message],
    stored_messages: list[ai_messages.Message],
) -> list[ai_messages.Message]:
    """For now, only the request's last user message can be new."""
    latest_user = next(
        (message for message in reversed(request_messages) if message.role == "user"),
        None,
    )
    if latest_user is None:
        return []

    stored_ids = {message.id for message in stored_messages}
    return [] if latest_user.id in stored_ids else [latest_user]
