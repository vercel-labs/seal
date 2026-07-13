"""Session metadata (id, title, timestamps).

History is owned by the durable agent (``session.read_session`` / the durable
event stream); this store only tracks the lightweight metadata the UI needs to
list, title, and delete chats. Postgres when ``DATABASE_URL`` is set, otherwise
one JSON file per session for local dev.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import threading
import urllib.parse
from typing import Any

import ai
import pydantic

import db
from agent import session

_TITLE_PROMPT = (
    "Generate a concise 3-6 word title for a conversation that starts with "
    "the following message. Reply with ONLY the title, no quotes or punctuation."
)
_TITLE_MODEL = "anthropic/claude-sonnet-4.6"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS seal_sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# One lock per session id, guarding the local-file metadata read/modify/write.
# A threading.Lock has no event-loop affinity (so it's safe under any loop or
# thread that touches it) and is only ever held across the synchronous file I/O
# below -- never across an await -- so it can't deadlock the loop.
_locks: dict[str, threading.Lock] = {}


class SessionMeta(pydantic.BaseModel):
    id: str
    title: str | None
    created_at: str
    updated_at: str


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _use_postgres() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _root() -> pathlib.Path:
    configured = os.environ.get("SEAL_SESSIONS_DIR")
    return (
        pathlib.Path(configured)
        if configured
        else pathlib.Path(__file__).resolve().parents[1] / ".seal" / "sessions"
    )


def _path(session_id: str) -> pathlib.Path:
    return _root() / f"{urllib.parse.quote(session_id, safe='')}.json"


def _lock_for(session_id: str) -> threading.Lock:
    """Get a lock that protects against other tasks on the same dev server.

    THIS MUST ONLY BE HELD FOR SYNCHRONOUS OPERATIONS. NO await MAY BE PERFORMED
    WHILE HOLDING IT.
    """
    return _locks.setdefault(session_id, threading.Lock())


def _read_local(session_id: str) -> SessionMeta | None:
    path = _path(session_id)
    if not path.exists():
        return None
    try:
        return SessionMeta.model_validate_json(path.read_text(encoding="utf-8"))
    except pydantic.ValidationError:
        return None


def _write_local(meta: SessionMeta) -> None:
    _root().mkdir(parents=True, exist_ok=True)
    _path(meta.id).write_text(meta.model_dump_json(), encoding="utf-8")


def _from_row(row: Any) -> SessionMeta:
    return SessionMeta(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
    )


async def ensure_schema() -> None:
    if _use_postgres():
        await (await db.get_pool()).execute(_SCHEMA)
    else:
        _root().mkdir(parents=True, exist_ok=True)


async def list_sessions() -> list[SessionMeta]:
    if _use_postgres():
        rows = await (await db.get_pool()).fetch(
            "SELECT id, title, created_at, updated_at "
            "FROM seal_sessions ORDER BY updated_at DESC"
        )
        return [_from_row(row) for row in rows]

    if not _root().exists():
        return []
    metas = [
        meta
        for path in _root().glob("*.json")
        if (meta := _read_local(urllib.parse.unquote(path.stem))) is not None
    ]
    metas.sort(key=lambda meta: meta.updated_at, reverse=True)
    return metas


async def get_session(session_id: str) -> SessionMeta | None:
    if _use_postgres():
        row = await (await db.get_pool()).fetchrow(
            "SELECT id, title, created_at, updated_at FROM seal_sessions WHERE id = $1",
            session_id,
        )
        return _from_row(row) if row is not None else None

    with _lock_for(session_id):
        return _read_local(session_id)


async def create_session(session_id: str, title: str | None = None) -> SessionMeta:
    if _use_postgres():
        async with (await db.get_pool()).acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO seal_sessions (id, title) VALUES ($1, $2) "
                "ON CONFLICT (id) DO NOTHING "
                "RETURNING id, title, created_at, updated_at",
                session_id,
                title,
            ) or await conn.fetchrow(
                "SELECT id, title, created_at, updated_at "
                "FROM seal_sessions WHERE id = $1",
                session_id,
            )
        return _from_row(row)

    with _lock_for(session_id):
        existing = _read_local(session_id)
        if existing is not None:
            return existing
        now = _now()
        meta = SessionMeta(id=session_id, title=title, created_at=now, updated_at=now)
        _write_local(meta)
        return meta


async def touch(session_id: str) -> None:
    """Bump ``updated_at`` (and ensure the row exists) on new activity."""
    await create_session(session_id)
    if _use_postgres():
        await (await db.get_pool()).execute(
            "UPDATE seal_sessions SET updated_at = now() WHERE id = $1", session_id
        )
        return
    with _lock_for(session_id):
        existing = _read_local(session_id)
        if existing is not None:
            _write_local(existing.model_copy(update={"updated_at": _now()}))


async def set_title(session_id: str, title: str) -> SessionMeta | None:
    if _use_postgres():
        row = await (await db.get_pool()).fetchrow(
            "UPDATE seal_sessions SET title = $2, updated_at = now() WHERE id = $1 "
            "RETURNING id, title, created_at, updated_at",
            session_id,
            title,
        )
        return _from_row(row) if row is not None else None

    with _lock_for(session_id):
        existing = _read_local(session_id)
        if existing is None:
            return None
        meta = existing.model_copy(update={"title": title, "updated_at": _now()})
        _write_local(meta)
        return meta


async def delete_session(session_id: str) -> bool:
    if _use_postgres():
        result = await (await db.get_pool()).execute(
            "DELETE FROM seal_sessions WHERE id = $1", session_id
        )
        return bool(result == "DELETE 1")

    with _lock_for(session_id):
        path = _path(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True


async def history(session_id: str) -> list[ai.messages.Message]:
    """Return the durable agent's persisted message history for a session."""
    state = await session.read_session(session_id)
    return state.messages if state is not None else []


async def first_user_text(session_id: str) -> str | None:
    for message in await history(session_id):
        if message.role == "user" and message.text:
            return message.text
    return None


async def generate_title(session_id: str, first_message: str) -> str:
    """Generate a short title from the first user message via a cheap LLM call."""
    messages = [ai.system_message(_TITLE_PROMPT), ai.user_message(first_message)]
    # the named root span keeps this call out of the trace list as an
    # anonymous chat and groups it with its session (``session.id``).
    async with (
        ai.experimental_telemetry.span(
            "generate_title",
            {"session.id": session_id, "input.value": first_message},
        ) as span,
        ai.stream(ai.get_model(_TITLE_MODEL), messages) as stream,
    ):
        async for _ in stream:
            pass
        title = stream.text.strip()
        span.set({"output.value": title})
        return title
