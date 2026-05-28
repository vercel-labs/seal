"""Append-only chat stream event storage."""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import pathlib
import urllib.parse
from typing import Any, Literal, cast

import pydantic

from core import db

StreamStatus = Literal["idle", "running", "waiting", "completed", "failed"]


class StreamEvent(pydantic.BaseModel):
    index: int
    data: dict[str, Any]


class UIStream(pydantic.BaseModel):
    id: str
    session_id: str
    status: StreamStatus
    source_stream_id: str
    source_start_index: int
    source_next_index: int
    history_message_count: int


class UIChunk(pydantic.BaseModel):
    index: int
    chunk: str


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS seal_streams (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'idle',
    active_start_idx INTEGER NOT NULL DEFAULT 0,
    source_stream_id TEXT,
    source_start_idx INTEGER NOT NULL DEFAULT 0,
    source_next_idx INTEGER NOT NULL DEFAULT 0,
    source_claimed_idx INTEGER,
    history_message_count INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE seal_streams
    ADD COLUMN IF NOT EXISTS active_start_idx INTEGER NOT NULL DEFAULT 0;
ALTER TABLE seal_streams
    ADD COLUMN IF NOT EXISTS source_stream_id TEXT;
ALTER TABLE seal_streams
    ADD COLUMN IF NOT EXISTS source_start_idx INTEGER NOT NULL DEFAULT 0;
ALTER TABLE seal_streams
    ADD COLUMN IF NOT EXISTS source_next_idx INTEGER NOT NULL DEFAULT 0;
ALTER TABLE seal_streams
    ADD COLUMN IF NOT EXISTS source_claimed_idx INTEGER;
ALTER TABLE seal_streams
    ADD COLUMN IF NOT EXISTS history_message_count INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS seal_stream_events (
    stream_id   TEXT NOT NULL REFERENCES seal_streams(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    data        JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stream_id, idx)
);

CREATE TABLE IF NOT EXISTS seal_stream_approvals (
    stream_id    TEXT NOT NULL REFERENCES seal_streams(id) ON DELETE CASCADE,
    approval_id  TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    granted      BOOLEAN NOT NULL,
    reason       TEXT,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stream_id, approval_id)
);
"""

_DEFAULT_DIR = pathlib.Path("data/streams")
_locks: dict[str, asyncio.Lock] = {}


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _use_postgres() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _root() -> pathlib.Path:
    configured = os.environ.get("SEAL_STREAMS_DIR")
    return pathlib.Path(configured) if configured else _DEFAULT_DIR


def _path(stream_id: str) -> pathlib.Path:
    return _root() / f"{urllib.parse.quote(stream_id, safe='')}.jsonl"


def _lock_for(stream_id: str) -> asyncio.Lock:
    lock = _locks.get(stream_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[stream_id] = lock
    return lock


def _read_local(stream_id: str) -> list[dict[str, Any]]:
    path = _path(stream_id)
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _load_event_data(value: Any) -> dict[str, Any]:
    data = json.loads(value) if isinstance(value, str) else value
    return data if isinstance(data, dict) else {}


def _write_local(stream_id: str, item: dict[str, Any]) -> None:
    _root().mkdir(parents=True, exist_ok=True)
    with _path(stream_id).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(item, separators=(",", ":")))
        fh.write("\n")


async def ensure_schema() -> None:
    if not _use_postgres():
        _root().mkdir(parents=True, exist_ok=True)
        return
    pool = await db.get_pool()
    await pool.execute(_SCHEMA)


async def ensure_stream(stream_id: str, session_id: str | None = None) -> None:
    session_id = session_id or stream_id
    if _use_postgres():
        pool = await db.get_pool()
        await pool.execute(
            "INSERT INTO seal_streams (id, session_id) VALUES ($1, $2) "
            "ON CONFLICT (id) DO NOTHING",
            stream_id,
            session_id,
        )
        return

    async with _lock_for(stream_id):
        if not _path(stream_id).exists():
            _write_local(
                stream_id,
                {
                    "kind": "meta",
                    "session_id": session_id,
                    "status": "idle",
                    "updated_at": _now(),
                },
            )


async def set_status(stream_id: str, status: StreamStatus) -> None:
    await ensure_stream(stream_id)
    if _use_postgres():
        pool = await db.get_pool()
        await pool.execute(
            "UPDATE seal_streams SET status = $2, updated_at = now() WHERE id = $1",
            stream_id,
            status,
        )
        return

    async with _lock_for(stream_id):
        _write_local(
            stream_id,
            {
                "kind": "meta",
                "status": status,
                "updated_at": _now(),
            },
        )


async def set_active_start_index(stream_id: str, start_index: int) -> None:
    await ensure_stream(stream_id)
    if _use_postgres():
        pool = await db.get_pool()
        await pool.execute(
            "UPDATE seal_streams SET active_start_idx = $2, updated_at = now() "
            "WHERE id = $1",
            stream_id,
            start_index,
        )
        return

    async with _lock_for(stream_id):
        _write_local(
            stream_id,
            {
                "kind": "meta",
                "active_start_index": start_index,
                "updated_at": _now(),
            },
        )


async def get_active_start_index(stream_id: str) -> int:
    await ensure_stream(stream_id)
    if _use_postgres():
        pool = await db.get_pool()
        row = await pool.fetchrow(
            "SELECT active_start_idx FROM seal_streams WHERE id = $1",
            stream_id,
        )
        return int(row["active_start_idx"]) if row is not None else 0

    async with _lock_for(stream_id):
        start_index = 0
        for item in _read_local(stream_id):
            if item.get("kind") != "meta":
                continue
            value = item.get("active_start_index")
            if isinstance(value, int):
                start_index = value
        return start_index


async def get_status(stream_id: str) -> StreamStatus:
    await ensure_stream(stream_id)
    if _use_postgres():
        pool = await db.get_pool()
        row = await pool.fetchrow(
            "SELECT status FROM seal_streams WHERE id = $1",
            stream_id,
        )
        status = row["status"] if row is not None else "idle"
        return cast(StreamStatus, status) if status in _STATUS_VALUES else "idle"

    async with _lock_for(stream_id):
        status = "idle"
        for item in _read_local(stream_id):
            if item.get("kind") == "meta" and item.get("status") in _STATUS_VALUES:
                status = str(item["status"])
        return cast(StreamStatus, status) if status in _STATUS_VALUES else "idle"


async def append_event(stream_id: str, data: dict[str, Any]) -> int:
    await ensure_stream(stream_id)
    if _use_postgres():
        pool = await db.get_pool()
        async with pool.acquire() as conn, conn.transaction():
            index = await conn.fetchval(
                "SELECT COALESCE(MAX(idx), -1) + 1 "
                "FROM seal_stream_events WHERE stream_id = $1",
                stream_id,
            )
            await conn.execute(
                "INSERT INTO seal_stream_events (stream_id, idx, data) "
                "VALUES ($1, $2, $3::jsonb)",
                stream_id,
                index,
                json.dumps(data, separators=(",", ":")),
            )
            return int(index)

    async with _lock_for(stream_id):
        index = sum(1 for item in _read_local(stream_id) if item.get("kind") == "event")
        _write_local(
            stream_id,
            {
                "kind": "event",
                "idx": index,
                "data": data,
                "created_at": _now(),
            },
        )
        return index


async def list_events(stream_id: str, start_index: int) -> list[StreamEvent]:
    await ensure_stream(stream_id)
    if _use_postgres():
        pool = await db.get_pool()
        rows = await pool.fetch(
            "SELECT idx, data FROM seal_stream_events "
            "WHERE stream_id = $1 AND idx >= $2 ORDER BY idx ASC",
            stream_id,
            start_index,
        )
        return [
            StreamEvent(index=row["idx"], data=_load_event_data(row["data"]))
            for row in rows
        ]

    async with _lock_for(stream_id):
        events: list[StreamEvent] = []
        for item in _read_local(stream_id):
            if item.get("kind") != "event":
                continue
            index = int(item.get("idx", -1))
            data = item.get("data")
            if index >= start_index and isinstance(data, dict):
                events.append(StreamEvent(index=index, data=data))
        return events


async def count_events(stream_id: str) -> int:
    await ensure_stream(stream_id)
    if _use_postgres():
        pool = await db.get_pool()
        value = await pool.fetchval(
            "SELECT COUNT(*) FROM seal_stream_events WHERE stream_id = $1",
            stream_id,
        )
        return int(value)

    async with _lock_for(stream_id):
        return sum(1 for item in _read_local(stream_id) if item.get("kind") == "event")


async def create_ui_stream(
    stream_id: str,
    *,
    session_id: str,
    source_stream_id: str,
    source_start_index: int,
    history_message_count: int,
) -> None:
    """Create a client-facing UI stream backed by source agent events."""
    await ensure_stream(stream_id, session_id)
    if _use_postgres():
        pool = await db.get_pool()
        await pool.execute(
            "UPDATE seal_streams SET status = 'running', "
            "source_stream_id = $2, source_start_idx = $3, "
            "source_next_idx = $3, source_claimed_idx = NULL, "
            "history_message_count = $4, "
            "updated_at = now() WHERE id = $1",
            stream_id,
            source_stream_id,
            source_start_index,
            history_message_count,
        )
        return

    async with _lock_for(stream_id):
        _write_local(
            stream_id,
            {
                "kind": "meta",
                "session_id": session_id,
                "status": "running",
                "source_stream_id": source_stream_id,
                "source_start_index": source_start_index,
                "source_next_index": source_start_index,
                "source_claimed_index": None,
                "history_message_count": history_message_count,
                "updated_at": _now(),
            },
        )


async def get_ui_stream(stream_id: str) -> UIStream | None:
    """Return UI stream metadata without creating a missing stream."""
    if _use_postgres():
        pool = await db.get_pool()
        row = await pool.fetchrow(
            "SELECT id, session_id, status, source_stream_id, "
            "source_start_idx, source_next_idx, history_message_count "
            "FROM seal_streams WHERE id = $1",
            stream_id,
        )
        if row is None or row["source_stream_id"] is None:
            return None
        raw_status_value = row["status"] if row["status"] in _STATUS_VALUES else "idle"
        return UIStream(
            id=row["id"],
            session_id=row["session_id"],
            status=cast(StreamStatus, raw_status_value),
            source_stream_id=row["source_stream_id"],
            source_start_index=int(row["source_start_idx"]),
            source_next_index=int(row["source_next_idx"]),
            history_message_count=int(row["history_message_count"]),
        )

    async with _lock_for(stream_id):
        if not _path(stream_id).exists():
            return None
        session_id = stream_id
        status: StreamStatus = "idle"
        source_stream_id: str | None = None
        source_start_index = 0
        source_next_index = 0
        history_message_count = 0
        for item in _read_local(stream_id):
            if item.get("kind") != "meta":
                continue
            raw_session_id = item.get("session_id")
            if isinstance(raw_session_id, str):
                session_id = raw_session_id
            raw_status = item.get("status")
            if raw_status in _STATUS_VALUES:
                status = cast(StreamStatus, raw_status)
            raw_source_stream_id = item.get("source_stream_id")
            if isinstance(raw_source_stream_id, str):
                source_stream_id = raw_source_stream_id
            raw_source_start = item.get("source_start_index")
            if isinstance(raw_source_start, int):
                source_start_index = raw_source_start
            raw_source_next = item.get("source_next_index")
            if isinstance(raw_source_next, int):
                source_next_index = raw_source_next
            raw_history_count = item.get("history_message_count")
            if isinstance(raw_history_count, int):
                history_message_count = raw_history_count
        if source_stream_id is None:
            return None
        return UIStream(
            id=stream_id,
            session_id=session_id,
            status=status,
            source_stream_id=source_stream_id,
            source_start_index=source_start_index,
            source_next_index=source_next_index,
            history_message_count=history_message_count,
        )


async def claim_ui_stream_source_index(
    stream_id: str,
    source_index: int,
) -> bool:
    """Claim the next source event before converting it to UI chunks."""
    await ensure_stream(stream_id)
    if _use_postgres():
        pool = await db.get_pool()
        result = await pool.execute(
            "UPDATE seal_streams SET source_claimed_idx = $2, updated_at = now() "
            "WHERE id = $1 AND status = 'running' AND source_next_idx = $2 "
            "AND source_claimed_idx IS NULL",
            stream_id,
            source_index,
        )
        return str(result) == "UPDATE 1"

    async with _lock_for(stream_id):
        status: StreamStatus = "idle"
        source_next_index = 0
        source_claimed_index: int | None = None
        for item in _read_local(stream_id):
            if item.get("kind") != "meta":
                continue
            raw_status = item.get("status")
            if raw_status in _STATUS_VALUES:
                status = cast(StreamStatus, raw_status)
            raw_source_next = item.get("source_next_index")
            if isinstance(raw_source_next, int):
                source_next_index = raw_source_next
            if "source_claimed_index" in item:
                raw_source_claimed = item.get("source_claimed_index")
                source_claimed_index = (
                    raw_source_claimed if isinstance(raw_source_claimed, int) else None
                )
        if (
            status != "running"
            or source_next_index != source_index
            or source_claimed_index is not None
        ):
            return False
        _write_local(
            stream_id,
            {
                "kind": "meta",
                "source_claimed_index": source_index,
                "updated_at": _now(),
            },
        )
        return True


async def complete_ui_stream_source_index(
    stream_id: str,
    source_index: int,
    source_next_index: int,
) -> None:
    await ensure_stream(stream_id)
    if _use_postgres():
        pool = await db.get_pool()
        await pool.execute(
            "UPDATE seal_streams SET source_next_idx = $3, "
            "source_claimed_idx = NULL, updated_at = now() "
            "WHERE id = $1 AND source_claimed_idx = $2",
            stream_id,
            source_index,
            source_next_index,
        )
        return

    async with _lock_for(stream_id):
        _write_local(
            stream_id,
            {
                "kind": "meta",
                "source_next_index": source_next_index,
                "source_claimed_index": None,
                "updated_at": _now(),
            },
        )


async def release_ui_stream_source_index(
    stream_id: str,
    source_index: int,
) -> None:
    await ensure_stream(stream_id)
    if _use_postgres():
        pool = await db.get_pool()
        await pool.execute(
            "UPDATE seal_streams SET source_claimed_idx = NULL, updated_at = now() "
            "WHERE id = $1 AND source_claimed_idx = $2",
            stream_id,
            source_index,
        )
        return

    async with _lock_for(stream_id):
        _write_local(
            stream_id,
            {
                "kind": "meta",
                "source_claimed_index": None,
                "updated_at": _now(),
            },
        )


async def append_ui_chunk(stream_id: str, chunk: str) -> int:
    return await append_event(stream_id, {"kind": "ui_chunk", "chunk": chunk})


async def list_ui_chunks(stream_id: str, start_index: int) -> list[UIChunk]:
    events = await list_events(stream_id, start_index)
    chunks: list[UIChunk] = []
    for event in events:
        if event.data.get("kind") != "ui_chunk":
            continue
        chunk = event.data.get("chunk")
        if isinstance(chunk, str):
            chunks.append(UIChunk(index=event.index, chunk=chunk))
    return chunks


async def count_ui_chunks(stream_id: str) -> int:
    await ensure_stream(stream_id)
    if _use_postgres():
        pool = await db.get_pool()
        value = await pool.fetchval(
            "SELECT COUNT(*) FROM seal_stream_events "
            "WHERE stream_id = $1 AND data->>'kind' = 'ui_chunk'",
            stream_id,
        )
        return int(value)

    async with _lock_for(stream_id):
        return sum(
            1
            for item in _read_local(stream_id)
            if item.get("kind") == "event"
            and isinstance(item.get("data"), dict)
            and item["data"].get("kind") == "ui_chunk"
        )


async def save_tool_approval(
    stream_id: str,
    *,
    approval_id: str,
    tool_call_id: str,
    granted: bool,
    reason: str | None,
) -> None:
    await ensure_stream(stream_id)
    if _use_postgres():
        pool = await db.get_pool()
        await pool.execute(
            "INSERT INTO seal_stream_approvals "
            "(stream_id, approval_id, tool_call_id, granted, reason) "
            "VALUES ($1, $2, $3, $4, $5) "
            "ON CONFLICT (stream_id, approval_id) DO UPDATE SET "
            "tool_call_id = $3, granted = $4, reason = $5, updated_at = now()",
            stream_id,
            approval_id,
            tool_call_id,
            granted,
            reason,
        )
        return

    async with _lock_for(stream_id):
        _write_local(
            stream_id,
            {
                "kind": "approval",
                "approval_id": approval_id,
                "tool_call_id": tool_call_id,
                "granted": granted,
                "reason": reason,
                "updated_at": _now(),
            },
        )


async def list_tool_approvals(
    stream_id: str,
    approval_ids: list[str],
) -> dict[str, dict[str, Any]]:
    await ensure_stream(stream_id)
    if not approval_ids:
        return {}

    approval_id_set = set(approval_ids)
    if _use_postgres():
        pool = await db.get_pool()
        rows = await pool.fetch(
            "SELECT approval_id, tool_call_id, granted, reason "
            "FROM seal_stream_approvals "
            "WHERE stream_id = $1 AND approval_id = ANY($2::text[])",
            stream_id,
            approval_ids,
        )
        return {
            str(row["approval_id"]): {
                "tool_call_id": str(row["tool_call_id"]),
                "granted": bool(row["granted"]),
                "reason": row["reason"],
            }
            for row in rows
        }

    async with _lock_for(stream_id):
        approvals: dict[str, dict[str, Any]] = {}
        for item in _read_local(stream_id):
            if item.get("kind") != "approval":
                continue
            approval_id = item.get("approval_id")
            if not isinstance(approval_id, str) or approval_id not in approval_id_set:
                continue
            tool_call_id = item.get("tool_call_id")
            granted = item.get("granted")
            if not isinstance(tool_call_id, str) or not isinstance(granted, bool):
                continue
            reason = item.get("reason")
            approvals[approval_id] = {
                "tool_call_id": tool_call_id,
                "granted": granted,
                "reason": reason if isinstance(reason, str) else None,
            }
        return approvals


_STATUS_VALUES = frozenset({"idle", "running", "waiting", "completed", "failed"})
