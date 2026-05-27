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

import db

StreamStatus = Literal["idle", "running", "waiting", "completed", "failed"]


class StreamEvent(pydantic.BaseModel):
    index: int
    data: dict[str, Any]


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS seal_streams (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'idle',
    active_start_idx INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE seal_streams
    ADD COLUMN IF NOT EXISTS active_start_idx INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS seal_stream_events (
    stream_id   TEXT NOT NULL REFERENCES seal_streams(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    data        JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stream_id, idx)
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


_STATUS_VALUES = frozenset({"idle", "running", "waiting", "completed", "failed"})
