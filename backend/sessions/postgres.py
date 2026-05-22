"""Postgres-backed session log and repo, used in deployed environments."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any

import asyncpg  # type: ignore[import-untyped]
import pydantic

import db

from .core import (
    ForkPosition,
    Session,
    SessionError,
    SessionLog,
    SessionMeta,
    SessionRepo,
    entries_to_replay,
)
from .entries import (
    ENTRY_ADAPTER,
    SessionEntry,
    SessionInfoEntry,
    leaf_after,
    now_iso,
)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    leaf_id     TEXT,
    cwd         TEXT,
    parent_id   TEXT
);

CREATE TABLE IF NOT EXISTS session_entries (
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    entry_id    TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    parent_id   TEXT,
    type        TEXT NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL,
    entry_json  JSONB NOT NULL,
    PRIMARY KEY (session_id, entry_id),
    UNIQUE (session_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_session_entries_session_seq
    ON session_entries(session_id, seq);
"""


def _parse_jsonb(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)  # type: ignore[no-any-return]
    return value  # type: ignore[no-any-return]


def _meta_from_row(row: asyncpg.Record) -> SessionMeta:
    return SessionMeta(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
    )


class PostgresSessionLog(SessionLog):
    def __init__(
        self,
        *,
        pool: asyncpg.Pool,
        meta: SessionMeta,
        parent_id: str | None,
        entries: list[SessionEntry],
        leaf_id: str | None,
        lock: asyncio.Lock,
    ) -> None:
        super().__init__(
            meta=meta,
            entries=entries,
            leaf_id=leaf_id,
            parent_id=parent_id,
            lock=lock,
        )
        self._pool = pool

    async def _persist(self, entry: SessionEntry) -> None:
        entry_data = entry.model_dump(mode="json")
        ts = datetime.fromisoformat(entry.timestamp)
        async with self._pool.acquire() as conn, conn.transaction():
            seq = await conn.fetchval(
                "SELECT COALESCE(MAX(seq) + 1, 0) "
                "FROM session_entries WHERE session_id = $1",
                self.id,
            )
            await conn.execute(
                "INSERT INTO session_entries "
                "(session_id, entry_id, seq, parent_id, type, timestamp, entry_json) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)",
                self.id,
                entry.id,
                seq,
                entry.parent_id,
                entry.type,
                ts,
                json.dumps(entry_data),
            )
            new_leaf = leaf_after(entry)
            if isinstance(entry, SessionInfoEntry):
                await conn.execute(
                    "UPDATE sessions SET leaf_id = $2, title = $3, "
                    "updated_at = now() WHERE id = $1",
                    self.id,
                    new_leaf,
                    entry.title,
                )
            else:
                await conn.execute(
                    "UPDATE sessions SET leaf_id = $2, updated_at = now() "
                    "WHERE id = $1",
                    self.id,
                    new_leaf,
                )


class PostgresSessionRepo(SessionRepo):
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._cache: dict[str, PostgresSessionLog] = {}

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    async def ensure_schema(self) -> None:
        pool = await db.get_pool()
        await pool.execute(_SCHEMA)

    async def _load_entries(
        self, conn: Any, session_id: str
    ) -> list[SessionEntry]:
        rows = await conn.fetch(
            "SELECT entry_json FROM session_entries "
            "WHERE session_id = $1 ORDER BY seq",
            session_id,
        )
        entries: list[SessionEntry] = []
        for line_no, row in enumerate(rows, start=1):
            try:
                data = _parse_jsonb(row["entry_json"])
                entries.append(ENTRY_ADAPTER.validate_python(data))
            except (json.JSONDecodeError, pydantic.ValidationError) as exc:
                raise SessionError(
                    "invalid_entry",
                    f"Invalid entry #{line_no} in session {session_id}",
                    cause=exc,
                ) from exc
        return entries

    async def _build_log(
        self,
        conn: Any,
        row: asyncpg.Record,
        lock: asyncio.Lock,
    ) -> PostgresSessionLog:
        entries = await self._load_entries(conn, row["id"])
        leaf_id = None
        for entry in entries:
            leaf_id = leaf_after(entry)
        pool = await db.get_pool()
        return PostgresSessionLog(
            pool=pool,
            meta=_meta_from_row(row),
            parent_id=row["parent_id"],
            entries=entries,
            leaf_id=leaf_id,
            lock=lock,
        )

    async def create(
        self,
        session_id: str,
        *,
        title: str | None = None,
        parent_id: str | None = None,
    ) -> Session:
        await self.ensure_schema()
        lock = self._lock_for(session_id)
        async with lock:
            cached = self._cache.get(session_id)
            if cached is not None:
                return Session(cached)

            pool = await db.get_pool()
            async with pool.acquire() as conn, conn.transaction():
                row = await conn.fetchrow(
                    "SELECT id, title, created_at, updated_at, leaf_id, parent_id "
                    "FROM sessions WHERE id = $1 FOR UPDATE",
                    session_id,
                )
                created_now = row is None
                if row is None:
                    row = await conn.fetchrow(
                        "INSERT INTO sessions (id, cwd, parent_id) "
                        "VALUES ($1, $2, $3) "
                        "RETURNING id, title, created_at, updated_at, leaf_id, "
                        "parent_id",
                        session_id,
                        os.getcwd(),
                        parent_id,
                    )
                log = await self._build_log(conn, row, lock)

            self._cache[session_id] = log
            if created_now and title is not None:
                entry = SessionInfoEntry(
                    id=log.make_entry_id(),
                    parent_id=log.leaf_id(),
                    timestamp=now_iso(),
                    title=title,
                )
                await log.append(entry)
            return Session(log)

    async def open(self, session_id: str) -> Session | None:
        lock = self._lock_for(session_id)
        async with lock:
            cached = self._cache.get(session_id)
            if cached is not None:
                return Session(cached)

            pool = await db.get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT id, title, created_at, updated_at, leaf_id, parent_id "
                    "FROM sessions WHERE id = $1",
                    session_id,
                )
                if row is None:
                    return None
                log = await self._build_log(conn, row, lock)
            self._cache[session_id] = log
            return Session(log)

    async def list(self) -> list[SessionMeta]:
        pool = await db.get_pool()
        rows = await pool.fetch(
            "SELECT id, title, created_at, updated_at "
            "FROM sessions ORDER BY updated_at DESC"
        )
        return [_meta_from_row(r) for r in rows]

    async def delete(self, session_id: str) -> bool:
        lock = self._lock_for(session_id)
        async with lock:
            pool = await db.get_pool()
            result = await pool.execute(
                "DELETE FROM sessions WHERE id = $1", session_id
            )
            self._cache.pop(session_id, None)
            return bool(result == "DELETE 1")

    async def fork(
        self,
        source_id: str,
        new_id: str,
        *,
        entry_id: str | None = None,
        position: ForkPosition = "at",
    ) -> Session:
        source = await self.open(source_id)
        if source is None:
            raise SessionError(
                "not_found", f"Source session {source_id} not found"
            )
        async with source.log.lock:
            replay = await entries_to_replay(
                source.log, entry_id=entry_id, position=position
            )

        new_session = await self.create(new_id, parent_id=source_id)
        async with new_session.log.lock:
            for entry in replay:
                await new_session.log.append(entry)
        return new_session
