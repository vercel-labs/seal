"""Append-only durable storage for streams and session snapshots.

One backend seam (``_Store``) with two implementations (jsonl, postgres),
selected by ``DATABASE_URL``. Both durable streams (``stream.py``) and session
snapshots (``session.py``) ride this single primitive, keyed by
``(stream_id, namespace)``.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import typing
import urllib.parse

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS seal_durable_streams (
    stream_id  TEXT NOT NULL,
    namespace  TEXT NOT NULL,
    tail_index INTEGER NOT NULL DEFAULT 0,
    closed_at  TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stream_id, namespace)
);

CREATE TABLE IF NOT EXISTS seal_durable_stream_events (
    stream_id  TEXT NOT NULL,
    namespace  TEXT NOT NULL,
    idx        INTEGER NOT NULL,
    data       JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stream_id, namespace, idx),
    FOREIGN KEY (stream_id, namespace)
        REFERENCES seal_durable_streams(stream_id, namespace)
        ON DELETE CASCADE
);
"""


class _Store(typing.Protocol):
    """The only backend seam: an append-only log keyed by (stream_id, namespace)."""

    async def ensure_ready(self) -> None: ...

    async def append(
        self,
        stream_id: str,
        namespace: str,
        data: dict[str, typing.Any],
    ) -> int:
        """Append ``data`` and return its 0-based index."""
        ...

    async def read(
        self,
        stream_id: str,
        namespace: str,
        start_index: int,
    ) -> list[tuple[int, dict[str, typing.Any]]]:
        """Return ``(index, data)`` pairs with ``index >= start_index``."""
        ...

    async def info(self, stream_id: str, namespace: str) -> tuple[int, bool]:
        """Return ``(tail_index, closed)``; ``tail_index`` is -1 when empty."""
        ...

    async def close(self, stream_id: str, namespace: str) -> None: ...


def store() -> _Store:
    """Return the configured store (postgres when ``DATABASE_URL`` is set)."""
    return _PgStore() if os.environ.get("DATABASE_URL") else _JsonlStore()


# --- jsonl --------------------------------------------------------------------

_locks: dict[str, asyncio.Lock] = {}


class _JsonlStore:
    def _paths(
        self, stream_id: str, namespace: str
    ) -> tuple[pathlib.Path, pathlib.Path]:
        configured = os.environ.get("SEAL_STREAMS_DIR")
        root = (
            pathlib.Path(configured)
            if configured
            else pathlib.Path(__file__).resolve().parents[1] / ".seal"
        )
        directory = root / urllib.parse.quote(stream_id, safe="")
        name = urllib.parse.quote(namespace, safe="")
        return directory / f"{name}.jsonl", directory / f"{name}.closed"

    def _lock(self, stream_id: str, namespace: str) -> asyncio.Lock:
        key = f"{stream_id}\x00{namespace}"
        if key not in _locks:
            _locks[key] = asyncio.Lock()
        return _locks[key]

    async def ensure_ready(self) -> None:
        pass

    async def append(
        self,
        stream_id: str,
        namespace: str,
        data: dict[str, typing.Any],
    ) -> int:
        events, closed = self._paths(stream_id, namespace)
        async with self._lock(stream_id, namespace):
            if closed.exists():
                raise RuntimeError("cannot write to a closed stream")
            index = (
                sum(1 for line in events.read_text().splitlines() if line)
                if events.exists()
                else 0
            )
            events.parent.mkdir(parents=True, exist_ok=True)
            with events.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(data, separators=(",", ":")))
                fh.write("\n")
                fh.flush()
            return index

    async def read(
        self,
        stream_id: str,
        namespace: str,
        start_index: int,
    ) -> list[tuple[int, dict[str, typing.Any]]]:
        events, _ = self._paths(stream_id, namespace)
        async with self._lock(stream_id, namespace):
            if not events.exists():
                return []
            records: list[tuple[int, dict[str, typing.Any]]] = []
            for index, line in enumerate(events.read_text().splitlines()):
                if line and index >= start_index:
                    records.append((index, json.loads(line)))
            return records

    async def info(self, stream_id: str, namespace: str) -> tuple[int, bool]:
        events, closed = self._paths(stream_id, namespace)
        async with self._lock(stream_id, namespace):
            count = (
                sum(1 for line in events.read_text().splitlines() if line)
                if events.exists()
                else 0
            )
            return count - 1, closed.exists()

    async def close(self, stream_id: str, namespace: str) -> None:
        _, closed = self._paths(stream_id, namespace)
        async with self._lock(stream_id, namespace):
            closed.parent.mkdir(parents=True, exist_ok=True)
            closed.touch()


# --- postgres -----------------------------------------------------------------

_schema_ready = False
_schema_lock = asyncio.Lock()


class _PgStore:
    async def _pool(self) -> typing.Any:
        import db

        return await db.get_pool()

    async def ensure_ready(self) -> None:
        global _schema_ready
        if _schema_ready:
            return
        async with _schema_lock:
            if _schema_ready:
                return
            await (await self._pool()).execute(_SCHEMA)
            _schema_ready = True

    async def append(
        self,
        stream_id: str,
        namespace: str,
        data: dict[str, typing.Any],
    ) -> int:
        pool = await self._pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO seal_durable_streams (stream_id, namespace) "
                "VALUES ($1, $2) ON CONFLICT DO NOTHING",
                stream_id,
                namespace,
            )
            row = await conn.fetchrow(
                "UPDATE seal_durable_streams "
                "SET tail_index = tail_index + 1, updated_at = now() "
                "WHERE stream_id = $1 AND namespace = $2 AND closed_at IS NULL "
                "RETURNING tail_index - 1 AS idx",
                stream_id,
                namespace,
            )
            if row is None:
                raise RuntimeError("cannot write to a closed stream")
            index = int(row["idx"])
            await conn.execute(
                "INSERT INTO seal_durable_stream_events "
                "(stream_id, namespace, idx, data) VALUES ($1, $2, $3, $4::jsonb)",
                stream_id,
                namespace,
                index,
                json.dumps(data, separators=(",", ":")),
            )
            return index

    async def read(
        self,
        stream_id: str,
        namespace: str,
        start_index: int,
    ) -> list[tuple[int, dict[str, typing.Any]]]:
        pool = await self._pool()
        rows = await pool.fetch(
            "SELECT idx, data FROM seal_durable_stream_events "
            "WHERE stream_id = $1 AND namespace = $2 AND idx >= $3 "
            "ORDER BY idx ASC",
            stream_id,
            namespace,
            start_index,
        )
        return [
            (
                int(row["idx"]),
                json.loads(row["data"])
                if isinstance(row["data"], str)
                else row["data"],
            )
            for row in rows
        ]

    async def info(self, stream_id: str, namespace: str) -> tuple[int, bool]:
        pool = await self._pool()
        row = await pool.fetchrow(
            "SELECT tail_index, closed_at IS NOT NULL AS closed "
            "FROM seal_durable_streams WHERE stream_id = $1 AND namespace = $2",
            stream_id,
            namespace,
        )
        if row is None:
            return -1, False
        return int(row["tail_index"]) - 1, bool(row["closed"])

    async def close(self, stream_id: str, namespace: str) -> None:
        pool = await self._pool()
        await pool.execute(
            "INSERT INTO seal_durable_streams (stream_id, namespace, closed_at) "
            "VALUES ($1, $2, now()) "
            "ON CONFLICT (stream_id, namespace) DO UPDATE "
            "SET closed_at = COALESCE(seal_durable_streams.closed_at, now()), "
            "updated_at = now()",
            stream_id,
            namespace,
        )


async def ensure_ready() -> None:
    """Prepare the configured store (call once at startup)."""
    await store().ensure_ready()
