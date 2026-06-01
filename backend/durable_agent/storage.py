from __future__ import annotations

import asyncio
import collections.abc
import datetime
import json
import os
import pathlib
import typing
import urllib.parse

import asyncpg  # type: ignore[import-untyped]

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

_schema_ready = False
_schema_lock = asyncio.Lock()
_pool: typing.Any = None
_locks: dict[str, asyncio.Lock] = {}


async def ensure_ready() -> None:
    if not _use_postgres():
        _root().mkdir(parents=True, exist_ok=True)
        return

    global _schema_ready
    if _schema_ready:
        return

    async with _schema_lock:
        if _schema_ready:
            return
        pool = await _get_pool()
        await pool.execute(_SCHEMA)
        _schema_ready = True


async def append_event(
    stream_id: str,
    namespace: str,
    data: dict[str, typing.Any],
) -> int:
    await ensure_ready()
    if _use_postgres():
        return await _append_postgres(stream_id, namespace, data)
    return await _append_jsonl(stream_id, namespace, data)


async def close_stream(stream_id: str, namespace: str) -> None:
    await ensure_ready()
    if _use_postgres():
        await _close_postgres(stream_id, namespace)
        return
    await _close_jsonl(stream_id, namespace)


async def list_events(
    stream_id: str,
    namespace: str,
    start_index: int,
) -> collections.abc.AsyncIterator[tuple[int, dict[str, typing.Any]]]:
    await ensure_ready()
    if _use_postgres():
        async for row in _list_postgres(stream_id, namespace, start_index):
            yield row
        return

    async for row in _list_jsonl(stream_id, namespace, start_index):
        yield row


async def count_events(stream_id: str, namespace: str) -> int:
    await ensure_ready()
    if _use_postgres():
        pool = await _get_pool()
        value = await pool.fetchval(
            "SELECT tail_index FROM seal_durable_streams "
            "WHERE stream_id = $1 AND namespace = $2",
            stream_id,
            namespace,
        )
        return int(value or 0)

    return await _count_jsonl_events(_event_path(stream_id, namespace))


async def is_closed(stream_id: str, namespace: str) -> bool:
    await ensure_ready()
    if _use_postgres():
        pool = await _get_pool()
        value = await pool.fetchval(
            "SELECT closed_at IS NOT NULL FROM seal_durable_streams "
            "WHERE stream_id = $1 AND namespace = $2",
            stream_id,
            namespace,
        )
        return bool(value)

    return _closed_path(stream_id, namespace).exists()


async def _append_postgres(
    stream_id: str,
    namespace: str,
    data: dict[str, typing.Any],
) -> int:
    pool = await _get_pool()
    async with pool.acquire() as connection, connection.transaction():
        await connection.execute(
            "INSERT INTO seal_durable_streams (stream_id, namespace) "
            "VALUES ($1, $2) ON CONFLICT DO NOTHING",
            stream_id,
            namespace,
        )
        row = await connection.fetchrow(
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
        await connection.execute(
            "INSERT INTO seal_durable_stream_events "
            "(stream_id, namespace, idx, data) VALUES ($1, $2, $3, $4)",
            stream_id,
            namespace,
            index,
            json.dumps(data),
        )
        return index


async def _close_postgres(stream_id: str, namespace: str) -> None:
    pool = await _get_pool()
    await pool.execute(
        "INSERT INTO seal_durable_streams (stream_id, namespace, closed_at) "
        "VALUES ($1, $2, now()) "
        "ON CONFLICT (stream_id, namespace) DO UPDATE "
        "SET closed_at = COALESCE(seal_durable_streams.closed_at, now()), "
        "updated_at = now()",
        stream_id,
        namespace,
    )


async def _list_postgres(
    stream_id: str,
    namespace: str,
    start_index: int,
) -> collections.abc.AsyncIterator[tuple[int, dict[str, typing.Any]]]:
    pool = await _get_pool()
    rows = await pool.fetch(
        "SELECT idx, data FROM seal_durable_stream_events "
        "WHERE stream_id = $1 AND namespace = $2 AND idx >= $3 "
        "ORDER BY idx ASC",
        stream_id,
        namespace,
        start_index,
    )
    for row in rows:
        yield int(row["idx"]), _dict_from_json(row["data"])


async def _append_jsonl(
    stream_id: str,
    namespace: str,
    data: dict[str, typing.Any],
) -> int:
    path = _event_path(stream_id, namespace)
    async with _lock_for(str(path)):
        if _closed_path(stream_id, namespace).exists():
            raise RuntimeError("cannot write to a closed stream")

        index = _count_jsonl_events_unlocked(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, separators=(",", ":")))
            stream.write("\n")
            stream.flush()
        return index


async def _close_jsonl(stream_id: str, namespace: str) -> None:
    path = _event_path(stream_id, namespace)
    async with _lock_for(str(path)):
        path.parent.mkdir(parents=True, exist_ok=True)
        _closed_path(stream_id, namespace).write_text(
            datetime.datetime.now(datetime.UTC).isoformat(),
            encoding="utf-8",
        )


async def _list_jsonl(
    stream_id: str,
    namespace: str,
    start_index: int,
) -> collections.abc.AsyncIterator[tuple[int, dict[str, typing.Any]]]:
    path = _event_path(stream_id, namespace)
    records: list[tuple[int, dict[str, typing.Any]]] = []
    async with _lock_for(str(path)):
        if not path.exists():
            return

        index = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            if index >= start_index:
                records.append((index, _dict_from_json(json.loads(line))))
            index += 1

    for record in records:
        yield record


async def _count_jsonl_events(path: pathlib.Path) -> int:
    async with _lock_for(str(path)):
        return _count_jsonl_events_unlocked(path)


def _count_jsonl_events_unlocked(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


async def _get_pool() -> typing.Any:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=os.environ["DATABASE_URL"])
    return _pool


def _use_postgres() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _dict_from_json(value: typing.Any) -> dict[str, typing.Any]:
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, dict):
        raise ValueError("stored stream event is not a JSON object")
    return data


def _root() -> pathlib.Path:
    configured = os.environ.get("SEAL_DURABLE_AGENT_STREAMS_DIR")
    if configured:
        return pathlib.Path(configured)
    return pathlib.Path(__file__).resolve().parents[1] / ".durable_agent_streams"


def _event_path(stream_id: str, namespace: str) -> pathlib.Path:
    return _root() / _quote(stream_id) / f"{_quote(namespace)}.jsonl"


def _closed_path(stream_id: str, namespace: str) -> pathlib.Path:
    return _root() / _quote(stream_id) / f"{_quote(namespace)}.closed"


def _quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def _lock_for(key: str) -> asyncio.Lock:
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    return lock
