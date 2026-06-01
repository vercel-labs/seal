"""Durable workflow stream side-channel."""

from __future__ import annotations

import asyncio
import collections.abc
import datetime
import json
import os
import pathlib
import urllib.parse
from typing import Any, Literal

import pydantic

type StreamScope = Literal["agent", "llm"]
type StreamRecord = StreamStart | StreamDone | LLMStart | LLMEnd | StreamEvent


class StreamStart(pydantic.BaseModel):
    type: Literal["start"] = "start"
    scope: StreamScope = "agent"
    index: int | None = pydantic.Field(default=None, exclude=True)
    created_at: str = pydantic.Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


class StreamDone(pydantic.BaseModel):
    type: Literal["done"] = "done"
    scope: StreamScope = "agent"
    index: int | None = pydantic.Field(default=None, exclude=True)
    created_at: str = pydantic.Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


class LLMStart(pydantic.BaseModel):
    type: Literal["llm_start"] = "llm_start"
    scope: StreamScope = "llm"
    index: int | None = pydantic.Field(default=None, exclude=True)
    created_at: str = pydantic.Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


class LLMEnd(pydantic.BaseModel):
    type: Literal["llm_end"] = "llm_end"
    scope: StreamScope = "llm"
    index: int | None = pydantic.Field(default=None, exclude=True)
    created_at: str = pydantic.Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


class StreamEvent(pydantic.BaseModel):
    type: str
    scope: StreamScope = "llm"
    data: dict[str, Any]
    index: int | None = pydantic.Field(default=None, exclude=True)
    created_at: str = pydantic.Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


class WritableStream:
    def __init__(
        self,
        store: _JsonlStreamStore,
        *,
        stream_id: str,
        namespace: str,
        reset: bool,
    ) -> None:
        self._store = store
        self._stream_id = stream_id
        self._namespace = namespace
        self._reset = reset

    async def write(self, record: StreamRecord | dict[str, Any]) -> int:
        validated = _RECORD_ADAPTER.validate_python(record)
        reset = self._reset
        self._reset = False
        return await self._store.append(
            self._stream_id,
            self._namespace,
            validated,
            reset=reset,
        )


class _IndexedRecord(pydantic.BaseModel):
    index: int
    record: StreamRecord


_RECORD_ADAPTER: pydantic.TypeAdapter[StreamRecord] = pydantic.TypeAdapter(StreamRecord)
_DEFAULT_NAMESPACE = "default"
_DEFAULT_POLL_INTERVAL = 0.05
_locks: dict[str, asyncio.Lock] = {}


def get_writable(
    stream_id: str,
    *,
    namespace: str = _DEFAULT_NAMESPACE,
    reset: bool = False,
) -> WritableStream:
    """Return a durable writer for workflow/step code."""
    return WritableStream(
        _JsonlStreamStore(),
        stream_id=stream_id,
        namespace=namespace,
        reset=reset,
    )


async def get_readable(
    stream_id: str,
    *,
    namespace: str = _DEFAULT_NAMESPACE,
    start_index: int = 0,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
) -> collections.abc.AsyncIterator[StreamRecord]:
    """Yield validated stream records from ``start_index`` until agent done."""
    store = _JsonlStreamStore()
    next_index = start_index
    if next_index < 0:
        next_index = max(0, await store.count(stream_id, namespace) + next_index)

    while True:
        records = await store.list(stream_id, namespace, next_index)
        if records:
            for item in records:
                next_index = item.index + 1
                record = item.record.model_copy(update={"index": item.index})
                yield record
                if isinstance(record, StreamDone) and record.scope == "agent":
                    return
            continue

        await asyncio.sleep(poll_interval)


class _JsonlStreamStore:
    """Local store with the same append/list shape a PG store should expose."""

    def __init__(self) -> None:
        configured_root = os.environ.get("SEAL_DURABLE_STREAMS_DIR", ".streams")
        self._root = pathlib.Path(configured_root)

    async def append(
        self,
        stream_id: str,
        namespace: str,
        record: StreamRecord,
        *,
        reset: bool,
    ) -> int:
        path = self._path(stream_id, namespace)
        async with _lock_for(str(path)):
            if reset:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            index = (
                0
                if not path.exists()
                else sum(
                    1 for line in path.read_text(encoding="utf-8").splitlines() if line
                )
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        record.model_dump(mode="json", exclude={"index"}),
                        separators=(",", ":"),
                    )
                )
                stream.write("\n")
                stream.flush()
            return index

    async def list(
        self,
        stream_id: str,
        namespace: str,
        start_index: int,
    ) -> list[_IndexedRecord]:
        path = self._path(stream_id, namespace)
        async with _lock_for(str(path)):
            if not path.exists():
                return []

            records: list[_IndexedRecord] = []
            index = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                if index < start_index:
                    index += 1
                    continue
                record = _RECORD_ADAPTER.validate_python(json.loads(line))
                records.append(_IndexedRecord(index=index, record=record))
                index += 1
            return records

    async def count(self, stream_id: str, namespace: str) -> int:
        path = self._path(stream_id, namespace)
        async with _lock_for(str(path)):
            if not path.exists():
                return 0
            return sum(
                1 for line in path.read_text(encoding="utf-8").splitlines() if line
            )

    def _path(self, stream_id: str, namespace: str) -> pathlib.Path:
        safe_stream_id = urllib.parse.quote(stream_id, safe="")
        if namespace == _DEFAULT_NAMESPACE:
            return self._root / f"{safe_stream_id}.jsonl"
        safe_namespace = urllib.parse.quote(namespace, safe="")
        return self._root / f"{safe_stream_id}.{safe_namespace}.jsonl"


def _lock_for(key: str) -> asyncio.Lock:
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    return lock
