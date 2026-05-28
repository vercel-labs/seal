"""Resumable async string streams."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Sequence,
)
from typing import Protocol


class StoredChunk(Protocol):
    index: int
    chunk: str


@dataclasses.dataclass
class ChunkBatch:
    """Chunks produced as one commit unit."""

    chunks: list[str]
    commit: Callable[[], Awaitable[None]] | None = None
    rollback: Callable[[], Awaitable[None]] | None = None


StreamStatus = str
ChunkProducer = Callable[[], AsyncIterable[str | ChunkBatch]]
ListChunks = Callable[[str, int], Awaitable[Sequence[StoredChunk]]]
AppendChunk = Callable[[str, str], Awaitable[int]]
GetStatus = Callable[[str], Awaitable[StreamStatus]]

_locks: dict[str, asyncio.Lock] = {}


def _lock_for(stream_id: str) -> asyncio.Lock:
    lock = _locks.get(stream_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[stream_id] = lock
    return lock


def resolve_start_index(start_index: int, tail_index: int) -> int:
    if start_index < 0:
        return max(0, tail_index + 1 + start_index)
    return start_index


def parse_start_index(raw: str | None) -> int:
    if raw is None:
        return 0
    try:
        return int(raw)
    except ValueError:
        raise ValueError("startIndex must be an integer") from None


async def replay_or_produce(
    *,
    stream_id: str,
    start_index: int,
    list_chunks: ListChunks,
    append_chunk: AppendChunk,
    get_status: GetStatus,
    producer: ChunkProducer,
) -> AsyncGenerator[str]:
    """Replay stored chunks, then produce and store missing chunks.

    Only one caller produces for a stream in a process. Other callers tail
    persisted chunks until the stream status is no longer ``running``.
    """
    lock = _lock_for(stream_id)
    if lock.locked():
        async for chunk in tail(
            stream_id=stream_id,
            start_index=start_index,
            list_chunks=list_chunks,
            get_status=get_status,
        ):
            yield chunk
        return

    await lock.acquire()
    try:
        async for chunk in _produce(
            stream_id=stream_id,
            start_index=start_index,
            list_chunks=list_chunks,
            append_chunk=append_chunk,
            get_status=get_status,
            producer=producer,
        ):
            yield chunk
    finally:
        lock.release()


async def tail(
    *,
    stream_id: str,
    start_index: int,
    list_chunks: ListChunks,
    get_status: GetStatus,
) -> AsyncGenerator[str]:
    next_index = start_index
    while True:
        chunks = await list_chunks(stream_id, next_index)
        if chunks:
            for stored_chunk in chunks:
                next_index = stored_chunk.index + 1
                yield stored_chunk.chunk
            continue

        if await get_status(stream_id) != "running":
            return

        await asyncio.sleep(0.5)


async def _produce(
    *,
    stream_id: str,
    start_index: int,
    list_chunks: ListChunks,
    append_chunk: AppendChunk,
    get_status: GetStatus,
    producer: ChunkProducer,
) -> AsyncGenerator[str]:
    next_index = start_index
    for stored_chunk in await list_chunks(stream_id, start_index):
        next_index = stored_chunk.index + 1
        yield stored_chunk.chunk

    if await get_status(stream_id) != "running":
        return

    async for item in producer():
        batch = item if isinstance(item, ChunkBatch) else ChunkBatch([item])
        write_task = asyncio.create_task(_append_batch(stream_id, batch, append_chunk))
        try:
            chunks, batch_next_index = await asyncio.shield(write_task)
        except asyncio.CancelledError:
            chunks, batch_next_index = await write_task
            raise
        if batch_next_index is not None:
            next_index = batch_next_index
        for chunk in chunks:
            yield chunk

    if await get_status(stream_id) == "running":
        async for chunk in tail(
            stream_id=stream_id,
            start_index=next_index,
            list_chunks=list_chunks,
            get_status=get_status,
        ):
            yield chunk


async def _append_batch(
    stream_id: str,
    batch: ChunkBatch,
    append_chunk: AppendChunk,
) -> tuple[list[str], int | None]:
    next_index: int | None = None
    try:
        for chunk in batch.chunks:
            index = await append_chunk(stream_id, chunk)
            next_index = index + 1 if next_index is None else max(next_index, index + 1)
        if batch.commit is not None:
            await batch.commit()
        return batch.chunks, next_index
    except Exception:
        if batch.rollback is not None:
            await batch.rollback()
        raise
