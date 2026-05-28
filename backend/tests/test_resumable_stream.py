from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncGenerator, Sequence

from core import resumable_stream


@dataclasses.dataclass
class Stored:
    index: int
    chunk: str


def test_resumable_stream_replays_persisted_chunks() -> None:
    stored: list[str] = []
    status = "running"
    committed = False

    async def list_chunks(
        stream_id: str,
        start_index: int,
    ) -> Sequence[Stored]:
        return [
            Stored(index=index, chunk=chunk)
            for index, chunk in enumerate(stored)
            if index >= start_index
        ]

    async def append_chunk(stream_id: str, chunk: str) -> int:
        stored.append(chunk)
        return len(stored) - 1

    async def get_status(stream_id: str) -> str:
        return status

    async def run() -> tuple[list[str], list[str], bool]:
        nonlocal committed, status

        async def commit() -> None:
            nonlocal committed, status
            committed = True
            status = "completed"

        async def producer() -> AsyncGenerator[str | resumable_stream.ChunkBatch]:
            yield "first"
            yield resumable_stream.ChunkBatch(["second"], commit=commit)

        first = [
            chunk
            async for chunk in resumable_stream.replay_or_produce(
                stream_id="s1",
                start_index=0,
                list_chunks=list_chunks,
                append_chunk=append_chunk,
                get_status=get_status,
                producer=producer,
            )
        ]
        replay = [
            chunk
            async for chunk in resumable_stream.replay_or_produce(
                stream_id="s1",
                start_index=1,
                list_chunks=list_chunks,
                append_chunk=append_chunk,
                get_status=get_status,
                producer=producer,
            )
        ]
        return first, replay, committed

    first, replay, was_committed = asyncio.run(run())

    assert first == ["first", "second"]
    assert replay == ["second"]
    assert stored == ["first", "second"]
    assert was_committed is True


def test_resumable_stream_resolves_negative_start_index() -> None:
    assert resumable_stream.resolve_start_index(-2, 9) == 8
    assert resumable_stream.resolve_start_index(-20, 9) == 0
    assert resumable_stream.resolve_start_index(3, 9) == 3
