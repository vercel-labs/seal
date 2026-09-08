"""Run-stream read semantics over the workflow SDK's run streams.

``get_readable``/``replay``/``tail_index`` are the seams every consumer (the
UI bridge, subagent pumps) reads through; a regression here is a missing
message, a duplicate, or a hang. Writes in these tests go through the world
directly — in the app the writers are workflow steps, which make the same
world calls.
"""

from __future__ import annotations

import asyncio

import ai.types.events as events_
import stream_codec
import vercel.workflow._internal.streams as wf_streams
import vercel.workflow._internal.world as wf_world

from agent import proto, stream


async def _write(run_id: str, *events: proto.StreamEvent) -> None:
    world = wf_world.get_world()
    name = wf_streams.workflow_run_stream_id(run_id)
    for event in events:
        data = stream_codec.adapter.dump_python(event, mode="json")
        await world.streams_write(run_id, name, wf_streams.encode_value(data))


async def _close(run_id: str) -> None:
    world = wf_world.get_world()
    await world.streams_close(run_id, wf_streams.workflow_run_stream_id(run_id))


def _chunks(events: list[proto.StreamEvent]) -> list[str]:
    # every event in these tests is a TextDelta; a mismatch means an event
    # was dropped, duplicated, or morphed in transit.
    assert all(isinstance(event, events_.TextDelta) for event in events)
    return [event.chunk for event in events if isinstance(event, events_.TextDelta)]


async def _collect(
    run_id: str, *, start_index: int = 0, timeout: float = 5.0
) -> list[proto.StreamEvent]:
    async def drain() -> list[proto.StreamEvent]:
        return [
            event
            async for event in stream.get_readable(run_id, start_index=start_index)
        ]

    return await asyncio.wait_for(drain(), timeout)


async def test_reader_sees_every_event_exactly_once_and_terminates() -> None:
    async def produce() -> None:
        for n in range(20):
            await _write("r1", events_.TextDelta(block_id="b", chunk=str(n)))
            await asyncio.sleep(0.005)
        await _close("r1")

    producer = asyncio.create_task(produce())
    events = await _collect("r1")
    await producer

    assert _chunks(events) == [str(n) for n in range(20)]


async def test_reader_drains_events_written_just_before_close() -> None:
    # everything already written when the reader starts must still arrive.
    await _write(
        "r1",
        events_.TextDelta(block_id="b", chunk="a"),
        events_.TextDelta(block_id="b", chunk="b"),
    )
    await _close("r1")

    events = await _collect("r1")
    assert _chunks(events) == ["a", "b"]


async def test_replay_reads_whats_there_without_tailing() -> None:
    # the stream stays open; replay must still terminate.
    await _write(
        "r1",
        events_.TextDelta(block_id="b", chunk="a"),
        events_.TextDelta(block_id="b", chunk="b"),
        events_.TextDelta(block_id="b", chunk="c"),
    )

    async def drain() -> list[proto.StreamEvent]:
        return [event async for event in stream.replay("r1", start_index=1)]

    events = await asyncio.wait_for(drain(), 5)
    assert _chunks(events) == ["b", "c"]


async def test_replay_of_unwritten_run_is_empty() -> None:
    assert [event async for event in stream.replay("nope")] == []


async def test_tail_index_tracks_writes() -> None:
    assert await stream.tail_index("r1") == -1
    await _write("r1", events_.TextDelta(block_id="b", chunk="a"))
    assert await stream.tail_index("r1") == 0
    await _write("r1", events_.TextDelta(block_id="b", chunk="b"))
    assert await stream.tail_index("r1") == 1


async def test_agent_events_pass_through_unchanged() -> None:
    sent = events_.TextDelta(block_id="b", chunk="hello")
    await _write("r1", sent)
    await _close("r1")

    [received] = await _collect("r1")
    assert received.model_dump(mode="json") == sent.model_dump(mode="json")
