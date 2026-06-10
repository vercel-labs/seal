"""Durable stream read/write semantics.

``get_readable`` is the seam every consumer (the UI bridge, subagent pumps)
tails; a regression here is a missing message, a duplicate, or a hang.
"""

from __future__ import annotations

import asyncio

import ai.types.events as events_

from agent import proto, stream


def _chunks(events: list[proto.StreamEvent]) -> list[str]:
    # every event in these tests is a TextDelta; a mismatch means an event
    # was dropped, duplicated, or morphed in transit.
    assert all(isinstance(event, events_.TextDelta) for event in events)
    return [event.chunk for event in events if isinstance(event, events_.TextDelta)]


async def _collect(
    session_id: str, *, start_index: int = 0, timeout: float = 5.0
) -> list[proto.StreamEvent]:
    async def drain() -> list[proto.StreamEvent]:
        return [
            event
            async for event in stream.get_readable(
                session_id, start_index=start_index, poll_interval=0.01
            )
        ]

    return await asyncio.wait_for(drain(), timeout)


async def test_reader_sees_every_event_exactly_once_and_terminates() -> None:
    writer = await stream.get_writable("s1")

    async def produce() -> None:
        for n in range(20):
            await writer.write(events_.TextDelta(block_id="b", chunk=str(n)))
            await asyncio.sleep(0.005)
        await writer.close()

    producer = asyncio.create_task(produce())
    events = await _collect("s1")
    await producer

    assert _chunks(events) == [str(n) for n in range(20)]


async def test_reader_drains_events_written_just_before_close() -> None:
    # everything already written when the reader starts must still arrive.
    writer = await stream.get_writable("s1")
    await writer.write(events_.TextDelta(block_id="b", chunk="a"))
    await writer.write(events_.TextDelta(block_id="b", chunk="b"))
    await writer.close()

    events = await _collect("s1")
    assert _chunks(events) == ["a", "b"]


async def test_negative_start_index_reads_the_tail() -> None:
    writer = await stream.get_writable("s1")
    for n in range(5):
        await writer.write(events_.TextDelta(block_id="b", chunk=str(n)))
    await writer.close()

    events = await _collect("s1", start_index=-2)
    assert _chunks(events) == ["3", "4"]


async def test_lifecycle_events_are_stamped_on_write() -> None:
    # workflow bodies can't call datetime.now(); the write seam stamps ``at``.
    writer = await stream.get_writable("s1")
    await writer.write(stream.session_started(mode="infinite"))
    await writer.close()

    [event] = await _collect("s1")
    assert isinstance(event, proto.LifecycleEvent)
    assert event.type == proto.SESSION_STARTED
    assert event.at is not None


async def test_agent_events_pass_through_unchanged() -> None:
    writer = await stream.get_writable("s1")
    sent = events_.TextDelta(block_id="b", chunk="hello")
    await writer.write(sent)
    await writer.close()

    [received] = await _collect("s1")
    assert received.model_dump(mode="json") == sent.model_dump(mode="json")
