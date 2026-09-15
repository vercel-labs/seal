from __future__ import annotations

import asyncio
import datetime
from typing import Any, cast

import ai.types.events as events_
import temporalio.client
import temporalio.contrib.workflow_streams
from conftest import StreamFixtureWorkflow

from agent import proto, stream


async def _start(
    client: temporalio.client.Client,
    stream_id: str,
    events: list[proto.StreamEvent],
) -> temporalio.client.WorkflowHandle[Any, None]:
    return cast(
        temporalio.client.WorkflowHandle[Any, None],
        await client.start_workflow(
            cast(Any, StreamFixtureWorkflow.run),
            events,
            id=stream_id,
            task_queue="seal-temporal",
        ),
    )


async def test_replay_and_tail_index(
    temporal_client: temporalio.client.Client,
) -> None:
    handle = await _start(
        temporal_client,
        "r1",
        [
            events_.TextDelta(block_id="b", chunk="one"),
            events_.TextDelta(block_id="b", chunk="two"),
        ],
    )
    try:
        async with asyncio.timeout(10):
            while await stream.tail_index("r1") < 1:
                await asyncio.sleep(0.01)
        events = [event async for event in stream.replay("r1")]
        assert [
            event.chunk for event in events if isinstance(event, events_.TextDelta)
        ] == ["one", "two"]
        assert await stream.tail_index("r1") == 1
    finally:
        await handle.terminate("test complete")


async def test_readable_tails_new_events(
    temporal_client: temporalio.client.Client,
) -> None:
    handle = await _start(temporal_client, "r2", [])

    async def produce() -> None:
        await asyncio.sleep(0.05)
        stream_client = temporalio.contrib.workflow_streams.WorkflowStreamClient.create(
            temporal_client,
            "r2",
            batch_interval=datetime.timedelta(milliseconds=10),
        )
        topic = stream_client.topic(
            stream.EVENTS_TOPIC, type=cast(Any, proto.StreamEvent)
        )
        async with stream_client:
            topic.publish(
                events_.TextDelta(block_id="b", chunk="hello"),
                force_flush=True,
            )

    producer = asyncio.create_task(produce())
    try:
        async with asyncio.timeout(10):
            async for event in stream.get_readable("r2"):
                assert isinstance(event, events_.TextDelta)
                assert event.chunk == "hello"
                break
        await producer
    finally:
        await handle.terminate("test complete")
