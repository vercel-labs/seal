from __future__ import annotations

import json
from typing import Any, cast

import ai
import ai.types.events as events_
import ai.types.messages as messages_
import temporalio.client
from conftest import StreamFixtureWorkflow

from agent import proto, session_workflow_id, stream
from app import chat


def _text_events(text: str) -> list[proto.StreamEvent]:
    message = ai.messages.Message(
        role="assistant", parts=[messages_.TextPart(text=text)]
    )
    return [
        events_.StreamStart(),
        events_.TextStart(block_id="b"),
        events_.TextDelta(block_id="b", chunk=text),
        events_.TextEnd(block_id="b"),
        events_.StreamEnd(message=message),
    ]


async def _start_stream(
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


async def test_active_run_start_index_tracks_open_turn(
    temporal_client: temporalio.client.Client,
) -> None:
    stream_id = session_workflow_id("s1")
    handle = await _start_stream(
        temporal_client,
        stream_id,
        [
            stream.session_started(),
            stream.turn_started(turn_index=0),
            *_text_events("first"),
            stream.session_waiting(turn_index=0),
            stream.turn_started(turn_index=1),
            events_.StreamStart(),
        ],
    )
    try:
        assert await chat.active_run_start_index("s1") == 8
    finally:
        await handle.terminate("test complete")


async def test_completed_turn_is_not_resumable(
    temporal_client: temporalio.client.Client,
) -> None:
    stream_id = session_workflow_id("s2")
    handle = await _start_stream(
        temporal_client,
        stream_id,
        [
            stream.session_started(),
            stream.turn_started(turn_index=0),
            *_text_events("done"),
            stream.session_waiting(turn_index=0),
        ],
    )
    try:
        assert await chat.active_run_start_index("s2") is None
    finally:
        await handle.terminate("test complete")


async def test_to_sse_stops_at_waiting_boundary(
    temporal_client: temporalio.client.Client,
) -> None:
    stream_id = session_workflow_id("s3")
    handle = await _start_stream(
        temporal_client,
        stream_id,
        [
            stream.session_started(),
            stream.turn_started(turn_index=0),
            *_text_events("hello"),
            stream.session_waiting(turn_index=0),
        ],
    )
    try:
        lines = [line async for line in chat.to_sse("s3", 0)]
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in lines
            if line.startswith("data: {")
        ]
        assert any(payload.get("type") == "text-delta" for payload in payloads)
        assert "[DONE]" in lines[-1]
    finally:
        await handle.terminate("test complete")


def test_bundle_to_wire_folds_assistant_messages() -> None:
    nested = chat.bundle_to_wire(
        [
            ai.user_message("task"),
            ai.messages.Message(
                role="assistant", parts=[messages_.TextPart(text="one")]
            ),
            ai.messages.Message(
                role="assistant", parts=[messages_.TextPart(text="two")]
            ),
        ]
    )
    assert nested is not None
    parts = cast(list[dict[str, Any]], nested["parts"])
    texts = [part["text"] for part in parts if part["type"] == "text"]
    assert texts == ["one", "two"]
