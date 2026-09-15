"""Temporal Workflow Streams adapter for agent events."""

from __future__ import annotations

import collections.abc
import datetime
from typing import Any, cast

import temporalio.contrib.workflow_streams
import temporalio.service

from agent import proto

EVENTS_TOPIC = "events"


def session_started() -> proto.LifecycleEvent:
    return proto.LifecycleEvent(type=proto.SESSION_STARTED)


def session_waiting(*, turn_index: int) -> proto.LifecycleEvent:
    return proto.LifecycleEvent(
        type=proto.SESSION_WAITING, data={"turn_index": turn_index}
    )


def session_completed(*, is_error: bool = False) -> proto.LifecycleEvent:
    return proto.LifecycleEvent(
        type=proto.SESSION_COMPLETED, data={"is_error": is_error}
    )


def session_interrupted() -> proto.LifecycleEvent:
    return proto.LifecycleEvent(type=proto.SESSION_INTERRUPTED)


def turn_started(*, turn_index: int) -> proto.LifecycleEvent:
    return proto.LifecycleEvent(
        type=proto.TURN_STARTED, data={"turn_index": turn_index}
    )


def subagent_called(
    *, tool_call_id: str, child_session_id: str, child_run_id: str, name: str
) -> proto.LifecycleEvent:
    return proto.LifecycleEvent(
        type=proto.SUBAGENT_CALLED,
        data={
            "tool_call_id": tool_call_id,
            "child_session_id": child_session_id,
            "child_run_id": child_run_id,
            "name": name,
        },
    )


def subagent_completed(*, tool_call_id: str, is_error: bool) -> proto.LifecycleEvent:
    return proto.LifecycleEvent(
        type=proto.SUBAGENT_COMPLETED,
        data={"tool_call_id": tool_call_id, "is_error": is_error},
    )


def reload_requested() -> proto.LifecycleEvent:
    return proto.LifecycleEvent(type=proto.RELOAD_REQUESTED)


async def _client(
    stream_id: str,
) -> temporalio.contrib.workflow_streams.WorkflowStreamClient:
    from agent import temporal

    return temporalio.contrib.workflow_streams.WorkflowStreamClient.create(
        await temporal.get_client(), stream_id
    )


async def offset(stream_id: str) -> int:
    try:
        return await (await _client(stream_id)).get_offset()
    except temporalio.service.RPCError as error:
        if error.status == temporalio.service.RPCStatusCode.NOT_FOUND:
            return 0
        raise


async def tail_index(stream_id: str) -> int:
    return await offset(stream_id) - 1


async def replay(
    stream_id: str, *, start_index: int = 0
) -> collections.abc.AsyncIterator[proto.StreamEvent]:
    stream_client = await _client(stream_id)
    end_offset = await stream_client.get_offset()
    if end_offset <= start_index:
        return
    async for item in stream_client.subscribe(
        [EVENTS_TOPIC],
        from_offset=start_index,
        result_type=cast(Any, proto.StreamEvent),
        poll_cooldown=datetime.timedelta(milliseconds=20),
    ):
        yield cast(proto.StreamEvent, item.data)
        if item.offset + 1 >= end_offset:
            return


async def get_readable(
    stream_id: str, *, start_index: int = 0
) -> collections.abc.AsyncIterator[proto.StreamEvent]:
    async for item in (await _client(stream_id)).subscribe(
        [EVENTS_TOPIC],
        from_offset=start_index,
        result_type=cast(Any, proto.StreamEvent),
        poll_cooldown=datetime.timedelta(milliseconds=50),
    ):
        yield cast(proto.StreamEvent, item.data)
