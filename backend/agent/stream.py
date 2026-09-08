"""Session data on the workflow SDK's run streams.

A session's events live on the default stream of its ``run_session`` workflow
run (a subagent turn writes to its own run's stream), and its state snapshots
on the same run's ``session`` namespace stream. Writes happen inside workflow
steps through ``vercel.workflow`` writers (the owning workflow body gets a
handle from ``vercel.workflow.get_writable()`` and passes it into its steps);
this module owns the wire formats and the read side. Everything is keyed by
run id; a session's long-lived turn hook maps its ``session_id`` to that run.
"""

from __future__ import annotations

import collections.abc
import contextlib

import vercel.workflow

from agent import proto

__all__ = [
    "get_readable",
    "read_session",
    "replay",
    "session_run_id",
    "tail_index",
]

# session snapshots append to this namespace stream of the session's run
# (written by the driver's ``save_session`` step); the tail is the current
# state. The single-writer-per-session invariant (the driver workflow body)
# makes tail reads safe.
SESSION_NAMESPACE = "session"


# lifecycle event constructors


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


def tool_approval_requested() -> proto.LifecycleEvent:
    # the turn parked: every scheduled tool has finished or is awaiting approval.
    return proto.LifecycleEvent(type=proto.TOOL_APPROVAL_REQUESTED)


def reload_requested() -> proto.LifecycleEvent:
    # a retried step just wiped its own aborted first attempt's partial
    # output; a client that already streamed that output needs to reload
    # rather than trust it.
    return proto.LifecycleEvent(type=proto.RELOAD_REQUESTED)


async def get_readable(
    run_id: str, *, start_index: int = 0
) -> collections.abc.AsyncIterator[proto.StreamEvent]:
    """Tail a run's event stream from ``start_index`` until it is closed."""
    source = vercel.workflow.Run(run_id).readable(
        type=proto.StreamEvent, start_index=start_index
    )
    async with contextlib.aclosing(source):
        async for event in source:
            yield event


async def session_run_id(session_id: str) -> str | None:
    """Return the session workflow run advertised by its long-lived turn hook."""
    try:
        hook = await vercel.workflow.get_hook_by_token(
            proto.turn_hook_token(session_id)
        )
    except vercel.workflow.HookNotFoundError:
        return None
    return hook.run_id


async def read_session(run_id: str) -> proto.SessionState | None:
    """Return the latest snapshot for the session run, or ``None`` if absent."""
    run: vercel.workflow.Run[object] = vercel.workflow.Run(run_id)
    tail = (await run.stream_info(namespace=SESSION_NAMESPACE)).tail_index
    if tail < 0:
        return None
    source = run.readable(
        type=proto.SessionState, namespace=SESSION_NAMESPACE, start_index=tail
    )
    async with contextlib.aclosing(source):
        async for state in source:
            return state
    return None


async def tail_index(run_id: str) -> int:
    """Return the last written index (``-1`` when the stream is empty)."""
    info = await vercel.workflow.Run(run_id).stream_info()
    return info.tail_index


async def replay(
    run_id: str, *, start_index: int = 0
) -> collections.abc.AsyncIterator[proto.StreamEvent]:
    """Yield already-written events once, without tailing for new ones."""
    run: vercel.workflow.Run[object] = vercel.workflow.Run(run_id)
    remaining = (await run.stream_info()).tail_index - start_index + 1
    if remaining <= 0:
        return
    source = run.readable(type=proto.StreamEvent, start_index=start_index)
    async with contextlib.aclosing(source):
        async for event in source:
            yield event
            remaining -= 1
            if remaining == 0:
                return
