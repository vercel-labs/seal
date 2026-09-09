"""Bridge the durable session protocol to the AI SDK UI message stream.

The durable agent persists ``ai.events.AgentEvent | proto.LifecycleEvent`` to a
per-session stream. The browser's ``useChat`` speaks the AI SDK UI protocol, so
we tail one turn of the durable stream, hand the ``AgentEvent``s to the SDK's
``to_stream`` adapter (lifecycle events stay server-side), and drive control
flow off the lifecycle events.

Two lifecycle features surface to the UI:

  * tool approvals — a gated tool emits a ``tool-approval-request`` part (built by
    the SDK adapter from the pending hook). The turn parks; the browser replies
    with ``addToolApprovalResponse`` which arrives on the next ``POST /chat`` and
    is forwarded back into the durable hook by :func:`submit_approvals`.
  * subagents — a delegated child runs as background work owned by the session.
    The tool returns an immediate acknowledgement; when all children finish, the
    session injects their reports as a hidden user message and starts another turn.
"""

from __future__ import annotations

import asyncio
import collections.abc
import contextlib
import typing

import ai
import ai.ui.ai_sdk as ai_sdk
import ai.ui.ai_sdk.outbound_stream as outbound_stream
import ai.ui.ai_sdk.ui_events as ui_events
import vercel.workflow

from agent import driver, proto, stream, workflow_util

_SESSION_TERMINAL = {
    proto.SESSION_COMPLETED,
    proto.SESSION_FAILED,
}


class SessionUnavailableError(RuntimeError):
    pass


async def active_run_start_index(session_id: str) -> int | None:
    """Return the stream index to resume the in-flight run from, else ``None``.

    A *run* is the whole agent turn the SDK adapter folds into a single UI
    message (one ``UIStartEvent``). It begins at its opener (``session.started``
    / ``turn.started``) and ends only at a *terminal* boundary (``session.*``) —
    a ``tool_approval.requested`` park is mid-run, not the end of one, because the
    same ``run_turn`` resumes in place once the approval lands.

    Used only by the cold-reload path (``GET /chat/{id}/stream``): there is no
    submitted message to continue, so re-tailing from the opener replays the
    assistant message's stable id and the SDK rebuilds the same UI message. (The
    live approval POST doesn't need this — its resubmit already carries the
    assistant message, so its continuation folds in; see :func:`submit_approvals`.)

    The run is in flight (resumable) when its opener has no terminal after it.
    """
    run_id = await stream.session_run_id(session_id)
    if run_id is None:
        return None
    run_start: int | None = None
    seen_boundary = True
    session_opener = False
    index = -1
    async for event in stream.replay(run_id):
        index += 1
        if not isinstance(event, proto.LifecycleEvent):
            continue
        if event.type == proto.TURN_STARTED and (seen_boundary or session_opener):
            run_start = index
            seen_boundary = False
            session_opener = False
        elif event.type == proto.SESSION_STARTED and seen_boundary:
            run_start = index
            seen_boundary = False
            session_opener = True
        elif event.type == proto.SESSION_WAITING:
            seen_boundary = event.data.get("active_background_tasks", 0) == 0
        elif event.type == proto.SESSION_INTERRUPTED or event.type in _SESSION_TERMINAL:
            seen_boundary = True
    return None if seen_boundary else run_start


async def start_or_resume(session_id: str, prompt: str) -> int:
    """Start a new session or resume a parked one.

    Returns the stream index to tail from so only the new turn reaches the
    client.
    """
    if await active_run_start_index(session_id) is not None:
        raise SessionUnavailableError("A turn is already running")

    run_id = await stream.session_run_id(session_id)
    if run_id is None:
        # no turn hook: start the session, then wait for the workflow to publish
        # the hook that identifies its run before the response starts tailing it.
        await workflow_util.start(
            workflow_util.with_hooks(
                driver.run_session, [proto.turn_hook_token(session_id)]
            ),
            proto.SessionInput(session_id=session_id, prompt=prompt),
        )
        return 0

    start_index = await stream.tail_index(run_id) + 1
    await proto.SessionHook(payload=proto.NewUserMessage(prompt=prompt)).resume(
        proto.session_hook_token(session_id)
    )
    return start_index


async def submit_approvals(
    session_id: str, approvals: list[proto.ToolApprovalResponse]
) -> int:
    """Forward each UI approval decision into the session's shared hook.

    Resume after the durable approval marker: background lifecycle events may
    arrive while the human is deciding and must still reach the resumed stream.
    """
    start_index = await _latest_event_index(session_id, ai.events.RunBlocked) + 1
    await proto.ApprovalHook(responses=approvals).resume(
        proto.hooks_hook_token(session_id)
    )
    return start_index


async def interrupt(session_id: str) -> None:
    """Interrupt the active turn and wait for its durable stream boundary."""
    run_id = await stream.session_run_id(session_id)
    if run_id is None:
        raise SessionUnavailableError("Session is not running")

    start_index = await stream.tail_index(run_id) + 1
    try:
        await proto.InterruptHook().resume(proto.interrupt_hook_token(session_id))
    except vercel.workflow.HookNotFoundError:
        raise SessionUnavailableError("Session has no active turn") from None

    async with asyncio.timeout(30):
        async for event in stream.get_readable(run_id, start_index=start_index):
            if (
                isinstance(event, proto.LifecycleEvent)
                and event.type == proto.SESSION_INTERRUPTED
            ):
                return

    raise RuntimeError("Session stream closed before interruption was acknowledged")


async def to_sse(
    session_id: str, start_index: int
) -> collections.abc.AsyncIterator[str]:
    """Stream one logical UI run as AI SDK UI SSE chunks.

    The parent turn is converted by the SDK adapter. Subagent progress is tailed
    off-thread and interleaved as preliminary tool-output events (the adapter
    never sees them — they sit on tool calls it already started). All events
    funnel through one queue so the merge is sequential.
    """
    queue: asyncio.Queue[ui_events.UIMessageStreamEvent | None] = asyncio.Queue()
    children: dict[str, asyncio.Task[None]] = {}

    async def pump_adapter() -> None:
        events = _turn_events(session_id, start_index, queue, children)
        async for event in ai_sdk.to_stream(events):
            await queue.put(event)
        await queue.put(None)

    adapter_task = asyncio.create_task(pump_adapter())
    try:
        while True:
            line = await queue.get()
            if line is None:
                break
            formatted = outbound_stream.format_sse(line)
            yield formatted
        yield outbound_stream.format_done_sse()
    finally:
        adapter_task.cancel()
        for child in children.values():
            child.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await adapter_task


async def _turn_events(
    session_id: str,
    start_index: int,
    queue: asyncio.Queue[ui_events.UIMessageStreamEvent | None],
    children: dict[str, asyncio.Task[None]],
) -> collections.abc.AsyncIterator[ai.events.AgentEvent]:
    """Yield this turn's ``AgentEvent``s, ending at the next turn boundary.

    Lifecycle events stay server-side: ``subagent.called`` spins up a concurrent
    tail of the child stream (collected in ``children`` so the caller can cancel
    it; its progress lines go straight onto ``queue``), and the loop returns once
    the turn parks on an approval or finishes.

    An approval resume just tails the continuation from after the park; its first
    event is a tool result (no ``turn_id`` → id-less ``start``), so the client
    folds it into the assistant message it resubmitted.

    ``reload.requested`` is a permanent stream entry, not a one-shot signal --
    a later connection replaying history past it sees it again. Every connection
    forwards the reload marker and continues reading, so the client can discard
    the current step before applying events from the retried step.
    """
    run_id = await stream.session_run_id(session_id)
    assert run_id is not None  # both endpoints guarantee the run has started
    async for event in stream.get_readable(run_id, start_index=start_index):
        if isinstance(event, ai.events.RunBlocked):
            yield event
            return
        if not isinstance(event, proto.LifecycleEvent):
            yield event  # ai.events.AgentEvent
            continue

        task_id = event.data.get("tool_call_id")
        if event.type == proto.SUBAGENT_CALLED and isinstance(task_id, str):
            children[task_id] = asyncio.create_task(_pump_subagent(event, queue))
        elif event.type == proto.SUBAGENT_COMPLETED and isinstance(task_id, str):
            child = children.pop(task_id, None)
            if child is not None:
                child.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await child
            if event.data.get("is_error"):
                await queue.put(
                    ui_events.UIToolOutputErrorEvent(
                        tool_call_id=task_id,
                        error_text=str(
                            event.data.get("error") or "Unknown subagent error"
                        ),
                    )
                )
            else:
                messages = [
                    ai.messages.Message.model_validate(message)
                    for message in event.data.get("messages", [])
                ]
                await queue.put(
                    ui_events.UIToolOutputAvailableEvent(
                        tool_call_id=task_id,
                        output=background_task_output(messages),
                        preliminary=False,
                    )
                )
        elif event.type == proto.RELOAD_REQUESTED:
            # Tell the client to discard the current step, then keep reading so
            # events from the retried step can use the same connection.
            await queue.put(ui_events.UIFinishStepEvent())
            await queue.put(ui_events.UIDataEvent(data_type="reload", data={}))
            await queue.put(ui_events.UIStartStepEvent())

        elif event.type == proto.SESSION_WAITING:
            if event.data.get("active_background_tasks", 0) == 0:
                return
        elif event.type == proto.SESSION_INTERRUPTED or event.type in _SESSION_TERMINAL:
            return


def background_task_output(
    messages: collections.abc.Sequence[ai.messages.Message],
) -> list[dict[str, object]]:
    """Return child assistant bubbles in the nested UI shape."""
    return [
        message.model_dump(mode="json", by_alias=True)
        for message in ai_sdk.to_ui_messages(list(messages))
        if message.role == "assistant"
    ]


def project_background_tasks(
    ui_messages: list[dict[str, object]],
    tasks: collections.abc.Mapping[str, proto.BackgroundTaskState],
) -> list[dict[str, object]]:
    """Overlay durable task state without changing model history."""
    for message in ui_messages:
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        for raw_part in parts:
            if not isinstance(raw_part, dict):
                continue
            part = typing.cast(dict[str, object], raw_part)
            task_id = part.get("toolCallId")
            task = tasks.get(task_id) if isinstance(task_id, str) else None
            if task is None:
                continue
            if task.status == "failed":
                part["state"] = "output-error"
                part["errorText"] = task.error or "Unknown subagent error"
                part.pop("output", None)
                part.pop("preliminary", None)
            else:
                part["state"] = "output-available"
                if task.status == "completed" or task.messages:
                    part["output"] = background_task_output(task.messages)
                part["preliminary"] = task.status == "running"
    return ui_messages


async def _pump_subagent(
    event: proto.LifecycleEvent,
    queue: asyncio.Queue[ui_events.UIMessageStreamEvent | None],
) -> None:
    """Tail a child session stream, republishing it as preliminary tool output.

    Each child ``AgentEvent`` carrying a message is folded into nested assistant
    output and pushed as a preliminary ``tool-output-available`` SSE line on the
    parent's original ``subagent`` tool call.
    """
    tool_call_id = str(event.data.get("tool_call_id"))
    child_run_id = str(event.data.get("child_run_id"))

    hydrator = ai.events.MessageHydrator()
    async for child_event in stream.get_readable(child_run_id, start_index=0):
        if isinstance(child_event, proto.LifecycleEvent):
            continue
        child_event = hydrator.feed(child_event)
        output = background_task_output(hydrator.messages)
        if not output:
            continue
        await queue.put(
            ui_events.UIToolOutputAvailableEvent(
                tool_call_id=tool_call_id,
                output=output,
                preliminary=True,
            )
        )


async def _latest_event_index(session_id: str, event_type: type[object]) -> int:
    run_id = await stream.session_run_id(session_id)
    if run_id is None:
        raise SessionUnavailableError("Session is not running")
    found: int | None = None
    index = -1
    async for event in stream.replay(run_id):
        index += 1
        if isinstance(event, event_type):
            found = index
    if found is None:
        raise SessionUnavailableError(f"No {event_type.__name__} event to resume from")
    return found
