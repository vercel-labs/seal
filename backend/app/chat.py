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
  * subagents — a delegated child agent runs as its own durable workflow writing
    to its own run's stream. We tail that child stream concurrently and republish it as
    *preliminary* nested-``UIMessage`` output on the parent's ``subagent`` tool
    call, so the user watches the subagent work live. The driver then stores the
    child's full transcript (a ``MessageBundle``) as the final tool result; both
    the live preliminary output and :func:`bundle_to_wire` (used on reload) reduce
    that transcript to the identical nested ``UIMessage`` shape the UI expects.
"""

from __future__ import annotations

import asyncio
import collections.abc
import contextlib

import ai
import ai.ui.ai_sdk as ai_sdk
import ai.ui.ai_sdk.outbound_stream as outbound_stream
import ai.ui.ai_sdk.ui_events as ui_events
import vercel.workflow

from agent import driver, proto, stream, util

_TERMINAL = {proto.SESSION_WAITING, proto.SESSION_COMPLETED, proto.SESSION_FAILED}


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
    index = -1
    async for event in stream.replay(run_id):
        index += 1
        if not isinstance(event, proto.LifecycleEvent):
            continue
        if event.type in (proto.SESSION_STARTED, proto.TURN_STARTED) and seen_boundary:
            # first opener after a boundary marks where the next run begins.
            run_start = index
            seen_boundary = False
        elif event.type in _TERMINAL:
            seen_boundary = True
    return None if seen_boundary else run_start


async def start_or_resume(session_id: str, prompt: str) -> int:
    """Start a new session or resume a parked one.

    Returns the stream index to tail from so only the new turn reaches the
    client.
    """
    run_id = await stream.session_run_id(session_id)
    if run_id is None:
        # no turn hook: start the session, then wait for the workflow to publish
        # the hook that identifies its run before the response starts tailing it.
        await vercel.workflow.start(
            driver.run_session,
            proto.SessionInput(session_id=session_id, prompt=prompt),
        )
        async for _ in util.hook_retries():
            if await stream.session_run_id(session_id) is not None:
                return 0
        raise RuntimeError(
            f"session workflow did not register its turn hook: {session_id}"
        )

    start_index = await stream.tail_index(run_id) + 1
    await _resume(
        proto.session_hook_token(session_id),
        proto.SessionHook(payload=proto.NewUserMessage(prompt=prompt)),
    )
    return start_index


async def submit_approvals(
    session_id: str, approvals: list[proto.ToolApprovalResponse]
) -> int:
    """Forward each UI approval decision into its own parked hook.

    Returns the stream index to tail the continuation from: the next index after
    the park, computed *before* resuming so the continuation can't outrun it. The
    resubmit carries the parked assistant message, so the client keeps streaming
    into it and the continuation (tool output + answer) folds in.
    """
    run_id = await stream.session_run_id(session_id)
    assert run_id is not None  # approvals only park on a started run
    start_index = await stream.tail_index(run_id) + 1
    for approval in approvals:
        await _resume(
            proto.approval_hook_token(session_id, approval.tool_call_id),
            proto.ApprovalHook(response=approval),
        )
    return start_index


async def to_sse(
    session_id: str, start_index: int
) -> collections.abc.AsyncIterator[str]:
    """Stream one turn of the session as AI SDK UI SSE chunks.

    The parent turn is converted by the SDK adapter. Subagent progress is tailed
    off-thread and interleaved as preliminary tool-output events (the adapter
    never sees them — they sit on tool calls it already started). All events
    funnel through one queue so the merge is sequential.
    """
    queue: asyncio.Queue[ui_events.UIMessageStreamEvent | None] = asyncio.Queue()
    children: list[asyncio.Task[None]] = []

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
        for child in children:
            child.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await adapter_task


async def _turn_events(
    session_id: str,
    start_index: int,
    queue: asyncio.Queue[ui_events.UIMessageStreamEvent | None],
    children: list[asyncio.Task[None]],
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

        if event.type == proto.SUBAGENT_CALLED:
            children.append(asyncio.create_task(_pump_subagent(event, queue)))
        elif event.type == proto.RELOAD_REQUESTED:
            # Tell the client to discard the current step, then keep reading so
            # events from the retried step can use the same connection.
            await queue.put(ui_events.UIFinishStepEvent())
            await queue.put(ui_events.UIDataEvent(data_type="reload", data={}))
            await queue.put(ui_events.UIStartStepEvent())

        elif event.type in _TERMINAL:
            return


def bundle_to_wire(
    messages: collections.abc.Sequence[ai.messages.Message],
) -> dict[str, object] | None:
    """Flatten a subagent transcript into one nested wire ``UIMessage``.

    The child may take several turns; we fold all of its assistant bubbles into a
    single ``UIMessage`` (anchored on the first) so the whole trajectory renders
    under the parent's ``subagent`` tool call. Returns ``None`` when the child has
    produced no assistant message yet.
    """
    # TODO: Do we really need all this? It is very similar to stuff we
    # do in `ai`...

    bubbles = [
        bubble
        for bubble in ai_sdk.to_ui_messages(list(messages))
        if bubble.role == "assistant"
    ]
    if not bubbles:
        return None
    nested = bubbles[0].model_dump(mode="json", by_alias=True)
    nested["parts"] = [
        part
        for bubble in bubbles
        for part in bubble.model_dump(mode="json", by_alias=True)["parts"]
    ]
    return nested


async def _pump_subagent(
    event: proto.LifecycleEvent,
    queue: asyncio.Queue[ui_events.UIMessageStreamEvent | None],
) -> None:
    """Tail a child session stream, republishing it as preliminary tool output.

    Each child ``AgentEvent`` carrying a message is folded into a growing nested
    ``UIMessage`` and pushed as a preliminary ``tool-output-available`` SSE line
    on the parent's ``subagent`` tool call. The final, non-preliminary output is
    the same nested ``UIMessage``, rebuilt on reload from the driver-stored
    ``MessageBundle`` via :func:`bundle_to_wire`.
    """
    tool_call_id = str(event.data.get("tool_call_id"))
    child_run_id = str(event.data.get("child_run_id"))

    child_messages: list[ai.messages.Message] = []
    async for child_event in stream.get_readable(child_run_id, start_index=0):
        if isinstance(child_event, proto.LifecycleEvent):
            continue
        message = getattr(child_event, "message", None)
        if not isinstance(message, ai.messages.Message):
            continue
        _upsert(child_messages, message)
        nested = bundle_to_wire(child_messages)
        if nested is None:
            continue
        await queue.put(
            ui_events.UIToolOutputAvailableEvent(
                tool_call_id=tool_call_id,
                output=nested,
                preliminary=True,
            )
        )


def _upsert(messages: list[ai.messages.Message], message: ai.messages.Message) -> None:
    """Replace the message with the same id, else append it."""
    for index, existing in enumerate(messages):
        if existing.id == message.id:
            messages[index] = message
            return
    messages.append(message)


async def _resume(token: str, hook: vercel.workflow.BaseHook) -> None:
    """Resolve a workflow hook, retrying while the driver registers it."""
    async for last_attempt in util.hook_retries():
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if last_attempt:
                raise
