"""Bridge the durable session protocol to the AI SDK UI message stream.

The durable agent persists ``ai.events.AgentEvent | proto.LifecycleEvent`` to a
per-session stream. The browser's ``useChat`` speaks the AI SDK UI protocol, so
we tail one turn of the durable stream, hand the ``AgentEvent``s straight to the
SDK's ``to_sse`` adapter (lifecycle events stay server-side), and drive control
flow off the lifecycle events: auto-approve gated tools and stop at the turn
boundary.

Tools are auto-approved for now; there is no human-in-the-loop yet.
"""

from __future__ import annotations

import asyncio
import collections.abc

import ai
import ai.agents.ui.ai_sdk as ai_sdk
import vercel.workflow

from agent import driver, proto, session, stream

_TERMINAL = {proto.SESSION_WAITING, proto.SESSION_COMPLETED, proto.SESSION_FAILED}


async def start_or_resume(session_id: str, prompt: str) -> int:
    """Start a new session or resume a parked one.

    Returns the stream index to tail from so only the new turn reaches the
    client.
    """
    start_index = await stream.tail_index(session_id) + 1

    if await session.read_session(session_id) is None:
        await vercel.workflow.start(
            driver.run_session,
            proto.SessionInput(session_id=session_id, prompt=prompt).model_dump(
                mode="json"
            ),
        )
    else:
        turn_index = await _waiting_turn_index(session_id)
        await _resume(
            f"seal-session:{session_id}:{turn_index}",
            proto.NewUserMessage(prompt=prompt),
        )
    return start_index


async def to_sse(
    session_id: str, start_index: int
) -> collections.abc.AsyncIterator[str]:
    """Stream one turn of the session as AI SDK UI SSE chunks."""
    async for chunk in ai_sdk.to_sse(_turn_events(session_id, start_index)):
        yield chunk


async def _turn_events(
    session_id: str, start_index: int
) -> collections.abc.AsyncIterator[ai.events.AgentEvent]:
    """Yield this turn's ``AgentEvent``s, ending at the next turn boundary.

    Lifecycle events never reach the UI: ``tool_approval.requested`` is
    auto-approved here, and the loop returns once the session parks or finishes.
    """
    async for event in stream.get_readable(session_id, start_index=start_index):
        if not isinstance(event, proto.LifecycleEvent):
            yield event
            continue

        if event.type == proto.TOOL_APPROVAL_REQUESTED:
            await _auto_approve(session_id, event)
        elif event.type in _TERMINAL:
            return


async def _auto_approve(session_id: str, event: proto.LifecycleEvent) -> None:
    requests = event.data.get("requests") or []
    await _resume(
        f"seal-session:{session_id}:{event.data.get('turn_index', 0)}",
        proto.ToolApprovals(
            tool_approvals=[
                proto.ToolApprovalResponse(
                    tool_call_id=request["tool_call_id"], granted=True
                )
                for request in requests
            ]
        ),
    )


async def _resume(token: str, payload: proto.ResumePayload) -> None:
    """Resolve a session hook, retrying while the driver registers it."""
    hook = proto.SessionHook(payload=payload)
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except RuntimeError as error:
            if attempt == 39 or "not found" not in str(error).lower():
                raise
            await asyncio.sleep(0.05)


async def _waiting_turn_index(session_id: str) -> int:
    """The turn the session is currently parked on (latest ``session.waiting``)."""
    turn_index = 0
    async for event in stream.replay(session_id):
        if (
            isinstance(event, proto.LifecycleEvent)
            and event.type == proto.SESSION_WAITING
        ):
            turn_index = int(event.data.get("turn_index", turn_index))
    return turn_index
