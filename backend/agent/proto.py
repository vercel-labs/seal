from __future__ import annotations

from typing import Any, Literal

import ai
import pydantic
import vercel.workflow

# Session inputs / outputs


# external decision for a single gated tool call.
class ToolApprovalResponse(pydantic.BaseModel):
    tool_call_id: str
    granted: bool
    reason: str | None = None


# ai SDK gates a tool behind a hook labelled ``approve_{tool_call_id}``.
TOOL_APPROVAL_HOOK_PREFIX = "approve_"


# the durable hook a gated call parks on; its trailing segment is exactly the ai
# ``HookPart.hook_id``. tokens are global, so the session id keeps them unique.
def approval_hook_token(session_id: str, tool_call_id: str) -> str:
    return f"seal-approval:{session_id}:{TOOL_APPROVAL_HOOK_PREFIX}{tool_call_id}"


class SessionInput(pydantic.BaseModel):
    session_id: str
    prompt: str


class SessionOutput(pydantic.BaseModel):
    session_id: str
    output: str
    is_error: bool = False


class NewUserMessage(pydantic.BaseModel):
    kind: Literal["new_user_message"] = "new_user_message"
    prompt: str | None = None
    close: bool = False


# carries the next user message (or a close) to a session parked in ``suspend``.
class SessionHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    payload: NewUserMessage


# one gated call's decision, delivered on its own per-approval hook.
class ApprovalHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    response: ToolApprovalResponse


class SessionState(pydantic.BaseModel):
    session_id: str
    messages: list[ai.messages.Message]


# Turn inputs / outputs


class TurnInput(pydantic.BaseModel):
    session_id: str
    messages: list[ai.messages.Message]
    # gated turns expose bash behind approval + subagent; ungated (subagent
    # children) run bash directly and cannot delegate further.
    gated: bool = True
    turn_hook_token: str
    # index of this turn within its session (always 0 for subagent turns).
    turn_index: int = 0
    # turn's root span. llm_steps and child turns nest under it.
    turn_span: ai.experimental_telemetry.Span | None = None


# in-process context of the running tool call, set by the agent loop around
# each schedule so a tool can reach it without smuggling args. never journaled.
class ToolCallContext(pydantic.BaseModel):
    session_id: str
    tool_call_id: str
    # the enclosing turn's root span; a spawned child turn nests under it.
    turn_span: ai.experimental_telemetry.Span | None = None


class TurnOutput(pydantic.BaseModel):
    kind: Literal["suspend", "error"]
    messages: list[ai.messages.Message]
    error: str | None = None


class TurnHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    output: TurnOutput


# Durable stream

SESSION_NAMESPACE = "session"

SESSION_STARTED = "session.started"
SESSION_WAITING = "session.waiting"
SESSION_COMPLETED = "session.completed"
SESSION_FAILED = "session.failed"
TURN_STARTED = "turn.started"
TURN_COMPLETED = "turn.completed"
SUBAGENT_CALLED = "subagent.called"
SUBAGENT_COMPLETED = "subagent.completed"
TOOL_APPROVAL_REQUESTED = "tool_approval.requested"
DEFAULT_STREAM_NAMESPACE = "default"
DEFAULT_STREAM_POLL_INTERVAL = 0.05
WRITABLE_STREAM_HANDLE_TYPE = "seal.durable_agent.writable_stream"


class LifecycleEvent(pydantic.BaseModel):
    kind: Literal["lifecycle"] = "lifecycle"
    type: str
    data: dict[str, Any] = pydantic.Field(default_factory=dict)
    # ISO 8601 UTC string. None when constructed inside a workflow body
    # (datetime is sandbox-restricted); stamped by the write function.
    at: str | None = None


type StreamEvent = ai.events.AgentEvent | LifecycleEvent

STREAM_EVENT_ADAPTER: pydantic.TypeAdapter[StreamEvent] = pydantic.TypeAdapter(
    StreamEvent
)
