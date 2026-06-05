from __future__ import annotations

from typing import Any, Literal

import ai
import pydantic
import vercel.workflow

# Session inputs / outputs

type SessionMode = Literal["infinite", "task"]


class SubagentRequest(pydantic.BaseModel):
    tool_call_id: str
    name: str = "subagent"
    prompt: str


class ToolApprovalRequest(pydantic.BaseModel):
    tool_call_id: str
    tool_name: str = ""
    args: dict[str, Any] = pydantic.Field(default_factory=dict)


# external decision for a single gated tool call.
class ToolApprovalResponse(pydantic.BaseModel):
    tool_call_id: str
    granted: bool
    reason: str | None = None


# ai SDK gates a tool behind a hook labelled ``approve_{tool_call_id}``.
TOOL_APPROVAL_HOOK_PREFIX = "approve_"


class SessionInput(pydantic.BaseModel):
    session_id: str
    prompt: str
    mode: SessionMode = "infinite"

    # subagent-specific inputs
    session_hook_token: str | None = None  # hook id for parent's suspension hook
    tool_call_id: str = ""  # return along with outputs to identify itself


class SessionOutput(pydantic.BaseModel):
    tool_call_id: str = ""  # subagent-specific
    session_id: str
    output: str
    is_error: bool = False


class SubagentResult(pydantic.BaseModel):
    kind: Literal["subagent_result"] = "subagent_result"
    output: SessionOutput


class ToolApprovals(pydantic.BaseModel):
    kind: Literal["tool_approvals"] = "tool_approvals"
    tool_approvals: list[ToolApprovalResponse] = pydantic.Field(default_factory=list)


class NewUserMessage(pydantic.BaseModel):
    kind: Literal["new_user_message"] = "new_user_message"
    prompt: str | None = None
    close: bool = False


type ResumePayload = SubagentResult | ToolApprovals | NewUserMessage

RESUME_PAYLOAD_ADAPTER: pydantic.TypeAdapter[ResumePayload] = pydantic.TypeAdapter(
    ResumePayload
)


class SessionHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    payload: ResumePayload = pydantic.Field(discriminator="kind")


# in-flight out-of-loop work for the session. session uses this to keep track
# of which scheduled jobs have been completed and accumulate results.
class PendingState(pydantic.BaseModel):
    turn_index: int
    subagents: list[SubagentRequest] = pydantic.Field(default_factory=list)
    tool_approval_requests: list[ToolApprovalRequest] = pydantic.Field(
        default_factory=list
    )
    # side effects (child spawns + lifecycle events) fired exactly once.
    dispatched: bool = False
    # tool_call_id -> finished subagent output.
    subagent_outputs: dict[str, SessionOutput] = pydantic.Field(default_factory=dict)
    # human decision once resolved; None until then.
    tool_approvals: list[ToolApprovalResponse] | None = None


class SessionState(pydantic.BaseModel):
    session_id: str
    mode: SessionMode
    messages: list[ai.messages.Message]
    # decisions to pre-register on the next turn replay.
    tool_approvals: list[ToolApprovalResponse] = pydantic.Field(default_factory=list)
    # in-flight pending-request collection for the current turn, if any.
    pending: PendingState | None = None


# Turn inputs / outputs


class TurnInput(pydantic.BaseModel):
    session_id: str
    messages: list[ai.messages.Message]
    mode: SessionMode = "infinite"
    turn_hook_token: str
    # decisions to pre-register before the interrupted turn replays.
    tool_approvals: list[ToolApprovalResponse] = pydantic.Field(default_factory=list)


class TurnOutput(pydantic.BaseModel):
    kind: Literal["done", "suspend", "pending_requests", "error"]
    messages: list[ai.messages.Message]
    # subagents to dispatch and/or gated tool calls awaiting a human decision.
    pending_requests: list[SubagentRequest | ToolApprovalRequest] = pydantic.Field(
        default_factory=list
    )
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
TOOL_APPROVAL_RESOLVED = "tool_approval.resolved"
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
