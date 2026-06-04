from __future__ import annotations

import datetime
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
APPROVAL_HOOK_PREFIX = "approve_"


class SubagentHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    output: SessionOutput


class SessionInput(pydantic.BaseModel):
    session_id: str
    prompt: str
    mode: SessionMode = "infinite"
    subagent_hook_token: str | None = None
    # for subagent sessions to resolve hook in parent session


class SessionOutput(pydantic.BaseModel):
    session_id: str
    output: str
    is_error: bool = False


# external -> session ingress, shared by user messages and approval decisions.
class SessionResumeHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    prompt: str | None = None
    close: bool = False
    approvals: list[ToolApprovalResponse] = pydantic.Field(default_factory=list)


class SessionState(pydantic.BaseModel):
    session_id: str
    mode: SessionMode
    messages: list[ai.messages.Message]
    # decisions to pre-register on the next turn replay.
    approvals: list[ToolApprovalResponse] = pydantic.Field(default_factory=list)


# Turn inputs / outputs


class TurnInput(pydantic.BaseModel):
    session_id: str
    messages: list[ai.messages.Message]
    mode: SessionMode = "infinite"
    turn_hook_token: str
    # decisions to pre-register before the interrupted turn replays.
    approvals: list[ToolApprovalResponse] = pydantic.Field(default_factory=list)


class TurnOutput(pydantic.BaseModel):
    kind: Literal["done", "suspend", "pending_requests"]
    messages: list[ai.messages.Message]
    # subagents to dispatch and/or gated tool calls awaiting a human decision.
    pending_requests: list[SubagentRequest | ToolApprovalRequest] = pydantic.Field(
        default_factory=list
    )


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
    # has to be None when the event is constructed inside a workflow body
    # will get stamped by the write function
    at: datetime.datetime | None = None


type StreamEvent = ai.events.AgentEvent | LifecycleEvent

STREAM_EVENT_ADAPTER: pydantic.TypeAdapter[StreamEvent] = pydantic.TypeAdapter(
    StreamEvent
)
