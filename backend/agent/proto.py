from __future__ import annotations

from typing import Any, Literal

import ai
import pydantic
import vercel.workflow

# Session inputs / outputs


# external decision for a single gated tool call.
class ToolApprovalResponse(pydantic.BaseModel):
    hook_id: str
    tool_call_id: str
    granted: bool
    reason: str | None = None


# One durable hook carries all approval decisions for a turn. Tokens are global,
# so the session id keeps each session's hook unique.
def hooks_hook_token(session_id: str) -> str:
    return f"{session_id}:hooks"


def turn_hook_token(session_id: str) -> str:
    return f"seal-turn:{session_id}"


def session_hook_token(session_id: str) -> str:
    return f"seal-session:{session_id}"


class SessionInput(pydantic.BaseModel):
    session_id: str
    prompt: str


class NewUserMessage(pydantic.BaseModel):
    kind: Literal["new_user_message"] = "new_user_message"
    prompt: str


# carries the next user message to a parked session.
class SessionHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    payload: NewUserMessage


# one or more gated call decisions, delivered through the session's shared hook.
class ApprovalHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    responses: list[ToolApprovalResponse]


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
    # index of this turn within its session (always 0 for subagent turns).
    turn_index: int = 0
    # turn's root span. llm_steps and child turns nest under it.
    turn_span: ai.experimental_telemetry.Span | None = None


class TurnOutput(pydantic.BaseModel):
    kind: Literal["suspend", "error"]
    messages: list[ai.messages.Message]
    error: str | None = None


class TurnHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    output: TurnOutput


# Durable stream

SESSION_STARTED = "session.started"
SESSION_WAITING = "session.waiting"
SESSION_COMPLETED = "session.completed"
SESSION_FAILED = "session.failed"
TURN_STARTED = "turn.started"
SUBAGENT_CALLED = "subagent.called"
SUBAGENT_COMPLETED = "subagent.completed"
RELOAD_REQUESTED = "reload.requested"


class LifecycleEvent(pydantic.BaseModel):
    kind: Literal["lifecycle"] = "lifecycle"
    type: str
    data: dict[str, Any] = pydantic.Field(default_factory=dict)


type StreamEvent = ai.events.AgentEvent | LifecycleEvent
