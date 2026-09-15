from __future__ import annotations

from typing import Any, Literal

import ai
import pydantic


class ToolApprovalResponse(pydantic.BaseModel):
    hook_id: str
    tool_call_id: str
    granted: bool
    reason: str | None = None


class SessionInput(pydantic.BaseModel):
    session_id: str
    prompt: str


class NewUserMessage(pydantic.BaseModel):
    kind: Literal["new_user_message"] = "new_user_message"
    prompt: str


class ApprovalSignal(pydantic.BaseModel):
    responses: list[ToolApprovalResponse]


class EagerToolSignal(pydantic.BaseModel):
    tool_call: ai.messages.ToolCallPart


class SessionState(pydantic.BaseModel):
    session_id: str
    messages: list[ai.messages.Message]


class TurnInput(pydantic.BaseModel):
    session_id: str
    messages: list[ai.messages.Message]
    gated: bool = True
    turn_index: int = 0


class TurnOutput(pydantic.BaseModel):
    kind: Literal["suspend", "error", "interrupted"]
    messages: list[ai.messages.Message]
    error: str | None = None


SESSION_STARTED = "session.started"
SESSION_WAITING = "session.waiting"
SESSION_COMPLETED = "session.completed"
SESSION_FAILED = "session.failed"
SESSION_INTERRUPTED = "session.interrupted"
TURN_STARTED = "turn.started"
SUBAGENT_CALLED = "subagent.called"
SUBAGENT_COMPLETED = "subagent.completed"
RELOAD_REQUESTED = "reload.requested"


class LifecycleEvent(pydantic.BaseModel):
    kind: Literal["lifecycle"] = "lifecycle"
    type: str
    data: dict[str, Any] = pydantic.Field(default_factory=dict)


type StreamEvent = ai.events.OmitEventMessages[ai.events.AgentEvent] | LifecycleEvent
