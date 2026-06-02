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


class SessionResumeHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    prompt: str | None = None
    close: bool = False


class SessionState(pydantic.BaseModel):
    session_id: str
    mode: SessionMode
    messages: list[ai.messages.Message]


# Turn inputs / outputs


class TurnInput(pydantic.BaseModel):
    session_id: str
    messages: list[ai.messages.Message]
    mode: SessionMode = "infinite"
    turn_hook_token: str
    # for the turn to resolve the hook in the driver session


class TurnOutput(pydantic.BaseModel):
    kind: Literal["done", "suspend", "subagents"]
    messages: list[ai.messages.Message]
    subagent_requests: list[SubagentRequest] = pydantic.Field(default_factory=list)


class TurnHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    output: TurnOutput


# Durable stream

SESSION_NAMESPACE = "session"

SESSION_STARTED = "session.started"
SESSION_WAITING = "session.waiting"
SESSION_COMPLETED = "session.completed"
SESSION_FAILED = "session.failed"
SUBAGENT_CALLED = "subagent.called"
DEFAULT_STREAM_NAMESPACE = "default"
DEFAULT_STREAM_POLL_INTERVAL = 0.05
WRITABLE_STREAM_HANDLE_TYPE = "seal.durable_agent.writable_stream"


class ControlEvent(pydantic.BaseModel):
    kind: Literal["control"] = "control"
    type: str
    data: dict[str, Any] = pydantic.Field(default_factory=dict)


type StreamEvent = ai.events.AgentEvent | ControlEvent

STREAM_EVENT_ADAPTER: pydantic.TypeAdapter[StreamEvent] = pydantic.TypeAdapter(
    StreamEvent
)
