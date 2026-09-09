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


def interrupt_hook_token(session_id: str) -> str:
    return f"{session_id}:interrupt"


def turn_hook_token(session_id: str) -> str:
    return f"seal-turn:{session_id}"


def session_hook_token(session_id: str) -> str:
    return f"seal-session:{session_id}"


def launch_subagent_hook_token(session_id: str) -> str:
    return f"seal-subagent-launch:{session_id}"


def subagent_finished_hook_token(session_id: str) -> str:
    return f"seal-subagent-finished:{session_id}"


def subagent_launch_stream(task_id: str) -> str:
    return f"seal-subagent-launch:{task_id}"


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


class InterruptHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    pass


class BackgroundTaskState(pydantic.BaseModel):
    task_id: str
    child_session_id: str
    child_run_id: str
    name: str
    status: Literal["running", "completed", "failed"] = "running"
    messages: list[ai.messages.Message] = pydantic.Field(default_factory=list)
    error: str | None = None


class SessionState(pydantic.BaseModel):
    session_id: str
    messages: list[ai.messages.Message]
    background_tasks: dict[str, BackgroundTaskState] = pydantic.Field(
        default_factory=dict
    )
    hidden_ui_message_ids: set[str] = pydantic.Field(default_factory=set)
    active_ui_run_message_index: int | None = None


# Turn inputs / outputs


class TurnInput(pydantic.BaseModel):
    session_id: str
    messages: list[ai.messages.Message]
    # gated turns expose bash behind approval + subagent; ungated (subagent
    # children) run bash directly and cannot delegate further.
    gated: bool = True
    # Background turns report completion to the owning root session.
    parent_session_id: str | None = None
    background_task_id: str | None = None
    # index of this turn within its session (always 0 for subagent turns).
    turn_index: int = 0
    # turn's root span. llm_steps and child turns nest under it.
    turn_span: ai.experimental_telemetry.Span | None = None


class TurnOutput(pydantic.BaseModel):
    kind: Literal["suspend", "error", "interrupted"]
    messages: list[ai.messages.Message]
    error: str | None = None


class TurnHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    output: TurnOutput


class LaunchSubagentHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    task_id: str
    prompt: str
    name: str
    parent_span: ai.experimental_telemetry.Span | None = None


class SubagentFinishedHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    task_id: str
    output: TurnOutput


# Durable stream

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


# By using OmitEventMessage, we strip all the message fields from
# events (keeping only ids). We reconstruct them in the one place
# we need them (subagents).
type StreamEvent = ai.events.OmitEventMessages[ai.events.AgentEvent] | LifecycleEvent
