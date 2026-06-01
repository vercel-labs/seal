"""FastAPI server for the minimal durable agent workflow."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncGenerator

_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault(
    "SEAL_DURABLE_STREAMS_DIR",
    os.path.join(_BACKEND_DIR, ".streams"),
)
os.environ.setdefault(
    "SEAL_DURABLE_SESSIONS_DIR",
    os.path.join(_BACKEND_DIR, ".sessions"),
)
os.environ.setdefault(
    "WORKFLOW_LOCAL_DATA_DIR",
    os.path.join(_BACKEND_DIR, ".workflow-data"),
)

import fastapi  # noqa: E402
import fastapi.responses  # noqa: E402
import pydantic  # noqa: E402
import vercel.workflow  # noqa: E402

from agent import durable_stream  # noqa: E402
from agent import main as agent_main  # noqa: E402

app = fastapi.FastAPI(title="seal-agent-server")


class RunRequest(pydantic.BaseModel):
    prompt: str


class RunResponse(pydantic.BaseModel):
    run_id: str
    stream_key: str


class StatusResponse(pydantic.BaseModel):
    status: str
    output: str | None = None


class SessionRequest(pydantic.BaseModel):
    prompt: str | None = None
    session_id: str | None = None


class SessionResponse(pydantic.BaseModel):
    session_id: str
    run_id: str
    stream_key: str
    continuation_token: str


class TurnRequest(pydantic.BaseModel):
    prompt: str | None = None
    continuation_token: str | None = None
    close: bool = False


class TurnResponse(pydantic.BaseModel):
    session_id: str
    continuation_token: str | None = None


class SubagentRequest(pydantic.BaseModel):
    prompt: str
    name: str = "subagent"
    turn_id: str | None = None
    tool_call_id: str | None = None


@app.post("/run")
async def run(request: RunRequest) -> RunResponse:
    """Start a durable agent run and return its handle."""
    # stream_key must be stable outside the workflow (survives step retries).
    stream_key = uuid.uuid4().hex
    started = await vercel.workflow.start(
        agent_main.run_agent,
        request.prompt,
        stream_key,
    )
    return RunResponse(run_id=started.run_id, stream_key=stream_key)


@app.get("/run/{run_id}")
async def status(run_id: str) -> StatusResponse:
    """Poll a run; returns its final output once completed."""
    handle = vercel.workflow.Run(run_id)
    state = await handle.status()
    output = await handle.return_value() if state == "completed" else None
    return StatusResponse(status=state, output=output)


async def _tail_events(stream_key: str, start_index: int = 0) -> AsyncGenerator[str]:
    """Tail the jsonl side-channel for one full agent workflow run.

    Emits every durable stream record as a named Server-Sent Event. See
    ``agent_main.start_agent_stream`` for run ownership.
    """
    async for record in durable_stream.get_readable(
        stream_key,
        start_index=start_index,
    ):
        payload = record.model_dump(mode="json")
        if record.index is not None:
            payload["index"] = record.index
        event_name = (
            record.kind
            if isinstance(record, durable_stream.AgentStreamEvent)
            else record.type
        )
        yield f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"


@app.post("/run/stream")
async def run_stream(request: RunRequest) -> fastapi.responses.StreamingResponse:
    """Start a run and stream its events back as SSE."""
    stream_key = uuid.uuid4().hex
    await vercel.workflow.start(agent_main.run_agent, request.prompt, stream_key)
    return fastapi.responses.StreamingResponse(
        _tail_events(stream_key),
        media_type="text/event-stream",
        headers={"x-stream-key": stream_key},
    )


@app.get("/stream/{stream_key}")
async def stream(stream_key: str) -> fastapi.responses.StreamingResponse:
    """Tail live events for an already-started run by its stream_key."""
    return fastapi.responses.StreamingResponse(
        _tail_events(stream_key),
        media_type="text/event-stream",
    )


@app.post("/session")
async def session(request: SessionRequest) -> SessionResponse:
    """Start one long-lived durable agent session."""
    session_id = request.session_id or uuid.uuid4().hex
    stream_key = session_id
    started = await vercel.workflow.start(
        agent_main.run_session,
        session_id,
        stream_key,
        request.prompt,
    )
    turn_index = 2 if request.prompt is not None else 1
    return SessionResponse(
        session_id=session_id,
        run_id=started.run_id,
        stream_key=stream_key,
        continuation_token=f"seal-session:{session_id}:{turn_index}",
    )


@app.post("/session/{session_id}/turn")
async def session_turn(session_id: str, request: TurnRequest) -> TurnResponse:
    """Resume a waiting session workflow with the next user turn."""
    if not request.close and request.prompt is None:
        raise fastapi.HTTPException(
            status_code=400,
            detail="prompt is required unless close is true",
        )

    continuation_token = request.continuation_token
    if continuation_token is None:
        records = await durable_stream.list_records(session_id)
        for record in reversed(records):
            if (
                isinstance(record, durable_stream.AgentStreamEvent)
                and record.kind == "session.waiting"
            ):
                token = record.data.get("continuation_token")
                if isinstance(token, str):
                    continuation_token = token
                    break
        if continuation_token is None:
            raise fastapi.HTTPException(
                status_code=409,
                detail="session has no waiting continuation token",
            )

    retry_delay = 0.05
    for attempt in range(40):
        try:
            await agent_main.SessionTurn(
                prompt=request.prompt,
                close=request.close,
            ).resume(continuation_token)
            break
        except RuntimeError as error:
            error_text = str(error)
            lowered = error_text.lower()
            missing_hook = (
                "hook" in lowered
                and "not found" in lowered
                and continuation_token in error_text
            )
            if not missing_hook:
                raise fastapi.HTTPException(
                    status_code=409,
                    detail=error_text,
                ) from error
            if attempt == 39:
                raise fastapi.HTTPException(
                    status_code=409,
                    detail=error_text,
                ) from error
            await asyncio.sleep(retry_delay)

    next_token: str | None = None
    if not request.close:
        prefix, separator, index = continuation_token.rpartition(":")
        if separator and prefix == f"seal-session:{session_id}":
            try:
                next_token = f"{prefix}:{int(index) + 1}"
            except ValueError:
                next_token = None

    return TurnResponse(session_id=session_id, continuation_token=next_token)


@app.post("/session/{session_id}/subagent")
async def session_subagent(
    session_id: str,
    request: SubagentRequest,
) -> SessionResponse:
    """Start a child session and link it from the parent event stream."""
    child_session_id = f"{session_id}-child-{uuid.uuid4().hex}"
    parent = {"session_id": session_id, "name": request.name}
    if request.turn_id is not None:
        parent["turn_id"] = request.turn_id
    if request.tool_call_id is not None:
        parent["tool_call_id"] = request.tool_call_id

    stream = durable_stream.get_writable(session_id)
    await stream.write(
        durable_stream.AgentStreamEvent(
            kind="subagent.called",
            session_id=session_id,
            turn_id=request.turn_id,
            data={
                "child_session_id": child_session_id,
                "child_stream_key": child_session_id,
                "name": request.name,
                "tool_call_id": request.tool_call_id,
            },
        )
    )
    started = await vercel.workflow.start(
        agent_main.run_session,
        child_session_id,
        child_session_id,
        request.prompt,
        "subagent",
        parent,
    )
    return SessionResponse(
        session_id=child_session_id,
        run_id=started.run_id,
        stream_key=child_session_id,
        continuation_token=f"seal-session:{child_session_id}:2",
    )


@app.get("/session/{session_id}/events")
async def session_events(
    session_id: str,
    start_index: int = 0,
) -> fastapi.responses.StreamingResponse:
    """Tail the authoritative indexed event stream for a session."""
    return fastapi.responses.StreamingResponse(
        _tail_events(session_id, start_index=start_index),
        media_type="text/event-stream",
    )


def main() -> None:
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "pytest"],
                cwd=_BACKEND_DIR,
            )
        )
    raise SystemExit("Run tests with: python -m agent.server --test")


if __name__ == "__main__":
    main()
