"""Minimal FastAPI app to trigger the agent workflow."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator

import fastapi
import fastapi.responses
import pydantic
import vercel.workflow

from agent import durable_stream, main

app = fastapi.FastAPI(title="seal-agent-trigger")


class RunRequest(pydantic.BaseModel):
    prompt: str


class RunResponse(pydantic.BaseModel):
    run_id: str
    stream_key: str


class StatusResponse(pydantic.BaseModel):
    status: str
    output: str | None = None


@app.post("/run")
async def run(request: RunRequest) -> RunResponse:
    """Start a durable agent run and return its handle."""
    # stream_key must be stable outside the workflow (survives step retries).
    stream_key = uuid.uuid4().hex
    started = await vercel.workflow.start(main.run_agent, request.prompt, stream_key)
    return RunResponse(run_id=started.run_id, stream_key=stream_key)


@app.get("/run/{run_id}")
async def status(run_id: str) -> StatusResponse:
    """Poll a run; returns its final output once completed."""
    handle = vercel.workflow.Run(run_id)
    state = await handle.status()
    output = await handle.return_value() if state == "completed" else None
    return StatusResponse(status=state, output=output)


async def _tail_events(stream_key: str) -> AsyncGenerator[str]:
    """Tail the jsonl side-channel for one full agent workflow run.

    Emits every durable stream record as a named Server-Sent Event. See
    ``main.start_agent_stream`` for run ownership.
    """
    async for record in durable_stream.get_readable(stream_key):
        payload = record.model_dump(mode="json")
        if record.index is not None:
            payload["index"] = record.index
        yield f"event: {record.type}\ndata: {json.dumps(payload)}\n\n"


@app.post("/run/stream")
async def run_stream(request: RunRequest) -> fastapi.responses.StreamingResponse:
    """Start a run and stream its events back as SSE."""
    stream_key = uuid.uuid4().hex
    await vercel.workflow.start(main.run_agent, request.prompt, stream_key)
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
