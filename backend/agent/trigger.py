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


async def _tail_tokens(stream_key: str) -> AsyncGenerator[str]:
    """Tail the jsonl side-channel for one full agent workflow run.

    Emits Server-Sent Events: ``data: <text chunk>`` per token, then a final
    ``event: done``. See ``main.start_agent_stream`` for run ownership.
    """
    async for record in durable_stream.get_readable(stream_key):
        if not isinstance(record, durable_stream.StreamEvent):
            continue
        if record.type != "TextDelta":
            continue
        chunk = record.data.get("chunk")
        if isinstance(chunk, str):
            yield f"data: {json.dumps(chunk)}\n\n"
    yield "event: done\ndata: {}\n\n"


@app.post("/run/stream")
async def run_stream(request: RunRequest) -> fastapi.responses.StreamingResponse:
    """Start a run and stream its live tokens back as SSE."""
    stream_key = uuid.uuid4().hex
    await vercel.workflow.start(main.run_agent, request.prompt, stream_key)
    return fastapi.responses.StreamingResponse(
        _tail_tokens(stream_key),
        media_type="text/event-stream",
        headers={"x-stream-key": stream_key},
    )


@app.get("/stream/{stream_key}")
async def stream(stream_key: str) -> fastapi.responses.StreamingResponse:
    """Tail live tokens for an already-started run by its stream_key."""
    return fastapi.responses.StreamingResponse(
        _tail_tokens(stream_key),
        media_type="text/event-stream",
    )
