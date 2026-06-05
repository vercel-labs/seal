"""Minimal FastAPI trigger for the durable agent.

Three endpoints dispatch and observe one durable session:

  POST /session                  start a session, return its id
  GET  /session/{id}/events      tail the session's event stream as NDJSON
  POST /session/{id}/close       end a waiting session so its stream closes

`vercel dev` serves this app; the workflow worker (`_worker.py`) drives the
actual run. The preamble mirrors `_worker.py` because this process also calls
`vercel.workflow.start`, which imports and replays the workflow modules.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator

_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault(
    "WORKFLOW_LOCAL_DATA_DIR",
    os.path.join(_BACKEND_DIR, ".workflow-data"),
)
os.environ.setdefault(
    "SEAL_DURABLE_AGENT_STREAMS_DIR",
    os.path.join(_BACKEND_DIR, ".durable_agent_streams"),
)

import vercel._internal.workflow.py_sandbox  # noqa: E402

# See `_worker.py` for why these are served from the host inside the sandbox.
vercel._internal.workflow.py_sandbox._PASSTHROUGHS.update({"ai", "pathlib"})

import fastapi  # noqa: E402
import fastapi.responses  # noqa: E402
import pydantic  # noqa: E402
import vercel.workflow  # noqa: E402

from durable_agent import driver, proto, stream  # noqa: E402

app = fastapi.FastAPI(title="seal-durable-agent")


class StartRequest(pydantic.BaseModel):
    prompt: str


class StartResponse(pydantic.BaseModel):
    session_id: str
    run_id: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/session")
async def start_session(request: StartRequest) -> StartResponse:
    """Start an infinite-mode session and return immediately."""
    session_id = uuid.uuid4().hex
    session_input = proto.SessionInput(session_id=session_id, prompt=request.prompt)
    started = await vercel.workflow.start(
        driver.run_session,
        session_input.model_dump(mode="json"),
    )
    return StartResponse(session_id=session_id, run_id=started.run_id)


@app.get("/session/{session_id}/events")
async def session_events(
    session_id: str,
    start_index: int = 0,
) -> fastapi.responses.StreamingResponse:
    """Tail the session's durable stream as newline-delimited JSON."""

    async def body() -> AsyncIterator[str]:
        async for event in stream.get_readable(session_id, start_index=start_index):
            yield json.dumps(event.model_dump(mode="json")) + "\n"

    return fastapi.responses.StreamingResponse(
        body(),
        media_type="application/x-ndjson",
    )


class ApproveRequest(pydantic.BaseModel):
    tool_approvals: list[proto.ToolApprovalResponse]


async def _resume_session(token: str, payload: proto.ResumePayload) -> None:
    # the driver writes the request event just before parking on the hook, so a
    # fast client can resume before the hook is registered; retry while missing.
    hook = proto.SessionHook(payload=payload)
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except RuntimeError as error:
            if attempt == 39 or "not found" not in str(error).lower():
                raise
            await asyncio.sleep(0.05)


@app.post("/session/{session_id}/approve")
async def approve_session(
    session_id: str, request: ApproveRequest, turn_index: int = 0
) -> dict[str, bool]:
    """Resolve the waiting session hook with tool-approval decisions."""
    token = f"seal-session:{session_id}:{turn_index}"
    await _resume_session(
        token, proto.ToolApprovals(tool_approvals=request.tool_approvals)
    )
    return {"ok": True}


@app.post("/session/{session_id}/close")
async def close_session(session_id: str, turn_index: int = 0) -> dict[str, bool]:
    """Resolve the waiting session hook with close, so the run finishes."""
    token = f"seal-session:{session_id}:{turn_index}"
    await _resume_session(token, proto.NewUserMessage(close=True))
    return {"ok": True}
