"""FastAPI app for the seal durable agent — the UI-facing surface.

Endpoints (mounted under ``/api`` by ``vercel.json``):

  POST /chat                     run a turn, stream the AI SDK UI message stream
  GET  /chat/{id}/stream         resume an in-flight stream (not yet supported)
  GET  /sessions                 list sessions
  POST /sessions                 create a session
  GET  /sessions/{id}            session metadata + UI message history
  POST /sessions/{id}/title      generate a title from the first user message
  DELETE /sessions/{id}          delete a session
  POST /upload, GET /files/{p}   private blob upload + proxy

`vercel dev` serves this app; the workflow worker (`worker.py`) drives the run.
This process also calls ``vercel.workflow.start``, which imports and replays the
workflow modules, so the preamble mirrors ``worker.py``.
"""

from __future__ import annotations

import collections.abc
import os

_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault(
    "WORKFLOW_LOCAL_DATA_DIR",
    os.path.join(_BACKEND_DIR, ".workflow-data"),
)
os.environ.setdefault(
    "SEAL_STREAMS_DIR",
    os.path.join(_BACKEND_DIR, ".seal"),
)

import vercel._internal.workflow.py_sandbox  # noqa: E402

# See `worker.py` for why these are served from the host inside the sandbox.
vercel._internal.workflow.py_sandbox._PASSTHROUGHS.update({"ai", "pathlib"})

import contextlib  # noqa: E402

import ai.agents.ui.ai_sdk as ai_sdk  # noqa: E402
import fastapi  # noqa: E402
import fastapi.middleware.cors  # noqa: E402
import fastapi.responses  # noqa: E402
import pydantic  # noqa: E402
from vercel.blob import AsyncBlobClient  # noqa: E402

from app import attachments, chat, sessions  # noqa: E402


@contextlib.asynccontextmanager
async def lifespan(_app: fastapi.FastAPI) -> collections.abc.AsyncIterator[None]:
    await sessions.ensure_schema()
    yield


app = fastapi.FastAPI(title="seal-durable-agent", lifespan=lifespan)
app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# --- chat ---------------------------------------------------------------------


class ChatRequest(pydantic.BaseModel):
    session_id: str
    messages: list[ai_sdk.UIMessage]


@app.post("/chat")
async def post_chat(request: ChatRequest) -> fastapi.responses.StreamingResponse:
    messages, _approvals = ai_sdk.to_messages(request.messages)
    prompt = next(
        (m.text for m in reversed(messages) if m.role == "user" and m.text), None
    )
    if prompt is None:
        raise fastapi.HTTPException(status_code=400, detail="No user message to run")

    await sessions.touch(request.session_id)
    start_index = await chat.start_or_resume(request.session_id, prompt)

    return fastapi.responses.StreamingResponse(
        chat.to_sse(request.session_id, start_index),
        headers=ai_sdk.UI_MESSAGE_STREAM_HEADERS,
    )


@app.get("/chat/{session_id}/stream")
async def resume_chat(session_id: str) -> fastapi.responses.Response:
    # Stream resume after reconnect is not wired up yet.
    return fastapi.responses.Response(status_code=204)


# --- sessions -----------------------------------------------------------------


class CreateSessionRequest(pydantic.BaseModel):
    id: str
    title: str | None = None


@app.get("/sessions")
async def list_sessions() -> list[sessions.SessionMeta]:
    return await sessions.list_sessions()


@app.post("/sessions", status_code=201)
async def create_session(body: CreateSessionRequest) -> sessions.SessionMeta:
    return await sessions.create_session(body.id, title=body.title)


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, object]:
    meta = await sessions.get_session(session_id)
    if meta is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    ui_messages = ai_sdk.to_ui_messages(await sessions.history(session_id))
    return {
        **meta.model_dump(),
        "messages": [
            message.model_dump(mode="json", by_alias=True) for message in ui_messages
        ],
    }


@app.post("/sessions/{session_id}/title")
async def generate_title(session_id: str) -> sessions.SessionMeta:
    meta = await sessions.get_session(session_id)
    if meta is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    if meta.title:
        return meta

    first_text = await sessions.first_user_text(session_id)
    if not first_text:
        raise fastapi.HTTPException(
            status_code=400, detail="No user message to generate title from"
        )
    updated = await sessions.set_title(
        session_id, await sessions.generate_title(first_text)
    )
    if updated is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return updated


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    if not await sessions.delete_session(session_id):
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


# --- attachments --------------------------------------------------------------


class UploadResponse(pydantic.BaseModel):
    url: str
    media_type: str = pydantic.Field(serialization_alias="mediaType")
    filename: str


@app.post("/upload")
async def upload(file: fastapi.UploadFile) -> UploadResponse:
    content = await file.read()
    media_type = file.content_type or "application/octet-stream"
    filename = file.filename or "attachment"
    async with AsyncBlobClient() as client:
        result = await client.put(
            f"attachments/{filename}",
            content,
            access="private",
            content_type=media_type,
            add_random_suffix=True,
        )
    return UploadResponse(
        url=f"{attachments.FILES_PREFIX}{result.pathname}",
        media_type=media_type,
        filename=filename,
    )


@app.get("/files/{pathname:path}")
async def get_file(pathname: str) -> fastapi.responses.Response:
    async with AsyncBlobClient() as client:
        result = await client.get(pathname, access="private")
    return fastapi.responses.Response(
        content=result.content,
        media_type=result.content_type or "application/octet-stream",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
