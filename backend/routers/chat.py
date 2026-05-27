"""Chat, file upload, and file serving endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import NamedTuple

import ai
import fastapi
import fastapi.responses
import pydantic
from ai import events as ai_events
from ai import messages as ai_messages
from ai.agents.ui.ai_sdk import (
    UI_MESSAGE_STREAM_HEADERS,
    ApprovalResponse,
    UIMessage,
    to_messages,
    to_sse,
)
from vercel.blob import AsyncBlobClient
from vercel.workflow import Run, start

import attachments
import sessions
import workflows

router = fastapi.APIRouter()

# ---------------------------------------------------------------------------
# File upload & serving
# ---------------------------------------------------------------------------


class UploadResponse(pydantic.BaseModel):
    """Response from the file upload endpoint."""

    url: str
    media_type: str = pydantic.Field(serialization_alias="mediaType")
    filename: str


@router.post("/upload")
async def upload(file: fastapi.UploadFile) -> UploadResponse:
    """Upload a file to Vercel Blob storage (private)."""
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

    # Return a proxy URL so the browser fetches through our backend,
    # keeping the blob private.
    return UploadResponse(
        url=f"{attachments.FILES_PREFIX}{result.pathname}",
        media_type=media_type,
        filename=filename,
    )


@router.get("/files/{pathname:path}")
async def get_file(pathname: str) -> fastapi.responses.Response:
    """Proxy a private Vercel Blob file to the browser."""
    async with AsyncBlobClient() as client:
        result = await client.get(pathname, access="private")

    return fastapi.responses.Response(
        content=result.content,
        media_type=result.content_type or "application/octet-stream",
        headers={
            # Blob pathnames include a random suffix so each upload is unique.
            # Aggressive caching avoids re-fetching on every message re-render.
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


# ---------------------------------------------------------------------------
# Chat-request normalization
# ---------------------------------------------------------------------------


class PreparedChat(NamedTuple):
    """Canonical messages and side effects extracted from a useChat request."""

    messages: list[ai_messages.Message]
    approvals: list[ApprovalResponse]
    has_work: bool
    changed: bool


def prepare_chat_request(
    *,
    request_messages: list[UIMessage],
    stored_messages: list[ai_messages.Message],
) -> PreparedChat:
    """Use backend history plus the request's last new user message."""
    request_ai_messages, approvals = to_messages(request_messages)
    new_messages = sessions.get_new_messages(request_ai_messages, stored_messages)
    messages = [*stored_messages, *new_messages]
    return PreparedChat(
        messages=messages,
        approvals=approvals,
        has_work=bool(new_messages or approvals),
        changed=bool(new_messages),
    )


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


class ChatRequest(pydantic.BaseModel):
    """Request body for the chat endpoint."""

    messages: list[UIMessage]
    session_id: str


async def _events_for_messages(
    messages: list[ai_messages.Message],
) -> AsyncGenerator[ai_events.AgentEvent]:
    for message in messages:
        if message.role == "assistant":
            async for event in ai_events.replay_message_events(message):
                yield event
        elif message.role == "tool":
            yield ai.tool_result(message)
        elif message.role == "internal":
            for part in message.parts:
                if isinstance(part, ai_messages.HookPart):
                    yield ai_events.HookEvent(message=message, hook=part)


async def _watch_workflow_events(
    *,
    session_id: str,
    run_id: str,
    start_index: int,
) -> AsyncGenerator[ai_events.AgentEvent]:
    last_index = start_index
    run = Run(run_id)
    terminal = {"completed", "failed", "cancelled"}

    while True:
        session = await sessions.get_session(session_id)
        messages = session.messages if session is not None else []
        if len(messages) > last_index:
            new_messages = messages[last_index:]
            last_index = len(messages)
            async for event in _events_for_messages(new_messages):
                yield event
            continue

        if await run.status() in terminal:
            current = await sessions.get_session(session_id)
            if current is not None and current.active_run_id == run_id:
                await sessions.set_active_run(session_id, None)
            return

        await asyncio.sleep(0.5)


@router.post("/chat")
async def chat(request: ChatRequest) -> fastapi.responses.StreamingResponse:
    """Handle chat requests and stream responses."""
    session_id = request.session_id

    session = await sessions.get_session(session_id)
    stored_messages = session.messages if session is not None else []
    if session is None:
        await sessions.create_session(session_id)

    prepared = prepare_chat_request(
        request_messages=request.messages,
        stored_messages=stored_messages,
    )

    if not prepared.has_work:
        raise fastapi.HTTPException(
            status_code=400,
            detail="No new user message or approval response to process",
        )

    if prepared.approvals and prepared.changed:
        raise fastapi.HTTPException(
            status_code=409,
            detail="Cannot approve a tool and send a new message in the same request",
        )

    active_run_id = session.active_run_id if session is not None else None
    run_id: str | None = active_run_id
    start_index = len(stored_messages)

    if prepared.changed:
        if active_run_id is not None:
            raise fastapi.HTTPException(
                status_code=409,
                detail="Session already has an active agent run",
            )
        await sessions.save_messages(session_id, prepared.messages)
        start_index = len(prepared.messages)
        run = await start(
            workflows.run_agent,
            session_id,
            workflows.dump_messages(prepared.messages),
        )
        run_id = run.run_id
        await sessions.set_active_run(session_id, run_id)

    for approval in prepared.approvals:
        if run_id is None:
            raise fastapi.HTTPException(
                status_code=400,
                detail="No active workflow run for approval response",
            )
        await workflows.resume_tool_approval(
            approval.hook_id,
            granted=approval.granted,
            reason=approval.reason,
        )

    if run_id is None:
        raise fastapi.HTTPException(
            status_code=400,
            detail="No workflow run to stream",
        )

    async def stream_response() -> AsyncGenerator[str]:
        async for chunk in to_sse(
            _watch_workflow_events(
                session_id=session_id,
                run_id=run_id,
                start_index=start_index,
            )
        ):
            yield chunk

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=UI_MESSAGE_STREAM_HEADERS,
    )
