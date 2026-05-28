"""Chat, file upload, and file serving endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any, NamedTuple, cast

import ai.agents.ui.ai_sdk.outbound_stream as ai_sdk_outbound_stream
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
    to_stream,
)
from vercel.blob import AsyncBlobClient
from vercel.workflow import Run, start

import attachments
import sessions
import stream_store
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


_MODEL_EVENT_ADAPTER: pydantic.TypeAdapter[Any] = pydantic.TypeAdapter(
    ai_events.DiscriminatedEvent
)


def _load_agent_event(data: dict[str, object]) -> ai_events.AgentEvent:
    kind = data.get("kind")
    if kind == "tool_call_result":
        return ai_events.ToolCallResult.model_validate(data)
    if kind == "hook":
        return ai_events.HookEvent.model_validate(data)
    if kind == "partial_tool_call_result":
        return ai_events.PartialToolCallResult.model_validate(data)
    return cast(ai_events.AgentEvent, _MODEL_EVENT_ADAPTER.validate_python(data))


async def _to_supported_sse(
    events: AsyncIterable[ai_events.AgentEvent],
) -> AsyncGenerator[str]:
    """Convert events to SSE chunks supported by the installed JS AI SDK."""
    denied_tool_call_ids: set[str] = set()

    async for event in to_stream(events):
        event_type = getattr(event, "type", None)
        tool_call_id = getattr(event, "tool_call_id", None)

        if event_type == "tool-approval-response":
            continue

        if isinstance(tool_call_id, str) and event_type == "tool-output-denied":
            denied_tool_call_ids.add(tool_call_id)

        if (
            isinstance(tool_call_id, str)
            and event_type == "tool-output-error"
            and tool_call_id in denied_tool_call_ids
        ):
            continue

        yield ai_sdk_outbound_stream.format_sse(event)

    yield ai_sdk_outbound_stream.format_done_sse()


async def _watch_stream_events(
    *,
    session_id: str,
    run_id: str,
    stream_id: str,
    start_index: int,
) -> AsyncGenerator[ai_events.AgentEvent]:
    last_index = start_index
    run = Run(run_id)
    terminal = {"completed", "failed", "cancelled"}

    while True:
        events = await stream_store.list_events(stream_id, last_index)
        if events:
            for event in events:
                last_index = event.index + 1
                yield _load_agent_event(event.data)
            continue

        status = await stream_store.get_status(stream_id)
        if status in ("waiting", "completed", "failed"):
            return

        run_status = await run.status()
        if run_status in terminal:
            if run_status == "failed":
                await stream_store.set_status(stream_id, "failed")
            current = await sessions.get_session(session_id)
            if current is not None and current.active_run_id == run_id:
                await sessions.set_active_run(session_id, None)
            return

        await asyncio.sleep(0.5)


@router.get("/chat/{session_id}/stream")
async def resume_chat_stream(
    session_id: str,
    request: fastapi.Request,
) -> fastapi.responses.Response:
    """Resume an active durable chat stream."""
    session = await sessions.get_session(session_id)
    if session is None or session.active_run_id is None:
        return fastapi.responses.Response(status_code=204)

    run_id = session.active_run_id
    stream_id = session_id
    status = await stream_store.get_status(stream_id)
    if status not in ("running", "waiting"):
        return fastapi.responses.Response(status_code=204)

    raw_start_index = request.query_params.get("startIndex")
    if raw_start_index is None:
        raw_start_index = request.query_params.get("start_index")

    if raw_start_index is None:
        start_index = await stream_store.get_active_start_index(stream_id)
    else:
        try:
            start_index = int(raw_start_index)
        except ValueError:
            raise fastapi.HTTPException(
                status_code=400,
                detail="startIndex must be a non-negative integer",
            ) from None

    if start_index < 0:
        raise fastapi.HTTPException(
            status_code=400,
            detail="startIndex must be a non-negative integer",
        )

    async def stream_response() -> AsyncGenerator[str]:
        async for chunk in _to_supported_sse(
            _watch_stream_events(
                session_id=session_id,
                run_id=run_id,
                stream_id=stream_id,
                start_index=start_index,
            )
        ):
            yield chunk

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=UI_MESSAGE_STREAM_HEADERS,
    )


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
    stream_id = session_id
    start_index = await stream_store.count_events(stream_id)

    if prepared.changed:
        if active_run_id is not None:
            raise fastapi.HTTPException(
                status_code=409,
                detail="Session already has an active agent run",
            )
        await sessions.save_messages(session_id, prepared.messages)
        await stream_store.set_active_start_index(stream_id, start_index)
        await stream_store.set_status(stream_id, "running")
        run = await start(
            workflows.run_agent,
            session_id,
            stream_id,
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
        await stream_store.save_tool_approval(
            stream_id,
            approval_id=approval.hook_id,
            tool_call_id=approval.tool_call_id,
            granted=approval.granted,
            reason=approval.reason,
        )
        await stream_store.set_status(stream_id, "running")

    if run_id is None:
        raise fastapi.HTTPException(
            status_code=400,
            detail="No workflow run to stream",
        )

    async def stream_response() -> AsyncGenerator[str]:
        async for chunk in _to_supported_sse(
            _watch_stream_events(
                session_id=session_id,
                run_id=run_id,
                stream_id=stream_id,
                start_index=start_index,
            )
        ):
            yield chunk

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=UI_MESSAGE_STREAM_HEADERS,
    )
