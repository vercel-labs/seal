"""Chat, file upload, and file serving endpoints."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any, NamedTuple

import fastapi
import fastapi.responses
import pydantic
from ai import messages as ai_messages
from ai.agents.ui.ai_sdk import (
    UI_MESSAGE_STREAM_HEADERS,
    ApprovalResponse,
    UIMessage,
    to_messages,
)
from vercel.blob import AsyncBlobClient
from vercel.workflow import Run, start

from core import attachments, durable_agent, resumable_stream, sessions, stream_store

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


async def _watch_stream_event_records(
    *,
    session_id: str,
    run_id: str,
    stream_id: str,
    start_index: int,
) -> AsyncGenerator[stream_store.StreamEvent]:
    last_index = start_index
    run = Run(run_id)
    terminal = {"completed", "failed", "cancelled"}

    while True:
        events = await stream_store.list_events(stream_id, last_index)
        if events:
            for event in events:
                last_index = event.index + 1
                yield event
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


def _new_ui_stream_id(session_id: str) -> str:
    return f"{session_id}:ui:{uuid.uuid4().hex[:12]}"


async def _create_active_ui_stream(
    *,
    session_id: str,
    source_stream_id: str,
    source_start_index: int,
    history_message_count: int,
) -> str:
    ui_stream_id = _new_ui_stream_id(session_id)
    await stream_store.create_ui_stream(
        ui_stream_id,
        session_id=session_id,
        source_stream_id=source_stream_id,
        source_start_index=source_start_index,
        history_message_count=history_message_count,
    )
    await sessions.set_active_stream(session_id, ui_stream_id)
    return ui_stream_id


async def _warm_ui_stream_state(
    ui_stream: stream_store.UIStream,
) -> tuple[Any, set[str]]:
    state = durable_agent.new_ui_stream_state()
    denied_tool_call_ids: set[str] = set()
    events = await stream_store.list_events(
        ui_stream.source_stream_id,
        ui_stream.source_start_index,
    )
    for event in events:
        if event.index >= ui_stream.source_next_index:
            break
        durable_agent.ui_sse_chunks_for_agent_event(
            state,
            event.data,
            denied_tool_call_ids,
        )
    return state, denied_tool_call_ids


async def _ui_event_chunk_batch(
    *,
    ui_stream: stream_store.UIStream,
    source_event: stream_store.StreamEvent,
    state: Any,
    denied_tool_call_ids: set[str],
) -> resumable_stream.ChunkBatch:
    new_chunks = durable_agent.ui_sse_chunks_for_agent_event(
        state,
        source_event.data,
        denied_tool_call_ids,
    )

    async def commit() -> None:
        await stream_store.complete_ui_stream_source_index(
            ui_stream.id,
            source_event.index,
            source_event.index + 1,
        )

    async def rollback() -> None:
        await stream_store.release_ui_stream_source_index(
            ui_stream.id,
            source_event.index,
        )

    return resumable_stream.ChunkBatch(
        chunks=new_chunks,
        commit=commit,
        rollback=rollback,
    )


def _ui_finish_chunk_batch(
    *,
    session_id: str,
    ui_stream: stream_store.UIStream,
    source_index: int,
    state: Any,
    denied_tool_call_ids: set[str],
) -> resumable_stream.ChunkBatch:
    finish_chunks = durable_agent.finish_ui_sse_chunks(
        state,
        denied_tool_call_ids,
    )
    finish_chunks.append(durable_agent.done_sse())

    async def commit() -> None:
        await stream_store.complete_ui_stream_source_index(
            ui_stream.id,
            source_index,
            source_index,
        )
        await stream_store.set_status(ui_stream.id, "completed")
        await sessions.clear_active_stream(session_id, ui_stream.id)

    async def rollback() -> None:
        await stream_store.release_ui_stream_source_index(ui_stream.id, source_index)

    return resumable_stream.ChunkBatch(
        chunks=finish_chunks,
        commit=commit,
        rollback=rollback,
    )


async def _ui_chunk_batches(
    *,
    session_id: str,
    run_id: str,
    ui_stream: stream_store.UIStream,
) -> AsyncGenerator[str | resumable_stream.ChunkBatch]:
    state, denied_tool_call_ids = await _warm_ui_stream_state(ui_stream)
    source_next_index = ui_stream.source_next_index

    async for event in _watch_stream_event_records(
        session_id=session_id,
        run_id=run_id,
        stream_id=ui_stream.source_stream_id,
        start_index=source_next_index,
    ):
        claimed = await stream_store.claim_ui_stream_source_index(
            ui_stream.id,
            event.index,
        )
        if not claimed:
            return

        yield await _ui_event_chunk_batch(
            ui_stream=ui_stream,
            source_event=event,
            state=state,
            denied_tool_call_ids=denied_tool_call_ids,
        )
        source_next_index = event.index + 1

    claimed = await stream_store.claim_ui_stream_source_index(
        ui_stream.id,
        source_next_index,
    )
    if not claimed:
        return

    yield _ui_finish_chunk_batch(
        session_id=session_id,
        ui_stream=ui_stream,
        source_index=source_next_index,
        state=state,
        denied_tool_call_ids=denied_tool_call_ids,
    )


async def _stream_ui_response(
    *,
    session_id: str,
    run_id: str,
    ui_stream: stream_store.UIStream,
    start_index: int,
) -> AsyncGenerator[str]:
    async def producer() -> AsyncGenerator[str | resumable_stream.ChunkBatch]:
        async for batch in _ui_chunk_batches(
            session_id=session_id,
            run_id=run_id,
            ui_stream=ui_stream,
        ):
            yield batch

    async for chunk in resumable_stream.replay_or_produce(
        stream_id=ui_stream.id,
        start_index=start_index,
        list_chunks=stream_store.list_ui_chunks,
        append_chunk=stream_store.append_ui_chunk,
        get_status=stream_store.get_status,
        producer=producer,
    ):
        yield chunk


@router.get("/chat/{session_id}/stream")
async def resume_chat_stream(
    session_id: str,
    request: fastapi.Request,
) -> fastapi.responses.Response:
    """Resume an active durable chat stream."""
    session = await sessions.get_session(session_id)
    if (
        session is None
        or session.active_run_id is None
        or session.active_stream_id is None
    ):
        return fastapi.responses.Response(status_code=204)

    run_id = session.active_run_id
    ui_stream = await stream_store.get_ui_stream(session.active_stream_id)
    if ui_stream is None or ui_stream.status != "running":
        await sessions.clear_active_stream(session_id, session.active_stream_id)
        return fastapi.responses.Response(status_code=204)

    raw_start_index = request.query_params.get("startIndex")
    if raw_start_index is None:
        raw_start_index = request.query_params.get("start_index")

    if raw_start_index is None:
        requested_start_index = 0
    else:
        try:
            requested_start_index = resumable_stream.parse_start_index(raw_start_index)
        except ValueError as exc:
            raise fastapi.HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    tail_index = await stream_store.count_ui_chunks(ui_stream.id) - 1
    start_index = resumable_stream.resolve_start_index(
        requested_start_index,
        tail_index,
    )

    async def stream_response() -> AsyncGenerator[str]:
        async for chunk in _stream_ui_response(
            session_id=session_id,
            run_id=run_id,
            ui_stream=ui_stream,
            start_index=start_index,
        ):
            yield chunk

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers={
            **UI_MESSAGE_STREAM_HEADERS,
            "x-workflow-stream-tail-index": str(tail_index),
        },
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
    ui_stream_id: str | None = None

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
            durable_agent.run_agent,
            session_id,
            stream_id,
            durable_agent.dump_messages(prepared.messages),
        )
        run_id = run.run_id
        await sessions.set_active_run(session_id, run_id)
        ui_stream_id = await _create_active_ui_stream(
            session_id=session_id,
            source_stream_id=stream_id,
            source_start_index=start_index,
            history_message_count=len(prepared.messages),
        )

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

    if prepared.approvals and ui_stream_id is None:
        ui_stream_id = await _create_active_ui_stream(
            session_id=session_id,
            source_stream_id=stream_id,
            source_start_index=start_index,
            history_message_count=len(stored_messages),
        )

    if run_id is None:
        raise fastapi.HTTPException(
            status_code=400,
            detail="No workflow run to stream",
        )

    if ui_stream_id is None:
        raise fastapi.HTTPException(
            status_code=400,
            detail="No UI stream to stream",
        )

    ui_stream = await stream_store.get_ui_stream(ui_stream_id)
    if ui_stream is None:
        raise fastapi.HTTPException(
            status_code=500,
            detail="UI stream was not created",
        )

    async def stream_response() -> AsyncGenerator[str]:
        async for chunk in _stream_ui_response(
            session_id=session_id,
            run_id=run_id,
            ui_stream=ui_stream,
            start_index=0,
        ):
            yield chunk

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=UI_MESSAGE_STREAM_HEADERS,
    )
