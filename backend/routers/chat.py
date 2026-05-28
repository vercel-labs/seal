"""Chat, file upload, and file serving endpoints."""

from __future__ import annotations

import asyncio
import uuid
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
_ui_stream_locks: dict[str, asyncio.Lock] = {}


def _load_agent_event(data: dict[str, object]) -> ai_events.AgentEvent:
    kind = data.get("kind")
    if kind == "tool_call_result":
        return ai_events.ToolCallResult.model_validate(data)
    if kind == "hook":
        return ai_events.HookEvent.model_validate(data)
    if kind == "partial_tool_call_result":
        return ai_events.PartialToolCallResult.model_validate(data)
    return cast(ai_events.AgentEvent, _MODEL_EVENT_ADAPTER.validate_python(data))


def _ui_stream_lock(stream_id: str) -> asyncio.Lock:
    lock = _ui_stream_locks.get(stream_id)
    if lock is None:
        lock = asyncio.Lock()
        _ui_stream_locks[stream_id] = lock
    return lock


async def _to_supported_sse(
    events: AsyncIterable[ai_events.AgentEvent],
) -> AsyncGenerator[str]:
    """Convert events to SSE chunks supported by the installed JS AI SDK."""
    state = ai_sdk_outbound_stream._StreamState()
    denied_tool_call_ids: set[str] = set()

    async for event in events:
        for ui_event in _ui_events_for_agent_event(state, event):
            chunk = _format_supported_sse_event(ui_event, denied_tool_call_ids)
            if chunk is not None:
                yield chunk

    for ui_event in state.finish():
        chunk = _format_supported_sse_event(ui_event, denied_tool_call_ids)
        if chunk is not None:
            yield chunk

    yield ai_sdk_outbound_stream.format_done_sse()


def _ui_events_for_agent_event(
    state: Any,
    event: ai_events.AgentEvent,
) -> list[Any]:
    if isinstance(event, ai_events.ToolCallResult):
        return cast(list[Any], state.on_tool_result(event))
    if isinstance(event, ai_events.PartialToolCallResult):
        return cast(list[Any], state.on_partial_tool_result(event))
    if isinstance(event, ai_events.HookEvent):
        return cast(list[Any], state.on_hook(event))
    return cast(list[Any], state.on_event(event))


def _format_supported_sse_event(
    event: Any,
    denied_tool_call_ids: set[str],
) -> str | None:
    event_type = getattr(event, "type", None)
    tool_call_id = getattr(event, "tool_call_id", None)

    if event_type == "tool-approval-response":
        return None

    if isinstance(tool_call_id, str) and event_type == "tool-output-denied":
        denied_tool_call_ids.add(tool_call_id)

    if (
        isinstance(tool_call_id, str)
        and event_type == "tool-output-error"
        and tool_call_id in denied_tool_call_ids
    ):
        return None

    return ai_sdk_outbound_stream.format_sse(event)


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


async def _watch_stream_events(
    *,
    session_id: str,
    run_id: str,
    stream_id: str,
    start_index: int,
) -> AsyncGenerator[ai_events.AgentEvent]:
    async for event in _watch_stream_event_records(
        session_id=session_id,
        run_id=run_id,
        stream_id=stream_id,
        start_index=start_index,
    ):
        yield _load_agent_event(event.data)


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
    state = ai_sdk_outbound_stream._StreamState()
    denied_tool_call_ids: set[str] = set()
    events = await stream_store.list_events(
        ui_stream.source_stream_id,
        ui_stream.source_start_index,
    )
    for event in events:
        if event.index >= ui_stream.source_next_index:
            break
        agent_event = _load_agent_event(event.data)
        for ui_event in _ui_events_for_agent_event(state, agent_event):
            _format_supported_sse_event(ui_event, denied_tool_call_ids)
    return state, denied_tool_call_ids


async def _stream_ui_response(
    *,
    session_id: str,
    run_id: str,
    ui_stream: stream_store.UIStream,
    start_index: int,
) -> AsyncGenerator[str]:
    lock = _ui_stream_lock(ui_stream.id)
    if lock.locked():
        async for chunk in _tail_ui_response(
            ui_stream=ui_stream,
            start_index=start_index,
        ):
            yield chunk
        return

    await lock.acquire()
    try:
        async for chunk in _produce_ui_response(
            session_id=session_id,
            run_id=run_id,
            ui_stream=ui_stream,
            start_index=start_index,
        ):
            yield chunk
    finally:
        lock.release()


async def _tail_ui_response(
    *,
    ui_stream: stream_store.UIStream,
    start_index: int,
) -> AsyncGenerator[str]:
    next_index = start_index
    while True:
        chunks = await stream_store.list_ui_chunks(ui_stream.id, next_index)
        if chunks:
            for stored_chunk in chunks:
                next_index = stored_chunk.index + 1
                yield stored_chunk.chunk
            continue

        current = await stream_store.get_ui_stream(ui_stream.id)
        if current is None or current.status != "running":
            return

        await asyncio.sleep(0.5)


async def _append_ui_event_chunks(
    *,
    ui_stream: stream_store.UIStream,
    source_event: stream_store.StreamEvent,
    state: Any,
    denied_tool_call_ids: set[str],
    next_ui_index: int,
) -> tuple[list[str], int]:
    try:
        new_chunks: list[str] = []
        agent_event = _load_agent_event(source_event.data)
        for ui_event in _ui_events_for_agent_event(state, agent_event):
            sse_chunk = _format_supported_sse_event(ui_event, denied_tool_call_ids)
            if sse_chunk is not None:
                index = await stream_store.append_ui_chunk(ui_stream.id, sse_chunk)
                next_ui_index = max(next_ui_index, index + 1)
                new_chunks.append(sse_chunk)
        await stream_store.complete_ui_stream_source_index(
            ui_stream.id,
            source_event.index,
            source_event.index + 1,
        )
        return new_chunks, next_ui_index
    except Exception:
        await stream_store.release_ui_stream_source_index(
            ui_stream.id,
            source_event.index,
        )
        raise


async def _append_ui_finish_chunks(
    *,
    session_id: str,
    ui_stream: stream_store.UIStream,
    source_index: int,
    state: Any,
    denied_tool_call_ids: set[str],
) -> list[str]:
    try:
        finish_chunks: list[str] = []
        for ui_event in state.finish():
            sse_chunk = _format_supported_sse_event(ui_event, denied_tool_call_ids)
            if sse_chunk is not None:
                await stream_store.append_ui_chunk(ui_stream.id, sse_chunk)
                finish_chunks.append(sse_chunk)

        done = ai_sdk_outbound_stream.format_done_sse()
        await stream_store.append_ui_chunk(ui_stream.id, done)
        await stream_store.complete_ui_stream_source_index(
            ui_stream.id,
            source_index,
            source_index,
        )
        await stream_store.set_status(ui_stream.id, "completed")
        await sessions.clear_active_stream(session_id, ui_stream.id)
        finish_chunks.append(done)
        return finish_chunks
    except Exception:
        await stream_store.release_ui_stream_source_index(ui_stream.id, source_index)
        raise


async def _produce_ui_response(
    *,
    session_id: str,
    run_id: str,
    ui_stream: stream_store.UIStream,
    start_index: int,
) -> AsyncGenerator[str]:
    next_ui_index = start_index
    for stored_chunk in await stream_store.list_ui_chunks(ui_stream.id, start_index):
        next_ui_index = stored_chunk.index + 1
        yield stored_chunk.chunk

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
            async for chunk in _tail_ui_response(
                ui_stream=ui_stream,
                start_index=next_ui_index,
            ):
                yield chunk
            return

        write_task = asyncio.create_task(
            _append_ui_event_chunks(
                ui_stream=ui_stream,
                source_event=event,
                state=state,
                denied_tool_call_ids=denied_tool_call_ids,
                next_ui_index=next_ui_index,
            )
        )
        try:
            new_chunks, next_ui_index = await asyncio.shield(write_task)
        except asyncio.CancelledError:
            await write_task
            raise
        source_next_index = event.index + 1
        for sse_chunk in new_chunks:
            yield sse_chunk

    claimed = await stream_store.claim_ui_stream_source_index(
        ui_stream.id,
        source_next_index,
    )
    if not claimed:
        async for chunk in _tail_ui_response(
            ui_stream=ui_stream,
            start_index=next_ui_index,
        ):
            yield chunk
        return

    finish_task = asyncio.create_task(
        _append_ui_finish_chunks(
            session_id=session_id,
            ui_stream=ui_stream,
            source_index=source_next_index,
            state=state,
            denied_tool_call_ids=denied_tool_call_ids,
        )
    )
    try:
        finish_chunks = await asyncio.shield(finish_task)
    except asyncio.CancelledError:
        await finish_task
        raise
    for sse_chunk in finish_chunks:
        yield sse_chunk


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
        start_index = 0
    else:
        try:
            start_index = int(raw_start_index)
        except ValueError:
            raise fastapi.HTTPException(
                status_code=400,
                detail="startIndex must be a non-negative integer",
            ) from None

    tail_index = await stream_store.count_ui_chunks(ui_stream.id) - 1
    if start_index < 0:
        start_index = max(0, tail_index + 1 + start_index)

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
            workflows.run_agent,
            session_id,
            stream_id,
            workflows.dump_messages(prepared.messages),
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
