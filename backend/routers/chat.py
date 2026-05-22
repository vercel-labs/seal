"""Chat, file upload, and file serving endpoints."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator

import ai
import fastapi
import fastapi.responses
import pydantic
from ai import events as ai_events
from ai import messages as ai_messages
from ai.agents.ui.ai_sdk import (
    UI_MESSAGE_STREAM_HEADERS,
    UIMessage,
    apply_approvals,
    to_messages,
    to_sse,
)
from vercel.blob import AsyncBlobClient

import agent
import sessions as session_store

router = fastapi.APIRouter()

# Prefix used by proxy URLs returned from the upload endpoint.
# Includes /api so the browser can fetch directly (Vercel routes /api/* to
# the backend and strips the prefix before forwarding).
FILES_PREFIX = "/api/files/"


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
        url=f"{FILES_PREFIX}{result.pathname}",
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
# File inlining
# ---------------------------------------------------------------------------


async def _inline_file_parts(
    messages: list[ai_messages.Message],
) -> list[ai_messages.Message]:
    """Replace proxy-URL file parts with inline base64 data URLs.

    The AI Gateway requires file content as data URLs (not raw HTTP URLs).
    Our proxy URLs (``/api/files/...``) aren't reachable from the gateway,
    so we fetch the blob content here and inline it before sending.
    """
    result: list[ai_messages.Message] = []
    for msg in messages:
        new_parts: list[ai_messages.Part] = []
        for part in msg.parts:
            pathname = (
                part.data[len(FILES_PREFIX) :]
                if (
                    isinstance(part, ai_messages.FilePart)
                    and isinstance(part.data, str)
                    and part.data.startswith(FILES_PREFIX)
                )
                else None
            )
            if isinstance(part, ai_messages.FilePart) and pathname is not None:
                async with AsyncBlobClient() as client:
                    blob = await client.get(pathname, access="private")

                b64 = base64.b64encode(blob.content).decode("ascii")
                media_type = blob.content_type or part.media_type
                data_url = f"data:{media_type};base64,{b64}"

                new_parts.append(
                    part.model_copy(update={"data": data_url, "media_type": media_type})
                )
            else:
                new_parts.append(part)

        result.append(msg.model_copy(update={"parts": new_parts}))
    return result


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


class ChatRequest(pydantic.BaseModel):
    """Request body for the chat endpoint."""

    messages: list[UIMessage]
    session_id: str


@router.post("/chat")
async def chat(request: ChatRequest) -> fastapi.responses.StreamingResponse:
    """Handle chat requests and stream responses."""
    session_id = request.session_id

    await session_store.get_or_create_session(session_id)
    stored_messages = await session_store.load_ai_messages(session_id)
    try:
        latest_user_ui = session_store.validate_chat_history(
            request_messages=request.messages,
            stored_messages=stored_messages,
        )
    except session_store.HistoryMismatchError as exc:
        raise fastapi.HTTPException(status_code=409, detail=str(exc)) from exc

    latest_user: ai_messages.Message | None = None
    if latest_user_ui is not None:
        latest_user_messages, _ = to_messages([latest_user_ui])
        if len(latest_user_messages) != 1 or latest_user_messages[0].role != "user":
            raise fastapi.HTTPException(
                status_code=400,
                detail="Latest message must be a user message",
            )
        latest_user = latest_user_messages[0]

    # Convert UI messages only to extract approval responses. Conversation
    # history itself comes from the DB framework snapshot above.
    _, approvals = to_messages(request.messages)
    apply_approvals(approvals)

    # Inline blob URLs so the gateway can see the bytes.
    messages = [m for m in stored_messages if m.role not in ("system", "internal")]
    if latest_user is not None:
        messages.append(latest_user)
    messages = await _inline_file_parts(messages)

    if latest_user is None and not approvals:
        raise fastapi.HTTPException(
            status_code=400,
            detail="No new user message or approval response to process",
        )

    # Prepend the system prompt.
    system = ai.system_message(agent.SYSTEM)
    full_messages = [system, *messages]

    model = agent.get_model()

    async def stream_response() -> AsyncGenerator[str]:
        pending_hook_messages: list[ai_messages.Message] = []

        async with agent.seal.run(model, full_messages) as result:

            async def process() -> AsyncGenerator[ai_events.AgentEvent]:
                async for event in result:
                    # Suspend the run when a tool approval is pending.
                    # The SDK persists the suspension as ``is_hook_pending``
                    # placeholders on the message history; the next request
                    # carrying the approval response resumes natively.
                    if (
                        isinstance(event, ai_events.HookEvent)
                        and event.hook.status == "pending"
                    ):
                        pending_hook_messages.append(event.message)
                        ai.abort_pending_hook(event.hook)
                    yield event

            async for chunk in to_sse(process()):
                yield chunk

            # Persist the full updated history once the run finishes
            # (whether normally or via approval suspension).
            await session_store.persist_ai_messages(
                session_id,
                session_store.with_pending_hook_messages(
                    result.messages,
                    pending_hook_messages,
                ),
            )
            await session_store.touch_session(session_id)

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=UI_MESSAGE_STREAM_HEADERS,
    )
