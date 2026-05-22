"""Chat, file upload, and file serving endpoints."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator
from typing import Any, NamedTuple

import ai
import fastapi
import fastapi.responses
import pydantic
from ai import events as ai_events
from ai import messages as ai_messages
from ai.agents.hooks import TOOL_APPROVAL_HOOK_TYPE
from ai.agents.ui.ai_sdk import (
    UI_MESSAGE_STREAM_HEADERS,
    ApprovalResponse,
    UIMessage,
    apply_approvals,
    to_messages,
    to_sse,
)
from vercel.blob import AsyncBlobClient

import agent
import sessions

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
# Chat-request normalization
# ---------------------------------------------------------------------------


class PreparedChat(NamedTuple):
    """Canonical messages and side effects extracted from a useChat request."""

    messages: list[ai_messages.Message]
    approvals: list[ApprovalResponse]
    has_work: bool
    changed: bool


def _message_signature(message: ai_messages.Message) -> str:
    return json.dumps(
        message.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )


def prepare_chat_request(
    *,
    request_messages: list[UIMessage],
    stored_messages: list[ai_messages.Message],
) -> PreparedChat:
    """Normalize a useChat request without raw UI-shape history validation."""
    request_ai_messages, approvals = to_messages(request_messages)
    if not request_ai_messages:
        return PreparedChat(stored_messages, approvals, bool(approvals), False)

    if not stored_messages:
        return PreparedChat(request_ai_messages, approvals, True, True)

    latest = request_ai_messages[-1]
    if latest.role == "user":
        stored_by_id = {m.id: m for m in stored_messages}
        stored_latest = stored_by_id.get(latest.id)
        if stored_latest is None:
            return PreparedChat([*stored_messages, latest], approvals, True, True)
        if _message_signature(stored_latest) != _message_signature(latest):
            return PreparedChat(request_ai_messages, approvals, True, True)

    if approvals:
        return PreparedChat(stored_messages, approvals, True, False)

    return PreparedChat(stored_messages, approvals, False, False)


def _tool_call_id_for_hook(hook: ai_messages.HookPart[Any]) -> str | None:
    if hook.hook_type != TOOL_APPROVAL_HOOK_TYPE:
        return None
    prefix = "approve_"
    if not hook.hook_id.startswith(prefix):
        return None
    return hook.hook_id[len(prefix) :]


def _turn_id_for_tool_call(
    messages: list[ai_messages.Message],
    tool_call_id: str,
) -> str | None:
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        if any(part.tool_call_id == tool_call_id for part in message.tool_calls):
            return message.turn_id or message.id
    return None


def with_pending_hook_messages(
    messages: list[ai_messages.Message],
    pending_hook_messages: list[ai_messages.Message],
) -> list[ai_messages.Message]:
    """Attach pending internal hook messages to the persisted snapshot."""
    result = list(messages)
    for message in pending_hook_messages:
        if len(message.parts) != 1 or not isinstance(
            message.parts[0], ai_messages.HookPart
        ):
            result.append(message)
            continue

        tool_call_id = _tool_call_id_for_hook(message.parts[0])
        turn_id = (
            _turn_id_for_tool_call(messages, tool_call_id)
            if tool_call_id is not None
            else None
        )
        result.append(message.model_copy(update={"turn_id": turn_id}))
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

    repo = sessions.get_repo()
    session = await repo.open(session_id) or await repo.create(session_id)
    stored_messages = await session.load_messages()
    prepared = prepare_chat_request(
        request_messages=request.messages,
        stored_messages=stored_messages,
    )
    apply_approvals(prepared.approvals)

    # Inline blob URLs so the gateway can see the bytes.
    messages = [m for m in prepared.messages if m.role not in ("system", "internal")]
    messages = await _inline_file_parts(messages)

    if not prepared.has_work:
        raise fastapi.HTTPException(
            status_code=400,
            detail="No new user message or approval response to process",
        )

    if prepared.changed:
        await session.save_messages(prepared.messages)

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
            await session.save_messages(
                with_pending_hook_messages(result.messages, pending_hook_messages)
            )

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=UI_MESSAGE_STREAM_HEADERS,
    )
