"""Chat, file upload, and file serving endpoints."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator
from typing import Any

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
import db

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


def _extract_blob_pathname(url: str) -> str | None:
    """Extract the blob pathname from a proxy URL, or return None."""
    if url.startswith(FILES_PREFIX):
        return url[len(FILES_PREFIX) :]
    return None


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
                _extract_blob_pathname(part.data)
                if isinstance(part, ai_messages.FilePart) and isinstance(part.data, str)
                else None
            )
            if isinstance(part, ai_messages.FilePart) and pathname is not None:
                async with AsyncBlobClient() as client:
                    blob = await client.get(pathname, access="private")

                b64 = base64.b64encode(blob.content).decode("ascii")
                media_type = blob.content_type or part.media_type
                data_url = f"data:{media_type};base64,{b64}"

                new_parts.append(
                    ai_messages.FilePart(
                        data=data_url,
                        media_type=media_type,
                        filename=part.filename,
                    )
                )
            else:
                new_parts.append(part)

        result.append(msg.model_copy(update={"parts": new_parts}))
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _serialize_parts(parts: list[ai_messages.Part]) -> list[dict[str, Any]]:
    """Serialize message parts to JSON-safe dicts for DB storage."""
    return [p.model_dump(mode="json") for p in parts]


_ASSISTANT_BUBBLE_ROLES = frozenset({"assistant", "tool", "internal"})


def _turn_id_from_message_id(message_id: str) -> str | None:
    """Extract the UI turn id from an adapter-stable message id."""
    for marker in (":assistant:", ":tool:", ":internal:"):
        if marker in message_id:
            return message_id.split(marker, 1)[0]
    return None


def _message_turn_id(message: ai_messages.Message) -> str | None:
    return message.turn_id or _turn_id_from_message_id(message.id)


def _stable_part_id(
    message_id: str,
    part: ai_messages.Part,
    index: int,
) -> str:
    """Return the same part-id shape used by the AI SDK UI inbound adapter."""
    match part:
        case ai_messages.TextPart():
            return f"{message_id}:text:{index}"
        case ai_messages.ReasoningPart():
            return f"{message_id}:reasoning:{index}"
        case ai_messages.ToolCallPart(tool_call_id=tool_call_id):
            return f"{message_id}:call:{tool_call_id}"
        case ai_messages.ToolResultPart(tool_call_id=tool_call_id):
            return f"{message_id}:result:{tool_call_id}"
        case ai_messages.BuiltinToolCallPart(tool_call_id=tool_call_id):
            return f"{message_id}:builtin-call:{tool_call_id}"
        case ai_messages.BuiltinToolReturnPart(tool_call_id=tool_call_id):
            return f"{message_id}:builtin-result:{tool_call_id}"
        case ai_messages.HookPart():
            return f"{message_id}:hook:{index}"
        case ai_messages.FilePart():
            return f"{message_id}:file:{index}"
    raise TypeError(f"Unsupported message part: {type(part).__name__}")


def _with_stable_part_ids(
    message_id: str,
    parts: list[ai_messages.Part],
) -> list[ai_messages.Part]:
    counts: dict[str, int] = {}
    result: list[ai_messages.Part] = []

    for index, part in enumerate(parts):
        base = _stable_part_id(message_id, part, index)
        seen = counts.get(base, 0)
        counts[base] = seen + 1
        part_id = base if seen == 0 else f"{base}:{seen}"
        result.append(part.model_copy(update={"id": part_id}))

    return result


def _canonicalize_run_messages(
    messages: list[ai_messages.Message],
) -> list[ai_messages.Message]:
    """Normalize message ids to the UI adapter's stable round-trip shape."""
    result: list[ai_messages.Message] = []
    i = 0

    while i < len(messages):
        message = messages[i]
        if message.role == "system":
            i += 1
            continue

        if message.role != "assistant":
            result.append(
                message.model_copy(
                    update={
                        "parts": _with_stable_part_ids(
                            message.id,
                            message.parts,
                        )
                    }
                )
            )
            i += 1
            continue

        turn_id = _message_turn_id(message) or message.id
        assistant_index = 0
        tool_index = 0
        internal_index = 0

        while i < len(messages) and messages[i].role in _ASSISTANT_BUBBLE_ROLES:
            current = messages[i]
            match current.role:
                case "assistant":
                    message_id = f"{turn_id}:assistant:{assistant_index}"
                    assistant_index += 1
                case "tool":
                    message_id = f"{turn_id}:tool:{tool_index}"
                    tool_index += 1
                case "internal":
                    message_id = f"{turn_id}:internal:{internal_index}"
                    internal_index += 1
                case _:
                    # Narrowed by the while condition; keeps type checkers happy.
                    message_id = current.id

            result.append(
                current.model_copy(
                    update={
                        "id": message_id,
                        "turn_id": turn_id,
                        "parts": _with_stable_part_ids(
                            message_id,
                            current.parts,
                        ),
                    }
                )
            )
            i += 1

    return result


async def _persist_run_messages(
    session_id: str,
    messages: list[ai_messages.Message],
) -> None:
    """Batch-upsert all messages from this run.

    The system message we prepended is dropped — it's reconstructed at
    request time from ``agent.SYSTEM`` and should never round-trip
    through the database.
    """
    messages = _canonicalize_run_messages(messages)
    rows: list[tuple[str, str, int, str, list[dict[str, Any]]]] = [
        (m.id, session_id, seq, m.role, _serialize_parts(m.parts))
        for seq, m in enumerate(messages)
    ]
    if rows:
        await db.save_messages_batch(rows)


async def _no_op_sse() -> AsyncGenerator[str]:
    """Return a valid empty UI-message stream for duplicate completed runs."""
    yield 'data: {"type":"finish","finishReason":"stop"}\n\n'


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


class ChatRequest(pydantic.BaseModel):
    """Request body for the chat endpoint."""

    messages: list[UIMessage]
    session_id: str
    request_id: str | None = None


@router.post("/chat")
async def chat(request: ChatRequest) -> fastapi.responses.StreamingResponse:
    """Handle chat requests and stream responses."""
    session_id = request.session_id

    # Ensure the session exists (create if somehow missing).
    if not await db.get_session(session_id):
        await db.create_session(session_id)

    # Convert UI messages to SDK messages and extract any approval responses.
    messages, approvals = to_messages(request.messages)
    apply_approvals(approvals)

    # Inline blob URLs so the gateway can see the bytes.
    messages = await _inline_file_parts(messages)

    # Prepend the system prompt.
    system = ai.system_message(agent.SYSTEM)
    full_messages = [system, *messages]

    request_id = request.request_id
    if request_id is not None:
        run_status = await db.start_chat_run(session_id, request_id)
        if run_status == "completed":
            return fastapi.responses.StreamingResponse(
                _no_op_sse(),
                headers=UI_MESSAGE_STREAM_HEADERS,
            )
        if run_status == "in_progress":
            raise fastapi.HTTPException(
                status_code=409,
                detail="Chat run already in progress",
            )

    model = agent.get_model()

    async def stream_response() -> AsyncGenerator[str]:
        try:
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
                            ai.abort_pending_hook(event.hook)
                        yield event

                async for chunk in to_sse(process()):
                    yield chunk

                # Persist the full updated history once the run finishes
                # (whether normally or via approval suspension).
                await _persist_run_messages(session_id, result.messages)
                await db.touch_session(session_id)
                if request_id is not None:
                    await db.complete_chat_run(session_id, request_id)
        except BaseException:
            if request_id is not None:
                await db.fail_chat_run(session_id, request_id)
            raise

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=UI_MESSAGE_STREAM_HEADERS,
    )
