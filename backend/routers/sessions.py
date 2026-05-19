"""Session management endpoints."""

from __future__ import annotations

from typing import Any

import fastapi
import pydantic
from ai import messages as ai_messages
from ai.agents.ui.ai_sdk import UIMessage, to_messages, to_ui_messages

import agent
import db

router = fastapi.APIRouter()


class CreateSessionRequest(pydantic.BaseModel):
    """Body for ``POST /sessions``."""

    id: str
    title: str | None = None


@router.get("/sessions")
async def list_sessions() -> list[db.Session]:
    """Return all sessions, most recent first."""
    return await db.list_sessions()


@router.post("/sessions", status_code=201)
async def create_session(body: CreateSessionRequest) -> db.Session:
    """Create a new session with a client-generated ID."""
    return await db.create_session(body.id, body.title)


def _is_legacy_ui_part(part: dict[str, Any]) -> bool:
    """Detect rows persisted in the old AI-SDK-UI-shape format.

    Legacy rows used the wire-protocol part keys (``toolCallId``,
    ``toolName``, ``state``, ``input``/``output``).  Canonical SDK parts
    instead carry a ``kind`` discriminator (``text``, ``tool_call``,
    ``tool_result``, ...).
    """
    return "kind" not in part and (
        part.get("type", "").startswith("tool-")
        or part.get("type") == "tool-invocation"
    )


def _stored_to_ai_messages(
    rows: list[db.StoredMessage],
) -> list[ai_messages.Message]:
    """Decode persisted rows into ``ai.messages.Message`` objects.

    Modern rows store canonical ``Message`` JSON.  Legacy rows store the
    UI-protocol shape; we route them back through ``to_messages`` so
    callers see a uniform internal representation.
    """
    legacy_ui_msgs: list[UIMessage] = []
    canonical: list[ai_messages.Message] = []

    for row in rows:
        # Heuristic: if any tool-like part lacks ``kind``, the whole row
        # is from the legacy UI-shape era.  We route the whole UIMessage
        # through the inbound adapter so multi-part bubbles stay grouped.
        is_legacy = any(_is_legacy_ui_part(p) for p in row.parts)
        if is_legacy:
            legacy_ui_msgs.append(
                UIMessage.model_validate(
                    {"id": row.id, "role": row.role, "parts": row.parts}
                )
            )
        else:
            canonical.append(
                ai_messages.Message.model_validate(
                    {"id": row.id, "role": row.role, "parts": row.parts}
                )
            )

    if legacy_ui_msgs:
        decoded, _ = to_messages(legacy_ui_msgs)
        canonical = [*canonical, *decoded]

    return canonical


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Return a session with its messages in AI SDK UI shape."""
    session = await db.get_session(session_id)
    if not session:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")

    stored = await db.get_messages(session_id)
    ai_msgs = _stored_to_ai_messages(stored)
    ui_msgs = to_ui_messages(ai_msgs)

    result = session.model_dump()
    result["messages"] = [m.model_dump(by_alias=True) for m in ui_msgs]
    return result


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """Delete a session (cascades to messages)."""
    found = await db.delete_session(session_id)
    if not found:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


def _extract_first_user_text(messages: list[db.StoredMessage]) -> str | None:
    """Return the text of the first user text-part, or None."""
    for msg in messages:
        if msg.role != "user":
            continue
        for part in msg.parts:
            if part.get("type") == "text" and part.get("text"):
                return str(part["text"])
    return None


@router.post("/sessions/{session_id}/title")
async def generate_title(session_id: str) -> db.Session:
    """Generate an LLM title for a session from its first message."""
    session = await db.get_session(session_id)
    if not session:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")

    if session.title:
        return session

    messages = await db.get_messages(session_id)
    first_text = _extract_first_user_text(messages)
    if not first_text:
        raise fastapi.HTTPException(
            status_code=400, detail="No user message to generate title from"
        )

    title = await agent.generate_title(first_text)
    row = await db.update_session_title(session_id, title)
    if row is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return row
