"""Session management endpoints."""

from __future__ import annotations

from typing import Any

import fastapi
import pydantic
from ai import messages as ai_messages
from ai.agents.ui.ai_sdk import to_ui_messages

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


def _stored_to_ai_messages(
    rows: list[db.StoredMessage],
) -> list[ai_messages.Message]:
    """Decode canonical persisted rows into ``ai.messages.Message`` objects."""
    return [
        ai_messages.Message.model_validate(
            {
                "id": row.id,
                "turn_id": row.turn_id,
                "role": row.role,
                "parts": row.parts,
            }
        )
        for row in rows
    ]


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


def _extract_first_user_text(messages: list[ai_messages.Message]) -> str | None:
    """Return the text of the first user message, or None."""
    for msg in messages:
        if msg.role != "user":
            continue
        if msg.text:
            return msg.text
    return None


@router.post("/sessions/{session_id}/title")
async def generate_title(session_id: str) -> db.Session:
    """Generate an LLM title for a session from its first message."""
    session = await db.get_session(session_id)
    if not session:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")

    if session.title:
        return session

    messages = _stored_to_ai_messages(await db.get_messages(session_id))
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
