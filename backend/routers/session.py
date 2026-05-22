"""Session management endpoints."""

from __future__ import annotations

from typing import Any

import fastapi
import pydantic
from ai.agents.ui.ai_sdk import to_ui_messages

import agent
import sessions as session_store

router = fastapi.APIRouter()


class CreateSessionRequest(pydantic.BaseModel):
    """Body for ``POST /sessions``."""

    id: str
    title: str | None = None


@router.get("/sessions")
async def list_sessions() -> list[session_store.Session]:
    """Return all sessions, most recent first."""
    return await session_store.list_sessions()


@router.post("/sessions", status_code=201)
async def create_session(body: CreateSessionRequest) -> session_store.Session:
    """Create a new session with a client-generated ID."""
    return await session_store.create_session(body.id, body.title)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Return a session with its messages in AI SDK UI shape."""
    session = await session_store.get_session(session_id)
    if not session:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")

    ai_msgs = await session_store.load_ai_messages(session_id)
    ui_msgs = to_ui_messages(ai_msgs)

    result = session.model_dump()
    result["messages"] = [m.model_dump(by_alias=True) for m in ui_msgs]
    return result


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """Delete a session (cascades to messages)."""
    found = await session_store.delete_session(session_id)
    if not found:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


@router.post("/sessions/{session_id}/title")
async def generate_title(session_id: str) -> session_store.Session:
    """Generate an LLM title for a session from its first message."""
    session = await session_store.get_session(session_id)
    if not session:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")

    if session.title:
        return session

    messages = await session_store.load_ai_messages(session_id)
    first_text = session_store.first_user_text(messages)
    if not first_text:
        raise fastapi.HTTPException(
            status_code=400, detail="No user message to generate title from"
        )

    title = await agent.generate_title(first_text)
    row = await session_store.update_session_title(session_id, title)
    if row is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return row
