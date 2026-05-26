"""Session management endpoints."""

from __future__ import annotations

from typing import Any

import fastapi
from ai.agents.ui.ai_sdk import to_ui_messages


router = fastapi.APIRouter()



@router.get("/sessions")
async def list_sessions() -> list[sessions.SessionMeta]:
    """Return all sessions, most recent first."""
    return await sessions.get_repo().list()


@router.post("/sessions", status_code=201)
async def create_session(body: CreateSessionRequest) -> sessions.SessionMeta:
    """Create a new session with a client-generated ID."""
    session = await sessions.get_repo().create(body.id, title=body.title)
    return session.meta


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Return a session with its messages in AI SDK UI shape."""
    session = await sessions.get_repo().open(session_id)
    if session is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")

    messages = await session.load_messages()
    ui_msgs = to_ui_messages(messages)

    result = session.meta.model_dump()
    result["messages"] = [m.model_dump(by_alias=True) for m in ui_msgs]
    return result


@router.patch("/sessions/{session_id}")
async def rename_session(
    session_id: str,
    body: RenameSessionRequest,
) -> sessions.SessionMeta:
    """Rename a session."""
    session = await sessions.get_repo().open(session_id)
    if session is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    await session.set_title(body.title)
    return session.meta


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """Delete a session (cascades to messages)."""
    if not await sessions.get_repo().delete(session_id):
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}
