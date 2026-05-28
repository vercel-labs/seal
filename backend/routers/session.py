"""Session management endpoints."""

from __future__ import annotations

from typing import Any

import fastapi
import pydantic
from ai.agents.ui.ai_sdk import to_ui_messages

from core import durable_agent, sessions, stream_store

router = fastapi.APIRouter()


class CreateSessionRequest(pydantic.BaseModel):
    id: str
    title: str | None = None


def _meta(session: sessions.SessionRecord) -> sessions.SessionMeta:
    return sessions.SessionMeta(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.get("/sessions")
async def list_sessions() -> list[sessions.SessionMeta]:
    return await sessions.list_sessions()


@router.post("/sessions", status_code=201)
async def create_session(body: CreateSessionRequest) -> sessions.SessionMeta:
    return await sessions.create_session(body.id, title=body.title)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    session = await sessions.get_session(session_id)
    if session is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    messages = session.messages
    if session.active_run_id is not None and session.active_stream_id is not None:
        active_stream = await stream_store.get_ui_stream(session.active_stream_id)
        if active_stream is not None and active_stream.status == "running":
            messages = messages[: active_stream.history_message_count]

    ui_messages = to_ui_messages(messages)
    result = _meta(session).model_dump()
    result["messages"] = [
        message.model_dump(mode="json", by_alias=True) for message in ui_messages
    ]
    return result


@router.post("/sessions/{session_id}/title")
async def generate_title(session_id: str) -> sessions.SessionMeta:
    session = await sessions.get_session(session_id)
    if session is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    if session.title:
        return _meta(session)

    first_text = next(
        (message.text for message in session.messages if message.role == "user"),
        None,
    )
    if not first_text:
        raise fastapi.HTTPException(
            status_code=400,
            detail="No user message to generate title from",
        )

    meta = await sessions.set_title(
        session_id,
        await durable_agent.generate_title(first_text),
    )
    if meta is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return meta


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    if not await sessions.delete_session(session_id):
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}
