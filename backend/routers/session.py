"""Session management endpoints."""

from __future__ import annotations

from typing import Any

import fastapi
import pydantic
from ai.agents.ui.ai_sdk import to_ui_messages

import agent
import sessions

router = fastapi.APIRouter()


class CreateSessionRequest(pydantic.BaseModel):
    """Body for ``POST /sessions``."""

    id: str
    title: str | None = None


class RenameSessionRequest(pydantic.BaseModel):
    """Body for ``PATCH /sessions/{session_id}``."""

    title: str


class MoveLeafRequest(pydantic.BaseModel):
    """Body for moving a session leaf."""

    entry_id: str | None = None
    summary: str | None = None


class LabelEntryRequest(pydantic.BaseModel):
    """Body for setting or clearing an entry label."""

    label: str | None = None


class ForkSessionRequest(pydantic.BaseModel):
    """Body for forking a session."""

    id: str
    entry_id: str | None = None
    position: sessions.ForkPosition = "at"


class CompactSessionRequest(pydantic.BaseModel):
    """Body for compacting a session."""

    summary: str
    first_kept_entry_id: str | None = None


def _http_error_for(error: sessions.SessionError) -> fastapi.HTTPException:
    if error.code in ("not_found", "invalid_fork_target"):
        return fastapi.HTTPException(status_code=404, detail=str(error))
    return fastapi.HTTPException(status_code=400, detail=str(error))


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


@router.post("/sessions/{session_id}/title")
async def generate_title(session_id: str) -> sessions.SessionMeta:
    """Generate an LLM title for a session from its first message."""
    session = await sessions.get_repo().open(session_id)
    if session is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")

    if session.meta.title:
        return session.meta

    messages = await session.load_messages()
    first_text = next(
        (m.text for m in messages if m.role == "user" and m.text), None
    )
    if not first_text:
        raise fastapi.HTTPException(
            status_code=400, detail="No user message to generate title from"
        )

    title = await agent.generate_title(first_text)
    await session.set_title(title)
    return session.meta


@router.get("/sessions/{session_id}/tree")
async def get_session_tree(session_id: str) -> dict[str, Any]:
    """Return the append-only session tree."""
    session = await sessions.get_repo().open(session_id)
    if session is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return session.build_tree()


@router.post("/sessions/{session_id}/branch")
async def branch_session(
    session_id: str,
    body: MoveLeafRequest,
) -> dict[str, str]:
    """Move the active leaf to an entry, optionally attaching a branch summary."""
    session = await sessions.get_repo().open(session_id)
    if session is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    try:
        await session.move_to(body.entry_id, summary=body.summary)
    except sessions.SessionError as exc:
        raise _http_error_for(exc) from exc
    return {"status": "branched"}


@router.post("/sessions/{session_id}/entries/{entry_id}/label")
async def label_session_entry(
    session_id: str,
    entry_id: str,
    body: LabelEntryRequest,
) -> dict[str, str]:
    """Set or clear an entry label."""
    session = await sessions.get_repo().open(session_id)
    if session is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    try:
        await session.label(entry_id, body.label)
    except sessions.SessionError as exc:
        raise _http_error_for(exc) from exc
    return {"status": "labeled"}


@router.post("/sessions/{session_id}/fork", status_code=201)
async def fork_session(
    session_id: str,
    body: ForkSessionRequest,
) -> sessions.SessionMeta:
    """Fork a session into a new session."""
    try:
        new_session = await sessions.get_repo().fork(
            session_id,
            body.id,
            entry_id=body.entry_id,
            position=body.position,
        )
    except sessions.SessionError as exc:
        raise _http_error_for(exc) from exc
    return new_session.meta


@router.post("/sessions/{session_id}/compact")
async def compact_session(
    session_id: str,
    body: CompactSessionRequest,
) -> dict[str, str]:
    """Compact a session by replacing older context with a summary."""
    session = await sessions.get_repo().open(session_id)
    if session is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    messages = await session.load_messages()
    if not messages:
        raise fastapi.HTTPException(
            status_code=400, detail="No messages to compact"
        )
    tokens_before = sum(len(m.text) for m in messages) // 4
    try:
        await session.compact(
            body.summary,
            first_kept_entry_id=body.first_kept_entry_id,
            tokens_before=tokens_before,
        )
    except sessions.SessionError as exc:
        raise _http_error_for(exc) from exc
    return {"status": "compacted"}
