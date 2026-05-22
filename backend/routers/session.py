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


class RenameSessionRequest(pydantic.BaseModel):
    """Body for ``PATCH /sessions/{session_id}``."""

    title: str


class MoveLeafRequest(pydantic.BaseModel):
    """Body for moving a session leaf."""

    entry_id: str | None = None


class LabelEntryRequest(pydantic.BaseModel):
    """Body for setting or clearing an entry label."""

    label: str | None = None


class ForkSessionRequest(pydantic.BaseModel):
    """Body for forking or cloning a session."""

    id: str
    entry_id: str | None = None


class CompactSessionRequest(pydantic.BaseModel):
    """Body for compacting a session."""

    summary: str
    keep_last: int = 8


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


@router.patch("/sessions/{session_id}")
async def rename_session(
    session_id: str,
    body: RenameSessionRequest,
) -> session_store.Session:
    """Rename a session."""
    row = await session_store.update_session_title(session_id, body.title)
    if row is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return row


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


@router.get("/sessions/{session_id}/tree")
async def get_session_tree(session_id: str) -> dict[str, Any]:
    """Return the append-only session tree."""
    tree = await session_store.get_tree(session_id)
    if tree is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return tree


@router.post("/sessions/{session_id}/branch")
async def branch_session(
    session_id: str,
    body: MoveLeafRequest,
) -> dict[str, str]:
    """Move the active leaf to an entry."""
    found = await session_store.move_leaf(session_id, body.entry_id)
    if not found:
        raise fastapi.HTTPException(
            status_code=404,
            detail="Session or entry not found",
        )
    return {"status": "branched"}


@router.post("/sessions/{session_id}/entries/{entry_id}/label")
async def label_session_entry(
    session_id: str,
    entry_id: str,
    body: LabelEntryRequest,
) -> dict[str, str]:
    """Set or clear an entry label."""
    found = await session_store.label_entry(session_id, entry_id, body.label)
    if not found:
        raise fastapi.HTTPException(
            status_code=404,
            detail="Session or entry not found",
        )
    return {"status": "labeled"}


@router.post("/sessions/{session_id}/fork", status_code=201)
async def fork_session(
    session_id: str,
    body: ForkSessionRequest,
) -> session_store.Session:
    """Fork a session into a new session."""
    session = await session_store.fork_session(session_id, body.id, body.entry_id)
    if session is None:
        raise fastapi.HTTPException(
            status_code=404,
            detail="Session or entry not found",
        )
    return session


@router.post("/sessions/{session_id}/clone", status_code=201)
async def clone_session(
    session_id: str,
    body: ForkSessionRequest,
) -> session_store.Session:
    """Clone the active branch into a new session."""
    session = await session_store.clone_session(session_id, body.id)
    if session is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/sessions/{session_id}/compact")
async def compact_session(
    session_id: str,
    body: CompactSessionRequest,
) -> dict[str, str]:
    """Compact a session by replacing older context with a summary."""
    found = await session_store.compact_session(
        session_id,
        body.summary,
        keep_last=body.keep_last,
    )
    if not found:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return {"status": "compacted"}
