"""Session log entry types.

Entries are append-only; mutations (e.g., a new title, a leaf move, a label
change) are recorded as fresh entries. Each entry's ``parent_id`` chains it to
the previous active leaf, forming the session tree.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

import pydantic
from ai import messages as ai_messages


def now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def new_entry_id(existing: set[str]) -> str:
    """Return a short, unique entry id not already in ``existing``."""
    for _ in range(100):
        candidate = uuid.uuid4().hex[:8]
        if candidate not in existing:
            return candidate
    return uuid.uuid4().hex


class _EntryBase(pydantic.BaseModel):
    id: str
    parent_id: str | None
    timestamp: str


class MessageEntry(_EntryBase):
    """One AI message in the chat history."""

    type: Literal["message"] = "message"
    message: ai_messages.Message


class LeafEntry(_EntryBase):
    """Records a move of the active leaf to ``target_id`` (or ``None`` to clear)."""

    type: Literal["leaf"] = "leaf"
    target_id: str | None


class LabelEntry(_EntryBase):
    """Set or clear (``label=None``) a human-readable label on an entry."""

    type: Literal["label"] = "label"
    target_id: str
    label: str | None


class SessionInfoEntry(_EntryBase):
    """Records a session title change."""

    type: Literal["session_info"] = "session_info"
    title: str | None


class BranchSummaryEntry(_EntryBase):
    """A short summary attached when moving to a different branch."""

    type: Literal["branch_summary"] = "branch_summary"
    from_id: str | None
    summary: str


class CompactionEntry(_EntryBase):
    """Compacts the path: replays ``summary`` then everything from
    ``first_kept_entry_id`` onward (or nothing prior if it is ``None``)."""

    type: Literal["compaction"] = "compaction"
    summary: str
    first_kept_entry_id: str | None
    tokens_before: int


SessionEntry = Annotated[
    MessageEntry
    | LeafEntry
    | LabelEntry
    | SessionInfoEntry
    | BranchSummaryEntry
    | CompactionEntry,
    pydantic.Field(discriminator="type"),
]


ENTRY_ADAPTER: pydantic.TypeAdapter[SessionEntry] = pydantic.TypeAdapter(SessionEntry)


def leaf_after(entry: SessionEntry) -> str | None:
    """Return the active leaf id after ``entry`` is appended."""
    if isinstance(entry, LeafEntry):
        return entry.target_id
    return entry.id
