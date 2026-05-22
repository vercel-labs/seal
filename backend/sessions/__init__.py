"""Session storage package.

Public API:

- :class:`Session` -- high-level chat-session operations.
- :class:`SessionRepo` -- collection of sessions (create, open, list, delete,
  fork).
- :class:`SessionMeta` -- serializable session header.
- :class:`SessionError` -- raised for storage and tree errors.
- :class:`SessionEntry` -- discriminated union of entry types.
- :func:`get_repo` -- returns the configured repo (JSONL or Postgres).
- :func:`set_repo` -- override the configured repo (used by tests).
- :func:`ensure_schema` -- prepares backing storage on startup.

Backend selection: ``DATABASE_URL`` chooses Postgres; otherwise JSONL in
``$SEAL_SESSIONS_DIR`` (default ``data/sessions``).
"""

from __future__ import annotations

import os
from pathlib import Path

from .core import (
    ForkPosition,
    Session,
    SessionError,
    SessionLog,
    SessionMeta,
    SessionRepo,
)
from .entries import (
    BranchSummaryEntry,
    CompactionEntry,
    LabelEntry,
    LeafEntry,
    MessageEntry,
    SessionEntry,
    SessionInfoEntry,
)
from .jsonl import JsonlSessionLog, JsonlSessionRepo
from .postgres import PostgresSessionLog, PostgresSessionRepo

__all__ = [
    "BranchSummaryEntry",
    "CompactionEntry",
    "ForkPosition",
    "JsonlSessionLog",
    "JsonlSessionRepo",
    "LabelEntry",
    "LeafEntry",
    "MessageEntry",
    "PostgresSessionLog",
    "PostgresSessionRepo",
    "Session",
    "SessionEntry",
    "SessionError",
    "SessionInfoEntry",
    "SessionLog",
    "SessionMeta",
    "SessionRepo",
    "ensure_schema",
    "get_repo",
    "set_repo",
]

_DEFAULT_DIR = Path("data/sessions")
_repo: SessionRepo | None = None


def get_repo() -> SessionRepo:
    """Return the configured repo, instantiating it on first call."""
    global _repo
    if _repo is None:
        if os.environ.get("DATABASE_URL"):
            _repo = PostgresSessionRepo()
        else:
            configured = os.environ.get("SEAL_SESSIONS_DIR")
            root = Path(configured) if configured else _DEFAULT_DIR
            _repo = JsonlSessionRepo(root)
    return _repo


def set_repo(repo: SessionRepo | None) -> None:
    """Override the configured repo (used by tests)."""
    global _repo
    _repo = repo


async def ensure_schema() -> None:
    """Prepare the configured backing storage (called from FastAPI startup)."""
    await get_repo().ensure_schema()
