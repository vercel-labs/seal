"""Session storage and AI message helpers."""

from __future__ import annotations

import abc
import asyncio
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, cast
from urllib.parse import quote

import asyncpg  # type: ignore[import-untyped]
import pydantic
from ai import messages as ai_messages
from ai.agents.hooks import TOOL_APPROVAL_HOOK_TYPE
from ai.agents.ui.ai_sdk import ApprovalResponse, UIMessage, to_messages

import db

JsonDict = dict[str, Any]

_SESSION_VERSION = 1
_DEFAULT_SESSIONS_DIR = Path("data/sessions")
_POSTGRES_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    leaf_id     TEXT,
    cwd         TEXT,
    parent_id   TEXT
);

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS leaf_id TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS cwd TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS parent_id TEXT;

CREATE TABLE IF NOT EXISTS session_entries (
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    entry_id    TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    parent_id   TEXT,
    type        TEXT NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL,
    entry_json  JSONB NOT NULL,
    PRIMARY KEY (session_id, entry_id),
    UNIQUE (session_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_session_entries_session_seq
    ON session_entries(session_id, seq);
"""


class Session(pydantic.BaseModel):
    """Serializable session record."""

    id: str
    title: str | None = None
    created_at: str
    updated_at: str


class PreparedChat(NamedTuple):
    """Canonical messages and side effects extracted from a useChat request."""

    messages: list[ai_messages.Message]
    approvals: list[ApprovalResponse]
    has_work: bool
    changed: bool


class _SessionState(NamedTuple):
    session: Session
    entries: list[JsonDict]
    leaf_id: str | None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _new_entry_id(existing: set[str]) -> str:
    for _ in range(100):
        entry_id = uuid.uuid4().hex[:8]
        if entry_id not in existing:
            return entry_id
    return uuid.uuid4().hex


def _message_to_json(message: ai_messages.Message) -> JsonDict:
    return message.model_dump(mode="json")


def _messages_to_json(messages: list[ai_messages.Message]) -> list[JsonDict]:
    return [_message_to_json(m) for m in messages if m.role != "system"]


def _messages_from_json(items: list[Any]) -> list[ai_messages.Message]:
    return [ai_messages.Message.model_validate(item) for item in items]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _messages_equal(
    left: list[ai_messages.Message],
    right: list[ai_messages.Message],
) -> bool:
    return _json_dumps(_messages_to_json(left)) == _json_dumps(_messages_to_json(right))


def _entry_leaf_after(entry: JsonDict) -> str | None:
    if entry["type"] == "leaf":
        return cast(str | None, entry.get("target_id"))
    return cast(str, entry["id"])


def _state_from_parts(
    *,
    session_id: str,
    title: str | None,
    created_at: str,
    entries: list[JsonDict],
    leaf_id: str | None = None,
) -> _SessionState:
    current_leaf = leaf_id
    current_title = title
    updated_at = created_at

    for entry in entries:
        updated_at = cast(str, entry.get("timestamp", updated_at))
        if entry["type"] == "session_info":
            current_title = cast(str | None, entry.get("title"))
        current_leaf = _entry_leaf_after(entry)

    return _SessionState(
        session=Session(
            id=session_id,
            title=current_title,
            created_at=created_at,
            updated_at=updated_at,
        ),
        entries=entries,
        leaf_id=current_leaf,
    )


def _path_to_root(entries: list[JsonDict], leaf_id: str | None) -> list[JsonDict]:
    if leaf_id is None:
        return []

    by_id = {cast(str, e["id"]): e for e in entries}
    path: list[JsonDict] = []
    current = by_id.get(leaf_id)
    while current is not None:
        path.insert(0, current)
        parent_id = cast(str | None, current.get("parent_id"))
        current = by_id.get(parent_id) if parent_id is not None else None
    return path


def _messages_for_leaf(
    entries: list[JsonDict],
    leaf_id: str | None,
) -> list[ai_messages.Message]:
    for entry in reversed(_path_to_root(entries, leaf_id)):
        if entry["type"] in {"messages", "compaction"}:
            return _messages_from_json(cast(list[Any], entry.get("messages", [])))
    return []


def _entry_summary(entry: JsonDict, labels: dict[str, str]) -> JsonDict:
    summary: JsonDict = {
        "id": entry["id"],
        "type": entry["type"],
        "parent_id": entry.get("parent_id"),
        "timestamp": entry.get("timestamp"),
        "label": labels.get(cast(str, entry["id"])),
    }

    if entry["type"] in {"messages", "compaction"}:
        messages = _messages_from_json(cast(list[Any], entry.get("messages", [])))
        last = next(
            (m for m in reversed(messages) if m.role in {"user", "assistant"}),
            None,
        )
        if last is not None:
            summary["role"] = last.role
            summary["text"] = last.text
        summary["message_count"] = len(messages)
    if entry["type"] == "session_info":
        summary["title"] = entry.get("title")
    if entry["type"] == "branch_summary":
        summary["summary"] = entry.get("summary")
    if entry["type"] == "compaction":
        summary["summary"] = entry.get("summary")
        summary["tokens_before"] = entry.get("tokens_before")
    if entry["type"] == "label":
        summary["target_id"] = entry.get("target_id")
    return summary


def _labels_by_id(entries: list[JsonDict]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for entry in entries:
        if entry["type"] != "label":
            continue
        target_id = cast(str, entry["target_id"])
        label = cast(str | None, entry.get("label"))
        if label:
            labels[target_id] = label
        else:
            labels.pop(target_id, None)
    return labels


class SessionStore(abc.ABC):
    """Storage boundary for local JSONL and Postgres session logs."""

    @abc.abstractmethod
    async def ensure_schema(self) -> None:
        """Create required backing storage."""

    @abc.abstractmethod
    async def create_session(
        self,
        session_id: str,
        title: str | None = None,
        parent_id: str | None = None,
    ) -> Session:
        """Create or return a session."""

    @abc.abstractmethod
    async def list_sessions(self) -> list[Session]:
        """List sessions by recent activity."""

    @abc.abstractmethod
    async def get_session(self, session_id: str) -> Session | None:
        """Return a session by id."""

    @abc.abstractmethod
    async def update_session_title(
        self,
        session_id: str,
        title: str,
    ) -> Session | None:
        """Set a display title."""

    @abc.abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""

    @abc.abstractmethod
    async def load_messages(self, session_id: str) -> list[ai_messages.Message]:
        """Load active-branch canonical framework messages."""

    @abc.abstractmethod
    async def save_messages(
        self,
        session_id: str,
        messages: list[ai_messages.Message],
        *,
        entry_type: str = "messages",
        metadata: JsonDict | None = None,
    ) -> None:
        """Append a canonical framework message snapshot."""

    @abc.abstractmethod
    async def get_tree(self, session_id: str) -> JsonDict | None:
        """Return append-only entries plus the current leaf."""

    @abc.abstractmethod
    async def move_leaf(self, session_id: str, entry_id: str | None) -> bool:
        """Move the active leaf."""

    @abc.abstractmethod
    async def label_entry(
        self,
        session_id: str,
        entry_id: str,
        label: str | None,
    ) -> bool:
        """Set or clear an entry label."""

    @abc.abstractmethod
    async def fork_session(
        self,
        source_session_id: str,
        new_session_id: str,
        entry_id: str | None = None,
    ) -> Session | None:
        """Create a new session from an entry or active leaf."""


class JsonlSessionStore(SessionStore):
    """Append-only JSONL session store for local development."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._locks: dict[str, asyncio.Lock] = {}

    async def ensure_schema(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{quote(session_id, safe='')}.jsonl"

    def _lock(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    def _read_state(self, session_id: str) -> _SessionState | None:
        path = self._path(session_id)
        if not path.exists():
            return None

        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            return None

        header = cast(JsonDict, json.loads(lines[0]))
        entries = [cast(JsonDict, json.loads(line)) for line in lines[1:]]
        return _state_from_parts(
            session_id=cast(str, header["id"]),
            title=cast(str | None, header.get("title")),
            created_at=cast(str, header["created_at"]),
            entries=entries,
        )

    def _append_entry(self, session_id: str, entry: JsonDict) -> None:
        with self._path(session_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, separators=(",", ":")))
            fh.write("\n")

    async def create_session(
        self,
        session_id: str,
        title: str | None = None,
        parent_id: str | None = None,
    ) -> Session:
        await self.ensure_schema()
        async with self._lock(session_id):
            state = self._read_state(session_id)
            if state is not None:
                return state.session

            timestamp = _now()
            header: JsonDict = {
                "type": "session",
                "version": _SESSION_VERSION,
                "id": session_id,
                "title": title,
                "created_at": timestamp,
                "cwd": os.getcwd(),
                "parent_id": parent_id,
            }
            self._path(session_id).write_text(
                json.dumps(header, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            return Session(
                id=session_id,
                title=title,
                created_at=timestamp,
                updated_at=timestamp,
            )

    async def list_sessions(self) -> list[Session]:
        await self.ensure_schema()
        sessions: list[Session] = []
        for path in self.root.glob("*.jsonl"):
            try:
                lines = [
                    line
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if not lines:
                    continue
                header = cast(JsonDict, json.loads(lines[0]))
                entries = [cast(JsonDict, json.loads(line)) for line in lines[1:]]
                sessions.append(
                    _state_from_parts(
                        session_id=cast(str, header["id"]),
                        title=cast(str | None, header.get("title")),
                        created_at=cast(str, header["created_at"]),
                        entries=entries,
                    ).session
                )
            except (OSError, KeyError, json.JSONDecodeError, pydantic.ValidationError):
                continue

        sessions.sort(key=lambda s: _parse_time(s.updated_at), reverse=True)
        return sessions

    async def get_session(self, session_id: str) -> Session | None:
        state = self._read_state(session_id)
        return state.session if state else None

    async def update_session_title(
        self,
        session_id: str,
        title: str,
    ) -> Session | None:
        async with self._lock(session_id):
            state = self._read_state(session_id)
            if state is None:
                return None

            entry = self._make_entry(state, "session_info", {"title": title})
            self._append_entry(session_id, entry)
            updated = self._read_state(session_id)
            return updated.session if updated else None

    async def delete_session(self, session_id: str) -> bool:
        async with self._lock(session_id):
            path = self._path(session_id)
            if not path.exists():
                return False
            path.unlink()
            return True

    async def load_messages(self, session_id: str) -> list[ai_messages.Message]:
        state = self._read_state(session_id)
        if state is None:
            return []
        return _messages_for_leaf(state.entries, state.leaf_id)

    def _make_entry(
        self,
        state: _SessionState,
        entry_type: str,
        extra: JsonDict,
    ) -> JsonDict:
        existing = {cast(str, entry["id"]) for entry in state.entries}
        entry: JsonDict = {
            "type": entry_type,
            "id": _new_entry_id(existing),
            "parent_id": state.leaf_id,
            "timestamp": _now(),
        }
        entry.update(extra)
        return entry

    async def save_messages(
        self,
        session_id: str,
        messages: list[ai_messages.Message],
        *,
        entry_type: str = "messages",
        metadata: JsonDict | None = None,
    ) -> None:
        if self._read_state(session_id) is None:
            await self.create_session(session_id)

        async with self._lock(session_id):
            state = self._read_state(session_id)
            assert state is not None

            current = _messages_for_leaf(state.entries, state.leaf_id)
            if _messages_equal(current, messages) and metadata is None:
                return

            extra: JsonDict = {"messages": _messages_to_json(messages)}
            if metadata is not None:
                extra.update(metadata)
            entry = self._make_entry(state, entry_type, extra)
            self._append_entry(session_id, entry)

    async def get_tree(self, session_id: str) -> JsonDict | None:
        state = self._read_state(session_id)
        if state is None:
            return None
        labels = _labels_by_id(state.entries)
        return {
            "session": state.session.model_dump(),
            "leaf_id": state.leaf_id,
            "entries": [_entry_summary(entry, labels) for entry in state.entries],
        }

    async def move_leaf(self, session_id: str, entry_id: str | None) -> bool:
        async with self._lock(session_id):
            state = self._read_state(session_id)
            if state is None:
                return False
            if entry_id is not None and all(e["id"] != entry_id for e in state.entries):
                return False
            entry = self._make_entry(state, "leaf", {"target_id": entry_id})
            self._append_entry(session_id, entry)
            return True

    async def label_entry(
        self,
        session_id: str,
        entry_id: str,
        label: str | None,
    ) -> bool:
        async with self._lock(session_id):
            state = self._read_state(session_id)
            if state is None or all(e["id"] != entry_id for e in state.entries):
                return False
            entry = self._make_entry(
                state,
                "label",
                {"target_id": entry_id, "label": label},
            )
            self._append_entry(session_id, entry)
            return True

    async def fork_session(
        self,
        source_session_id: str,
        new_session_id: str,
        entry_id: str | None = None,
    ) -> Session | None:
        source = self._read_state(source_session_id)
        if source is None:
            return None

        leaf_id = entry_id if entry_id is not None else source.leaf_id
        if leaf_id is not None and all(e["id"] != leaf_id for e in source.entries):
            return None

        messages = _messages_for_leaf(source.entries, leaf_id)
        session = await self.create_session(
            new_session_id,
            source.session.title,
            parent_id=source_session_id,
        )
        await self.save_messages(new_session_id, messages)
        return session


class PostgresSessionStore(SessionStore):
    """Append-only Postgres session store for deployed environments."""

    async def ensure_schema(self) -> None:
        pool = await db.get_pool()
        await pool.execute(_POSTGRES_SCHEMA)

    def _row_to_session(self, row: asyncpg.Record) -> Session:
        return Session(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"].isoformat(),
            updated_at=row["updated_at"].isoformat(),
        )

    def _parse_jsonb(self, value: Any) -> JsonDict:
        if isinstance(value, str):
            return cast(JsonDict, json.loads(value))
        return cast(JsonDict, value)

    async def create_session(
        self,
        session_id: str,
        title: str | None = None,
        parent_id: str | None = None,
    ) -> Session:
        pool = await db.get_pool()
        row = await pool.fetchrow(
            "INSERT INTO sessions (id, title, cwd, parent_id) "
            "VALUES ($1, $2, $3, $4) "
            "ON CONFLICT (id) DO UPDATE SET id = EXCLUDED.id "
            "RETURNING id, title, created_at, updated_at",
            session_id,
            title,
            os.getcwd(),
            parent_id,
        )
        return self._row_to_session(row)

    async def list_sessions(self) -> list[Session]:
        pool = await db.get_pool()
        rows = await pool.fetch(
            "SELECT id, title, created_at, updated_at "
            "FROM sessions ORDER BY updated_at DESC"
        )
        return [self._row_to_session(r) for r in rows]

    async def get_session(self, session_id: str) -> Session | None:
        pool = await db.get_pool()
        row = await pool.fetchrow(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = $1",
            session_id,
        )
        return self._row_to_session(row) if row else None

    async def update_session_title(
        self,
        session_id: str,
        title: str,
    ) -> Session | None:
        pool = await db.get_pool()
        async with pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT leaf_id FROM sessions WHERE id = $1 FOR UPDATE",
                session_id,
            )
            if row is None:
                return None
            await self._append_entry_conn(
                conn,
                session_id,
                "session_info",
                {"title": title},
                cast(str | None, row["leaf_id"]),
            )
            updated = await conn.fetchrow(
                "UPDATE sessions SET title = $2, updated_at = now() WHERE id = $1 "
                "RETURNING id, title, created_at, updated_at",
                session_id,
                title,
            )
            return self._row_to_session(updated)

    async def delete_session(self, session_id: str) -> bool:
        pool = await db.get_pool()
        result = await pool.execute("DELETE FROM sessions WHERE id = $1", session_id)
        return bool(result == "DELETE 1")

    async def _fetch_entries(self, session_id: str) -> list[JsonDict]:
        pool = await db.get_pool()
        rows = await pool.fetch(
            "SELECT entry_json FROM session_entries WHERE session_id = $1 ORDER BY seq",
            session_id,
        )
        return [self._parse_jsonb(r["entry_json"]) for r in rows]

    async def _fetch_state(self, session_id: str) -> _SessionState | None:
        pool = await db.get_pool()
        row = await pool.fetchrow(
            "SELECT id, title, created_at, updated_at, leaf_id "
            "FROM sessions WHERE id = $1",
            session_id,
        )
        if row is None:
            return None
        entries = await self._fetch_entries(session_id)
        state = _state_from_parts(
            session_id=row["id"],
            title=row["title"],
            created_at=row["created_at"].isoformat(),
            entries=entries,
            leaf_id=cast(str | None, row["leaf_id"]),
        )
        return state._replace(
            session=state.session.model_copy(
                update={"updated_at": row["updated_at"].isoformat()}
            )
        )

    async def load_messages(self, session_id: str) -> list[ai_messages.Message]:
        state = await self._fetch_state(session_id)
        if state is None:
            return []
        return _messages_for_leaf(state.entries, state.leaf_id)

    async def _append_entry_conn(
        self,
        conn: Any,
        session_id: str,
        entry_type: str,
        extra: JsonDict,
        parent_id: str | None,
    ) -> JsonDict:
        rows = await conn.fetch(
            "SELECT entry_id FROM session_entries WHERE session_id = $1",
            session_id,
        )
        existing = {cast(str, r["entry_id"]) for r in rows}
        entry_id = _new_entry_id(existing)
        timestamp = _now()
        entry: JsonDict = {
            "type": entry_type,
            "id": entry_id,
            "parent_id": parent_id,
            "timestamp": timestamp,
            **extra,
        }
        seq = await conn.fetchval(
            "SELECT COALESCE(MAX(seq) + 1, 0) "
            "FROM session_entries WHERE session_id = $1",
            session_id,
        )
        await conn.execute(
            "INSERT INTO session_entries "
            "(session_id, entry_id, seq, parent_id, type, timestamp, entry_json) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)",
            session_id,
            entry_id,
            seq,
            parent_id,
            entry_type,
            datetime.fromisoformat(timestamp),
            json.dumps(entry),
        )
        leaf_id = _entry_leaf_after(entry)
        await conn.execute(
            "UPDATE sessions SET leaf_id = $2, updated_at = $3 WHERE id = $1",
            session_id,
            leaf_id,
            datetime.fromisoformat(timestamp),
        )
        return entry

    async def save_messages(
        self,
        session_id: str,
        messages: list[ai_messages.Message],
        *,
        entry_type: str = "messages",
        metadata: JsonDict | None = None,
    ) -> None:
        await self.create_session(session_id)
        pool = await db.get_pool()
        async with pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT leaf_id FROM sessions WHERE id = $1 FOR UPDATE",
                session_id,
            )
            if row is None:
                return
            entries = [
                self._parse_jsonb(r["entry_json"])
                for r in await conn.fetch(
                    "SELECT entry_json FROM session_entries "
                    "WHERE session_id = $1 ORDER BY seq",
                    session_id,
                )
            ]
            parent_id = cast(str | None, row["leaf_id"])
            current = _messages_for_leaf(entries, parent_id)
            if _messages_equal(current, messages) and metadata is None:
                return
            extra: JsonDict = {"messages": _messages_to_json(messages)}
            if metadata is not None:
                extra.update(metadata)
            await self._append_entry_conn(
                conn,
                session_id,
                entry_type,
                extra,
                parent_id,
            )

    async def get_tree(self, session_id: str) -> JsonDict | None:
        state = await self._fetch_state(session_id)
        if state is None:
            return None
        labels = _labels_by_id(state.entries)
        return {
            "session": state.session.model_dump(),
            "leaf_id": state.leaf_id,
            "entries": [_entry_summary(entry, labels) for entry in state.entries],
        }

    async def move_leaf(self, session_id: str, entry_id: str | None) -> bool:
        pool = await db.get_pool()
        async with pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT leaf_id FROM sessions WHERE id = $1 FOR UPDATE",
                session_id,
            )
            if row is None:
                return False
            if entry_id is not None:
                exists = await conn.fetchval(
                    "SELECT 1 FROM session_entries "
                    "WHERE session_id = $1 AND entry_id = $2",
                    session_id,
                    entry_id,
                )
                if exists is None:
                    return False
            await self._append_entry_conn(
                conn,
                session_id,
                "leaf",
                {"target_id": entry_id},
                cast(str | None, row["leaf_id"]),
            )
            return True

    async def label_entry(
        self,
        session_id: str,
        entry_id: str,
        label: str | None,
    ) -> bool:
        pool = await db.get_pool()
        async with pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT leaf_id FROM sessions WHERE id = $1 FOR UPDATE",
                session_id,
            )
            if row is None:
                return False
            exists = await conn.fetchval(
                "SELECT 1 FROM session_entries WHERE session_id = $1 AND entry_id = $2",
                session_id,
                entry_id,
            )
            if exists is None:
                return False
            await self._append_entry_conn(
                conn,
                session_id,
                "label",
                {"target_id": entry_id, "label": label},
                cast(str | None, row["leaf_id"]),
            )
            return True

    async def fork_session(
        self,
        source_session_id: str,
        new_session_id: str,
        entry_id: str | None = None,
    ) -> Session | None:
        source = await self._fetch_state(source_session_id)
        if source is None:
            return None

        leaf_id = entry_id if entry_id is not None else source.leaf_id
        if leaf_id is not None and all(e["id"] != leaf_id for e in source.entries):
            return None

        messages = _messages_for_leaf(source.entries, leaf_id)
        session = await self.create_session(
            new_session_id,
            source.session.title,
            parent_id=source_session_id,
        )
        await self.save_messages(new_session_id, messages)
        return session


_store: SessionStore | None = None


def _jsonl_root() -> Path:
    configured = os.environ.get("SEAL_SESSIONS_DIR")
    return Path(configured) if configured else _DEFAULT_SESSIONS_DIR


def get_store() -> SessionStore:
    """Return the configured session store."""
    global _store
    if _store is None:
        _store = (
            PostgresSessionStore()
            if os.environ.get("DATABASE_URL")
            else JsonlSessionStore(_jsonl_root())
        )
    return _store


def set_store_for_tests(store: SessionStore | None) -> None:
    """Override the global store in tests."""
    global _store
    _store = store


async def ensure_schema() -> None:
    await get_store().ensure_schema()


async def create_session(session_id: str, title: str | None = None) -> Session:
    return await get_store().create_session(session_id, title)


async def get_or_create_session(session_id: str) -> Session:
    return await create_session(session_id)


async def list_sessions() -> list[Session]:
    return await get_store().list_sessions()


async def get_session(session_id: str) -> Session | None:
    return await get_store().get_session(session_id)


async def update_session_title(session_id: str, title: str) -> Session | None:
    return await get_store().update_session_title(session_id, title)


async def delete_session(session_id: str) -> bool:
    return await get_store().delete_session(session_id)


async def touch_session(_session_id: str) -> None:
    """Compatibility hook; appending messages already updates activity."""


async def load_ai_messages(session_id: str) -> list[ai_messages.Message]:
    return await get_store().load_messages(session_id)


async def persist_ai_messages(
    session_id: str,
    messages: list[ai_messages.Message],
) -> None:
    await get_store().save_messages(session_id, messages)


async def get_tree(session_id: str) -> JsonDict | None:
    return await get_store().get_tree(session_id)


async def move_leaf(session_id: str, entry_id: str | None) -> bool:
    return await get_store().move_leaf(session_id, entry_id)


async def label_entry(
    session_id: str,
    entry_id: str,
    label: str | None,
) -> bool:
    return await get_store().label_entry(session_id, entry_id, label)


async def fork_session(
    source_session_id: str,
    new_session_id: str,
    entry_id: str | None = None,
) -> Session | None:
    return await get_store().fork_session(source_session_id, new_session_id, entry_id)


async def clone_session(source_session_id: str, new_session_id: str) -> Session | None:
    return await fork_session(source_session_id, new_session_id)


async def compact_session(
    session_id: str,
    summary: str,
    keep_last: int = 8,
) -> bool:
    messages = await load_ai_messages(session_id)
    if not messages:
        return False

    kept = messages[-keep_last:] if keep_last > 0 else []
    summary_message = ai_messages.Message(
        role="user",
        parts=[
            ai_messages.TextPart(
                text="Previous conversation summary:\n\n" + summary.strip()
            )
        ],
    )
    await get_store().save_messages(
        session_id,
        [summary_message, *kept],
        entry_type="compaction",
        metadata={
            "summary": summary.strip(),
            "tokens_before": sum(len(m.text) for m in messages) // 4,
        },
    )
    return True


def prepare_chat_request(
    *,
    request_messages: list[UIMessage],
    stored_messages: list[ai_messages.Message],
) -> PreparedChat:
    """Normalize a useChat request without raw UI-shape history validation."""
    request_ai_messages, approvals = to_messages(request_messages)
    if not request_ai_messages:
        return PreparedChat(stored_messages, approvals, bool(approvals), False)

    if not stored_messages:
        return PreparedChat(request_ai_messages, approvals, True, True)

    latest = request_ai_messages[-1]
    if latest.role == "user":
        stored_by_id = {m.id: m for m in stored_messages}
        stored_latest = stored_by_id.get(latest.id)
        if stored_latest is None:
            return PreparedChat([*stored_messages, latest], approvals, True, True)
        if _message_to_json(stored_latest) != _message_to_json(latest):
            return PreparedChat(request_ai_messages, approvals, True, True)

    if approvals:
        return PreparedChat(stored_messages, approvals, True, False)

    return PreparedChat(stored_messages, approvals, False, False)


def first_user_text(messages: list[ai_messages.Message]) -> str | None:
    """Return the text of the first user message, or None."""
    for msg in messages:
        if msg.role == "user" and msg.text:
            return msg.text
    return None


def _tool_call_id_for_hook(hook: ai_messages.HookPart[Any]) -> str | None:
    if hook.hook_type != TOOL_APPROVAL_HOOK_TYPE:
        return None
    prefix = "approve_"
    if not hook.hook_id.startswith(prefix):
        return None
    return hook.hook_id[len(prefix) :]


def _turn_id_for_tool_call(
    messages: list[ai_messages.Message],
    tool_call_id: str,
) -> str | None:
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        if any(part.tool_call_id == tool_call_id for part in message.tool_calls):
            return message.turn_id or message.id
    return None


def with_pending_hook_messages(
    messages: list[ai_messages.Message],
    pending_hook_messages: list[ai_messages.Message],
) -> list[ai_messages.Message]:
    """Attach pending internal hook messages to the persisted snapshot."""
    result = list(messages)
    for message in pending_hook_messages:
        if len(message.parts) != 1 or not isinstance(
            message.parts[0], ai_messages.HookPart
        ):
            result.append(message)
            continue

        tool_call_id = _tool_call_id_for_hook(message.parts[0])
        turn_id = (
            _turn_id_for_tool_call(messages, tool_call_id)
            if tool_call_id is not None
            else None
        )
        result.append(message.model_copy(update={"turn_id": turn_id}))
    return result
