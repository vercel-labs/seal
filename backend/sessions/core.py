"""Per-session log abstraction and the user-facing ``Session`` class.

``SessionLog`` is the storage-facing boundary (one per session): it holds the
ordered entries plus in-memory indices (``_by_id``, ``_labels``, ``_leaf_id``)
and exposes a single mutation point: ``append``.  Subclasses implement
``_persist`` for their backing store.

``Session`` is the user-facing API. It wraps a ``SessionLog`` and provides the
high-level operations (``load_messages``, ``save_messages``, ``move_to``,
``label``, ``set_title``, ``compact``, ``build_tree``).  All read-then-write
operations are serialized through the per-session ``asyncio.Lock`` carried by
the log.

``SessionRepo`` is the collection-facing boundary: ``create``, ``open``,
``list``, ``delete``, ``fork``.
"""

from __future__ import annotations

import abc
import asyncio
import json
from typing import Any, Literal

import pydantic
from ai import messages as ai_messages

from .entries import (
    BranchSummaryEntry,
    CompactionEntry,
    LabelEntry,
    LeafEntry,
    MessageEntry,
    SessionEntry,
    SessionInfoEntry,
    leaf_after,
    new_entry_id,
    now_iso,
)

SessionErrorCode = Literal[
    "not_found",
    "invalid_session",
    "invalid_entry",
    "invalid_fork_target",
    "storage",
]


class SessionError(Exception):
    """Raised by session storage, logs, and high-level operations."""

    def __init__(
        self,
        code: SessionErrorCode,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        if cause is not None:
            self.__cause__ = cause


class SessionMeta(pydantic.BaseModel):
    """Serializable session header."""

    id: str
    title: str | None
    created_at: str
    updated_at: str


ForkPosition = Literal["at", "before"]


def _msg_signature(message: ai_messages.Message) -> str:
    return json.dumps(
        message.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )


def _entry_summary(entry: SessionEntry, labels: dict[str, str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "id": entry.id,
        "type": entry.type,
        "parent_id": entry.parent_id,
        "timestamp": entry.timestamp,
        "label": labels.get(entry.id),
    }
    if isinstance(entry, MessageEntry):
        msg = entry.message
        if msg.role in ("user", "assistant"):
            summary["role"] = msg.role
            summary["text"] = msg.text
    elif isinstance(entry, SessionInfoEntry):
        summary["title"] = entry.title
    elif isinstance(entry, BranchSummaryEntry):
        summary["summary"] = entry.summary
        summary["from_id"] = entry.from_id
    elif isinstance(entry, CompactionEntry):
        summary["summary"] = entry.summary
        summary["tokens_before"] = entry.tokens_before
        summary["first_kept_entry_id"] = entry.first_kept_entry_id
    elif isinstance(entry, LabelEntry | LeafEntry):
        summary["target_id"] = entry.target_id
    return summary


class SessionLog(abc.ABC):
    """Append-only log for one session.

    Subclasses persist entries via :meth:`_persist`; ``append`` then updates
    in-memory state. ``_persist`` and ``append`` assume the caller already
    holds :attr:`lock`. ``Session`` methods acquire the lock once around any
    read-then-write sequence.
    """

    lock: asyncio.Lock
    _meta: SessionMeta
    _entries: list[SessionEntry]
    _by_id: dict[str, SessionEntry]
    _labels: dict[str, str]
    _leaf_id: str | None
    _parent_id: str | None

    def __init__(
        self,
        *,
        meta: SessionMeta,
        entries: list[SessionEntry],
        leaf_id: str | None,
        parent_id: str | None,
        lock: asyncio.Lock,
    ) -> None:
        self.lock = lock
        self._meta = meta
        self._entries = list(entries)
        self._by_id = {e.id: e for e in self._entries}
        self._labels = _build_labels(self._entries)
        self._leaf_id = leaf_id
        self._parent_id = parent_id

    @property
    def id(self) -> str:
        return self._meta.id

    @property
    def parent_id(self) -> str | None:
        return self._parent_id

    def leaf_id(self) -> str | None:
        return self._leaf_id

    def entries(self) -> list[SessionEntry]:
        return list(self._entries)

    def by_id(self, entry_id: str) -> SessionEntry | None:
        return self._by_id.get(entry_id)

    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    def meta(self) -> SessionMeta:
        title = self._meta.title
        updated_at = self._meta.created_at
        for entry in self._entries:
            updated_at = entry.timestamp
            if isinstance(entry, SessionInfoEntry):
                title = entry.title
        return SessionMeta(
            id=self._meta.id,
            title=title,
            created_at=self._meta.created_at,
            updated_at=updated_at,
        )

    def make_entry_id(self) -> str:
        return new_entry_id(set(self._by_id.keys()))

    def path_to_root(self, leaf_id: str | None) -> list[SessionEntry]:
        if leaf_id is None:
            return []
        path: list[SessionEntry] = []
        current = self._by_id.get(leaf_id)
        if current is None:
            raise SessionError("not_found", f"Entry {leaf_id} not found")
        seen: set[str] = set()
        while current is not None:
            if current.id in seen:
                raise SessionError(
                    "invalid_session", f"Cycle at entry {current.id}"
                )
            seen.add(current.id)
            path.insert(0, current)
            if current.parent_id is None:
                break
            parent = self._by_id.get(current.parent_id)
            if parent is None:
                raise SessionError(
                    "invalid_session",
                    f"Parent entry {current.parent_id} not found",
                )
            current = parent
        return path

    async def append(self, entry: SessionEntry) -> None:
        """Persist ``entry`` and update in-memory state.

        Caller must hold :attr:`lock`.
        """
        await self._persist(entry)
        self._install(entry)

    def _install(self, entry: SessionEntry) -> None:
        self._entries.append(entry)
        self._by_id[entry.id] = entry
        if isinstance(entry, LabelEntry):
            label = entry.label.strip() if entry.label else ""
            if label:
                self._labels[entry.target_id] = label
            else:
                self._labels.pop(entry.target_id, None)
        self._leaf_id = leaf_after(entry)

    @abc.abstractmethod
    async def _persist(self, entry: SessionEntry) -> None:
        """Persist ``entry`` to backing storage. Caller holds :attr:`lock`."""


def _build_labels(entries: list[SessionEntry]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, LabelEntry):
            continue
        label = entry.label.strip() if entry.label else ""
        if label:
            labels[entry.target_id] = label
        else:
            labels.pop(entry.target_id, None)
    return labels


class Session:
    """High-level chat-session operations backed by a ``SessionLog``."""

    def __init__(self, log: SessionLog) -> None:
        self._log = log

    @property
    def log(self) -> SessionLog:
        return self._log

    @property
    def id(self) -> str:
        return self._log.id

    @property
    def meta(self) -> SessionMeta:
        return self._log.meta()

    def path(self) -> list[SessionEntry]:
        return self._log.path_to_root(self._log.leaf_id())

    def entries(self) -> list[SessionEntry]:
        return self._log.entries()

    def by_id(self, entry_id: str) -> SessionEntry | None:
        return self._log.by_id(entry_id)

    def labels(self) -> dict[str, str]:
        return self._log.labels()

    def build_tree(self) -> dict[str, Any]:
        labels = self._log.labels()
        return {
            "session": self.meta.model_dump(),
            "leaf_id": self._log.leaf_id(),
            "entries": [_entry_summary(e, labels) for e in self._log.entries()],
        }

    async def load_messages(self) -> list[ai_messages.Message]:
        """Return the active branch's messages, replaying any compaction."""
        path = self.path()
        comp_idx: int | None = None
        for i in range(len(path) - 1, -1, -1):
            if isinstance(path[i], CompactionEntry):
                comp_idx = i
                break
        if comp_idx is None:
            return [e.message for e in path if isinstance(e, MessageEntry)]

        comp = path[comp_idx]
        assert isinstance(comp, CompactionEntry)
        result: list[ai_messages.Message] = [
            ai_messages.Message(
                role="user",
                parts=[
                    ai_messages.TextPart(
                        text=f"Previous conversation summary:\n\n{comp.summary.strip()}"
                    )
                ],
            )
        ]
        found_first_kept = comp.first_kept_entry_id is None
        for i, entry in enumerate(path):
            if i == comp_idx:
                continue
            if i < comp_idx:
                if entry.id == comp.first_kept_entry_id:
                    found_first_kept = True
                if not found_first_kept:
                    continue
            if isinstance(entry, MessageEntry):
                result.append(entry.message)
        return result

    async def save_messages(self, messages: list[ai_messages.Message]) -> None:
        """Append ``messages`` to the active branch.

        Incoming messages must extend the current branch. If they diverge at
        message ``N``, the leaf rewinds to the parent of message ``N`` (an
        implicit fork) before the new tail is appended. ``system`` messages
        are dropped.
        """
        incoming = [m for m in messages if m.role != "system"]
        async with self._log.lock:
            await self._save_messages_locked(incoming)

    async def _save_messages_locked(
        self, incoming: list[ai_messages.Message]
    ) -> None:
        path = self._log.path_to_root(self._log.leaf_id())
        path_messages = [e for e in path if isinstance(e, MessageEntry)]
        n_common = 0
        for prev, new in zip(path_messages, incoming, strict=False):
            if prev.message.id != new.id:
                break
            if _msg_signature(prev.message) != _msg_signature(new):
                break
            n_common += 1
        if n_common < len(path_messages):
            target = path_messages[n_common - 1].id if n_common > 0 else None
            await self._append_leaf_locked(target)
        for msg in incoming[n_common:]:
            await self._append_message_locked(msg)

    async def set_title(self, title: str) -> None:
        async with self._log.lock:
            entry = SessionInfoEntry(
                id=self._log.make_entry_id(),
                parent_id=self._log.leaf_id(),
                timestamp=now_iso(),
                title=title,
            )
            await self._log.append(entry)

    async def move_to(
        self,
        entry_id: str | None,
        *,
        summary: str | None = None,
    ) -> None:
        async with self._log.lock:
            if entry_id is not None and self._log.by_id(entry_id) is None:
                raise SessionError("not_found", f"Entry {entry_id} not found")
            await self._append_leaf_locked(entry_id)
            if summary is not None:
                bs = BranchSummaryEntry(
                    id=self._log.make_entry_id(),
                    parent_id=self._log.leaf_id(),
                    timestamp=now_iso(),
                    from_id=entry_id,
                    summary=summary,
                )
                await self._log.append(bs)

    async def label(self, entry_id: str, label: str | None) -> None:
        async with self._log.lock:
            if self._log.by_id(entry_id) is None:
                raise SessionError("not_found", f"Entry {entry_id} not found")
            entry = LabelEntry(
                id=self._log.make_entry_id(),
                parent_id=self._log.leaf_id(),
                timestamp=now_iso(),
                target_id=entry_id,
                label=label,
            )
            await self._log.append(entry)

    async def compact(
        self,
        summary: str,
        *,
        first_kept_entry_id: str | None,
        tokens_before: int,
    ) -> None:
        async with self._log.lock:
            if (
                first_kept_entry_id is not None
                and self._log.by_id(first_kept_entry_id) is None
            ):
                raise SessionError(
                    "not_found", f"Entry {first_kept_entry_id} not found"
                )
            entry = CompactionEntry(
                id=self._log.make_entry_id(),
                parent_id=self._log.leaf_id(),
                timestamp=now_iso(),
                summary=summary,
                first_kept_entry_id=first_kept_entry_id,
                tokens_before=tokens_before,
            )
            await self._log.append(entry)

    async def _append_message_locked(self, message: ai_messages.Message) -> None:
        entry = MessageEntry(
            id=self._log.make_entry_id(),
            parent_id=self._log.leaf_id(),
            timestamp=now_iso(),
            message=message,
        )
        await self._log.append(entry)

    async def _append_leaf_locked(self, target_id: str | None) -> None:
        if target_id is not None and self._log.by_id(target_id) is None:
            raise SessionError("not_found", f"Entry {target_id} not found")
        entry = LeafEntry(
            id=self._log.make_entry_id(),
            parent_id=self._log.leaf_id(),
            timestamp=now_iso(),
            target_id=target_id,
        )
        await self._log.append(entry)


class SessionRepo(abc.ABC):
    """Collection of sessions backed by JSONL files or Postgres."""

    @abc.abstractmethod
    async def ensure_schema(self) -> None: ...

    @abc.abstractmethod
    async def create(
        self,
        session_id: str,
        *,
        title: str | None = None,
        parent_id: str | None = None,
    ) -> Session: ...

    @abc.abstractmethod
    async def open(self, session_id: str) -> Session | None: ...

    @abc.abstractmethod
    async def list(self) -> list[SessionMeta]: ...

    @abc.abstractmethod
    async def delete(self, session_id: str) -> bool: ...

    @abc.abstractmethod
    async def fork(
        self,
        source_id: str,
        new_id: str,
        *,
        entry_id: str | None = None,
        position: ForkPosition = "at",
    ) -> Session: ...


async def entries_to_replay(
    log: SessionLog,
    *,
    entry_id: str | None,
    position: ForkPosition,
) -> list[SessionEntry]:
    """Pick the path of entries to copy when forking.

    ``entry_id=None`` copies the source's full active path.  ``position="at"``
    copies the path up to and including ``entry_id``; ``position="before"``
    requires ``entry_id`` to be a user-message entry and copies the path up to
    (but excluding) it, so a new turn can replace it.
    """
    if entry_id is None:
        return log.path_to_root(log.leaf_id())
    target = log.by_id(entry_id)
    if target is None:
        raise SessionError(
            "invalid_fork_target", f"Entry {entry_id} not found"
        )
    if position == "at":
        return log.path_to_root(entry_id)
    if not isinstance(target, MessageEntry) or target.message.role != "user":
        raise SessionError(
            "invalid_fork_target",
            f"Entry {entry_id} is not a user message",
        )
    return log.path_to_root(target.parent_id)
