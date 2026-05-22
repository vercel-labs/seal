"""JSONL session log and repo, used for local development."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import quote

import pydantic

from .core import (
    ForkPosition,
    Session,
    SessionError,
    SessionLog,
    SessionMeta,
    SessionRepo,
    entries_to_replay,
)
from .entries import (
    ENTRY_ADAPTER,
    SessionEntry,
    SessionInfoEntry,
    leaf_after,
    now_iso,
)

_VERSION = 2


class JsonlSessionLog(SessionLog):
    """One JSONL file per session; appends a line per entry."""

    def __init__(
        self,
        *,
        path: Path,
        meta: SessionMeta,
        parent_id: str | None,
        entries: list[SessionEntry],
        leaf_id: str | None,
        lock: asyncio.Lock,
    ) -> None:
        super().__init__(
            meta=meta,
            entries=entries,
            leaf_id=leaf_id,
            parent_id=parent_id,
            lock=lock,
        )
        self._path = path

    async def _persist(self, entry: SessionEntry) -> None:
        line = json.dumps(entry.model_dump(mode="json"), separators=(",", ":"))
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")


def _file_name(session_id: str) -> str:
    return f"{quote(session_id, safe='')}.jsonl"


def _write_header(
    path: Path,
    *,
    session_id: str,
    created_at: str,
    parent_id: str | None,
) -> None:
    header = {
        "type": "session",
        "version": _VERSION,
        "id": session_id,
        "created_at": created_at,
        "cwd": os.getcwd(),
        "parent_id": parent_id,
    }
    path.write_text(
        json.dumps(header, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _load_file(
    path: Path,
) -> tuple[SessionMeta, str | None, list[SessionEntry], str | None]:
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise SessionError("invalid_session", f"Empty session file {path}")

    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise SessionError(
            "invalid_session", f"Invalid header in {path}", cause=exc
        ) from exc
    if not isinstance(header, dict) or header.get("type") != "session":
        raise SessionError("invalid_session", f"Missing session header in {path}")
    if header.get("version") != _VERSION:
        raise SessionError(
            "invalid_session", f"Unsupported session version in {path}"
        )

    entries: list[SessionEntry] = []
    leaf_id: str | None = None
    for line_no, raw in enumerate(lines[1:], start=2):
        try:
            data = json.loads(raw)
            entry = ENTRY_ADAPTER.validate_python(data)
        except (json.JSONDecodeError, pydantic.ValidationError) as exc:
            raise SessionError(
                "invalid_entry",
                f"Invalid entry on line {line_no} of {path}",
                cause=exc,
            ) from exc
        entries.append(entry)
        leaf_id = leaf_after(entry)

    created_at = header["created_at"]
    meta = SessionMeta(
        id=header["id"],
        title=None,
        created_at=created_at,
        updated_at=created_at,
    )
    return meta, header.get("parent_id"), entries, leaf_id


def _scan_meta(path: Path) -> SessionMeta | None:
    """Read just enough of a JSONL file to build a ``SessionMeta``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        header = json.loads(lines[0])
        if header.get("type") != "session":
            return None
        session_id = header["id"]
        created_at = header["created_at"]
    except (json.JSONDecodeError, KeyError):
        return None

    title: str | None = None
    updated_at = created_at
    for raw in lines[1:]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        ts = data.get("timestamp")
        if isinstance(ts, str):
            updated_at = ts
        if data.get("type") == "session_info":
            value = data.get("title")
            title = value if isinstance(value, str) or value is None else None
    return SessionMeta(
        id=session_id, title=title, created_at=created_at, updated_at=updated_at
    )


class JsonlSessionRepo(SessionRepo):
    """Filesystem-backed repo: one ``.jsonl`` per session."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._locks: dict[str, asyncio.Lock] = {}
        self._cache: dict[str, JsonlSessionLog] = {}

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    def _path(self, session_id: str) -> Path:
        return self._root / _file_name(session_id)

    async def ensure_schema(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    async def create(
        self,
        session_id: str,
        *,
        title: str | None = None,
        parent_id: str | None = None,
    ) -> Session:
        await self.ensure_schema()
        lock = self._lock_for(session_id)
        async with lock:
            cached = self._cache.get(session_id)
            if cached is not None:
                return Session(cached)

            path = self._path(session_id)
            if path.exists():
                meta, header_parent, entries, leaf_id = _load_file(path)
                log = JsonlSessionLog(
                    path=path,
                    meta=meta,
                    parent_id=header_parent,
                    entries=entries,
                    leaf_id=leaf_id,
                    lock=lock,
                )
                self._cache[session_id] = log
                return Session(log)

            created_at = now_iso()
            _write_header(
                path,
                session_id=session_id,
                created_at=created_at,
                parent_id=parent_id,
            )
            meta = SessionMeta(
                id=session_id,
                title=None,
                created_at=created_at,
                updated_at=created_at,
            )
            log = JsonlSessionLog(
                path=path,
                meta=meta,
                parent_id=parent_id,
                entries=[],
                leaf_id=None,
                lock=lock,
            )
            self._cache[session_id] = log
            if title is not None:
                entry = SessionInfoEntry(
                    id=log.make_entry_id(),
                    parent_id=log.leaf_id(),
                    timestamp=now_iso(),
                    title=title,
                )
                await log.append(entry)
            return Session(log)

    async def open(self, session_id: str) -> Session | None:
        lock = self._lock_for(session_id)
        async with lock:
            cached = self._cache.get(session_id)
            if cached is not None:
                return Session(cached)
            path = self._path(session_id)
            if not path.exists():
                return None
            meta, header_parent, entries, leaf_id = _load_file(path)
            log = JsonlSessionLog(
                path=path,
                meta=meta,
                parent_id=header_parent,
                entries=entries,
                leaf_id=leaf_id,
                lock=lock,
            )
            self._cache[session_id] = log
            return Session(log)

    async def list(self) -> list[SessionMeta]:
        if not self._root.exists():
            return []
        metas: list[SessionMeta] = []
        for path in self._root.glob("*.jsonl"):
            meta = _scan_meta(path)
            if meta is not None:
                metas.append(meta)
        metas.sort(key=lambda m: m.updated_at, reverse=True)
        return metas

    async def delete(self, session_id: str) -> bool:
        lock = self._lock_for(session_id)
        async with lock:
            path = self._path(session_id)
            if not path.exists():
                self._cache.pop(session_id, None)
                return False
            path.unlink()
            self._cache.pop(session_id, None)
            return True

    async def fork(
        self,
        source_id: str,
        new_id: str,
        *,
        entry_id: str | None = None,
        position: ForkPosition = "at",
    ) -> Session:
        source = await self.open(source_id)
        if source is None:
            raise SessionError(
                "not_found", f"Source session {source_id} not found"
            )
        async with source.log.lock:
            replay = await entries_to_replay(
                source.log, entry_id=entry_id, position=position
            )

        new_session = await self.create(new_id, parent_id=source_id)
        async with new_session.log.lock:
            for entry in replay:
                await new_session.log.append(entry)
        return new_session
