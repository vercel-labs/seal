
from __future__ import annotations

from typing import Protocol

from .entries import Entry


class Storage(Protocol):
    # abstract storage layer, can be backed by
    # postgres, json, in-memory
    # used by session tree as backend
    def append(self, session_id: str, entry: Entry) -> None: ...

    def read(self, session_id: str) -> list[Entry]: ...


class InMemoryStorage:
    def __init__(self) -> None:
        self._entries: dict[str, list[Entry]] = {}

    def append(self, session_id: str, entry: Entry) -> None:
        self._entries.setdefault(session_id, []).append(entry)

    def read(self, session_id: str) -> list[Entry]:
        return list(self._entries.get(session_id, []))
