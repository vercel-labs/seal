# management of the tree log
# uses entries.py entries arranged in a tree-like structure to store events
# separately, provides a user-facing interface with create, append, fork, branch, 
# compact, update, etc.
# delegates actual storage to storage.py abstract storage

from __future__ import annotations

from typing import Any

import ai

from .entries import (
    CustomEntry,
    Entry,
    LeafEntry,
    MessageEntry,
    ModelSettings,
    ModelSettingsEntry,
    SessionInfoEntry,
    new_entry_id,
    new_session_id,
    now_iso,
)
from .storage import Storage


class SessionTreeError(Exception):
    pass


class SessionTree:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def entries(self, session_id: str) -> list[Entry]:
        return self.storage.read(session_id)

    def append(self, session_id: str, entry: Entry) -> None:
        entries = self.storage.read(session_id)
        by_id = self._by_id(entries)

        if entry.id in by_id:
            raise SessionTreeError(f"Entry {entry.id} already exists")
        if entry.parent_id is not None and entry.parent_id not in by_id:
            raise SessionTreeError(f"Parent entry {entry.parent_id} not found")

        match entry:
            case LeafEntry(target_id=target_id) if (
                target_id is not None and target_id not in by_id
            ):
                raise SessionTreeError(f"Entry {target_id} not found")
            case _:
                pass

        self.storage.append(session_id, entry)

    def leaf_id(self, entries: list[Entry]) -> str | None:
        by_id = self._by_id(entries)
        leaf_id: str | None = None
        for entry in entries:
            match entry:
                case LeafEntry(target_id=target_id):
                    if target_id is not None and target_id not in by_id:
                        raise SessionTreeError(f"Entry {target_id} not found")
                    leaf_id = target_id
                case _:
                    leaf_id = entry.id
        return leaf_id

    def path(self, entries: list[Entry], leaf_id: str) -> list[Entry]:
        by_id = self._by_id(entries)
        current = by_id.get(leaf_id)
        if current is None:
            raise SessionTreeError(f"Entry {leaf_id} not found")

        path: list[Entry] = []
        seen: set[str] = set()

        while current is not None:
            if current.id in seen:
                raise SessionTreeError(f"Cycle at entry {current.id}")
            seen.add(current.id)
            path.append(current)

            if current.parent_id is None:
                break
            parent = by_id.get(current.parent_id)
            if parent is None:
                raise SessionTreeError(f"Parent entry {current.parent_id} not found")
            current = parent

        return list(reversed(path))

    def _by_id(self, entries: list[Entry]) -> dict[str, Entry]:
        by_id: dict[str, Entry] = {}
        for entry in entries:
            if entry.id in by_id:
                raise SessionTreeError(f"Entry {entry.id} already exists")
            by_id[entry.id] = entry
        return by_id


class Session:
    def __init__(self, tree: SessionTree, id: str | None = None) -> None:
        self._tree = tree
        self.id = id or new_session_id()
        self._entries: list[Entry] = []
        self._path: list[Entry] = []
        self._leaf_id: str | None = None
        self._refresh()

    @property
    def title(self) -> str | None:
        title: str | None = None
        for entry in self._path:
            match entry:
                case SessionInfoEntry(title=value):
                    title = value
                case _:
                    pass
        return title

    @property
    def model_settings(self) -> ModelSettings:
        settings = ModelSettings()
        for entry in self._path:
            match entry:
                case ModelSettingsEntry(settings=value):
                    settings = value
                case _:
                    pass
        return settings.model_copy(deep=True)

    @property
    def messages(self) -> list[ai.messages.Message]:
        messages: list[ai.messages.Message] = []
        for entry in self._path:
            match entry:
                case MessageEntry(message=message):
                    messages.append(message)
                case _:
                    pass
        return messages

    def append_message(self, message: ai.messages.Message) -> None:
        self._append(
            MessageEntry(
                id=new_entry_id({entry.id for entry in self._entries}),
                parent_id=self._leaf_id,
                timestamp=now_iso(),
                message=message,
            )
        )

    def set_model_settings(self, settings: ModelSettings) -> None:
        self._append(
            ModelSettingsEntry(
                id=new_entry_id({entry.id for entry in self._entries}),
                parent_id=self._leaf_id,
                timestamp=now_iso(),
                settings=settings,
            )
        )

    def set_title(self, title: str | None) -> None:
        self._append(
            SessionInfoEntry(
                id=new_entry_id({entry.id for entry in self._entries}),
                parent_id=self._leaf_id,
                timestamp=now_iso(),
                title=title,
            )
        )

    def append_custom(self, custom_type: str, data: Any | None = None) -> None:
        self._append(
            CustomEntry(
                id=new_entry_id({entry.id for entry in self._entries}),
                parent_id=self._leaf_id,
                timestamp=now_iso(),
                custom_type=custom_type,
                data=data,
            )
        )

    def branch(self, entry_id: str) -> Session:
        if entry_id == self._leaf_id:
            return self

        path = self._tree.path(self._entries, entry_id)
        session = Session(self._tree)
        for entry in path:
            session._tree.append(session.id, entry.model_copy(deep=True))
        session._refresh()
        return session

    def _append(self, entry: Entry) -> None:
        self._tree.append(self.id, entry)
        self._refresh()

    def _refresh(self) -> None:
        self._entries = self._tree.entries(self.id)
        self._leaf_id = self._tree.leaf_id(self._entries)
        self._path = (
            self._tree.path(self._entries, self._leaf_id)
            if self._leaf_id is not None
            else []
        )


def get_session(session_id: str | None = None, *, storage: Storage) -> Session:
    return Session(SessionTree(storage), session_id)
