# management of the tree log
# uses entries.py entries arranged in a tree-like structure to store events
# separately, provides a user-facing interface with create, append, fork, branch, 
# compact, update, etc.
# delegates actual storage to storage.py abstract storage

from __future__ import annotations

import ai
import pydantic

from .entries import (
    CustomEntry,
    Entry,
    LeafEntry,
    MessageEntry,
    ModelSettings,
    ModelSettingsEntry,
    SessionInfoEntry,
)


class SessionTreeError(Exception):
    pass


class SessionTreeLog(pydantic.BaseModel):
    entries: list[Entry] = pydantic.Field(default_factory=list)


class BuiltSession(pydantic.BaseModel):
    title: str | None = None
    model_settings: ModelSettings = pydantic.Field(default_factory=ModelSettings)
    messages: list[ai.messages.Message] = pydantic.Field(default_factory=list)
    leaf_id: str
    path_ids: list[str] = pydantic.Field(default_factory=list)


def build_session(tree: SessionTreeLog, leaf_id: str) -> BuiltSession:
    by_id = {entry.id: entry for entry in tree.entries}
    path: list[Entry] = []
    seen: set[str] = set()

    current = by_id.get(leaf_id)
    if current is None:
        raise SessionTreeError(f"Entry {leaf_id} not found")

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

    title: str | None = None
    model_settings = ModelSettings()
    messages: list[ai.messages.Message] = []

    for entry in reversed(path):
        if isinstance(entry, MessageEntry):
            messages.append(entry.message)
        elif isinstance(entry, ModelSettingsEntry):
            model_settings = entry.settings
        elif isinstance(entry, SessionInfoEntry):
            title = entry.title
        elif isinstance(entry, CustomEntry | LeafEntry):
            continue

    return BuiltSession(
        title=title,
        model_settings=model_settings,
        messages=messages,
        leaf_id=leaf_id,
        path_ids=[entry.id for entry in reversed(path)],
    )
