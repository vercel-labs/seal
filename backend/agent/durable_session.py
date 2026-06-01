"""Durable session snapshots for the workflow-backed agent."""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import urllib.parse
from typing import Literal

import pydantic


class DurableSessionState(pydantic.BaseModel):
    version: Literal[1] = 1
    session_id: str
    stream_key: str
    output: str = ""


class DurableSessionSnapshot(pydantic.BaseModel):
    version: Literal[1] = 1
    messages: list[dict[str, object]]
    output: str = ""
    pending_runtime_actions: list[dict[str, object]] = pydantic.Field(
        default_factory=list
    )
    pending_tool_results: list[dict[str, object]] = pydantic.Field(default_factory=list)


_locks: dict[str, asyncio.Lock] = {}


async def read(state: DurableSessionState) -> DurableSessionSnapshot:
    path = _path(state.session_id)
    async with _lock_for(str(path)):
        if not path.exists():
            raise RuntimeError(f"session snapshot not found: {state.session_id}")
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not lines:
            raise RuntimeError(f"session snapshot is empty: {state.session_id}")
        return DurableSessionSnapshot.model_validate(json.loads(lines[-1]))


async def write(
    state: DurableSessionState,
    snapshot: DurableSessionSnapshot,
    *,
    reset: bool = False,
) -> DurableSessionState:
    path = _path(state.session_id)
    async with _lock_for(str(path)):
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if reset else "a"
        with path.open(mode, encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    snapshot.model_dump(mode="json"),
                    separators=(",", ":"),
                )
            )
            stream.write("\n")
            stream.flush()
    return state.model_copy(update={"output": snapshot.output})


def _path(session_id: str) -> pathlib.Path:
    default_root = pathlib.Path(__file__).resolve().parents[1] / ".sessions"
    configured_root = os.environ.get("SEAL_DURABLE_SESSIONS_DIR", str(default_root))
    return (
        pathlib.Path(configured_root)
        / f"{urllib.parse.quote(session_id, safe='')}.jsonl"
    )


def _lock_for(key: str) -> asyncio.Lock:
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    return lock
