"""Jsonl store semantics: the append-only log everything else rides on."""

from __future__ import annotations

import pytest

from agent import storage


async def test_append_returns_sequential_indices() -> None:
    store = storage.store()
    assert await store.append("s1", "default", {"n": 0}) == 0
    assert await store.append("s1", "default", {"n": 1}) == 1
    # other (stream, namespace) keys are independent logs
    assert await store.append("s1", "session", {"n": 0}) == 0
    assert await store.append("s2", "default", {"n": 0}) == 0


async def test_read_from_start_index() -> None:
    store = storage.store()
    for n in range(4):
        await store.append("s1", "default", {"n": n})
    records = await store.read("s1", "default", 2)
    assert records == [(2, {"n": 2}), (3, {"n": 3})]
    assert await store.read("s1", "default", 99) == []
    assert await store.read("missing", "default", 0) == []


async def test_info_reports_tail_and_closed() -> None:
    store = storage.store()
    assert await store.info("s1", "default") == (-1, False)
    await store.append("s1", "default", {"n": 0})
    await store.append("s1", "default", {"n": 1})
    assert await store.info("s1", "default") == (1, False)
    await store.close("s1", "default")
    assert await store.info("s1", "default") == (1, True)


async def test_append_after_close_raises() -> None:
    store = storage.store()
    await store.append("s1", "default", {"n": 0})
    await store.close("s1", "default")
    with pytest.raises(RuntimeError, match="closed"):
        await store.append("s1", "default", {"n": 1})
