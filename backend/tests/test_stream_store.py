from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core import stream_store


def test_local_stream_store_appends_and_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SEAL_STREAMS_DIR", str(tmp_path))

    async def run() -> None:
        await stream_store.ensure_schema()
        await stream_store.ensure_stream("s1")
        assert await stream_store.get_status("s1") == "idle"
        assert await stream_store.count_events("s1") == 0

        await stream_store.set_status("s1", "running")
        await stream_store.set_active_start_index("s1", 3)
        first = await stream_store.append_event("s1", {"kind": "stream_start"})
        second = await stream_store.append_event(
            "s1",
            {"kind": "text_delta", "chunk": "hi", "block_id": "text-1"},
        )

        assert first == 0
        assert second == 1
        assert await stream_store.get_status("s1") == "running"
        assert await stream_store.get_active_start_index("s1") == 3
        assert await stream_store.count_events("s1") == 2

        events = await stream_store.list_events("s1", 1)
        assert [event.index for event in events] == [1]
        assert events[0].data["chunk"] == "hi"

        await stream_store.save_tool_approval(
            "s1",
            approval_id="approve_call-1",
            tool_call_id="call-1",
            granted=True,
            reason=None,
        )
        approvals = await stream_store.list_tool_approvals(
            "s1",
            ["approve_call-1", "approve_call-2"],
        )
        assert approvals == {
            "approve_call-1": {
                "tool_call_id": "call-1",
                "granted": True,
                "reason": None,
            }
        }
        assert await stream_store.count_events("s1") == 2

        await stream_store.set_status("s1", "completed")
        assert await stream_store.get_status("s1") == "completed"

    asyncio.run(run())


def test_local_stream_store_tracks_ui_stream_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SEAL_STREAMS_DIR", str(tmp_path))

    async def run() -> None:
        await stream_store.create_ui_stream(
            "ui-1",
            session_id="s1",
            source_stream_id="s1",
            source_start_index=3,
            history_message_count=2,
        )

        stream = await stream_store.get_ui_stream("ui-1")
        assert stream is not None
        assert stream.status == "running"
        assert stream.source_stream_id == "s1"
        assert stream.source_start_index == 3
        assert stream.source_next_index == 3
        assert stream.history_message_count == 2

        await stream_store.append_ui_chunk("ui-1", 'data: {"type":"start"}\n\n')
        await stream_store.append_ui_chunk("ui-1", 'data: {"type":"finish"}\n\n')
        assert await stream_store.claim_ui_stream_source_index("ui-1", 3) is True
        assert await stream_store.claim_ui_stream_source_index("ui-1", 3) is False
        await stream_store.complete_ui_stream_source_index("ui-1", 3, 5)
        assert await stream_store.claim_ui_stream_source_index("ui-1", 3) is False

        chunks = await stream_store.list_ui_chunks("ui-1", 1)
        assert [chunk.index for chunk in chunks] == [1]
        assert chunks[0].chunk == 'data: {"type":"finish"}\n\n'
        assert await stream_store.count_ui_chunks("ui-1") == 2

        stream = await stream_store.get_ui_stream("ui-1")
        assert stream is not None
        assert stream.source_next_index == 5

    asyncio.run(run())
