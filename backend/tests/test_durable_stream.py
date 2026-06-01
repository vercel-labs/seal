from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent import durable_session, durable_stream


def test_durable_stream_replays_protocol_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEAL_DURABLE_STREAMS_DIR", str(tmp_path))

    async def run() -> None:
        stream = durable_stream.get_writable("run-1", reset=True)
        assert await stream.write(durable_stream.StreamStart()) == 0
        assert await stream.write(durable_stream.LLMStart()) == 1
        assert (
            await stream.write(
                durable_stream.StreamEvent(
                    type="TextDelta",
                    data={"chunk": "hi"},
                )
            )
            == 2
        )
        assert await stream.write(durable_stream.LLMEnd()) == 3
        assert await stream.write(durable_stream.StreamDone()) == 4

        records = [
            record
            async for record in durable_stream.get_readable(
                "run-1", poll_interval=0.001
            )
        ]
        replay = [
            record
            async for record in durable_stream.get_readable(
                "run-1",
                start_index=2,
                poll_interval=0.001,
            )
        ]

        assert [record.type for record in records] == [
            "start",
            "llm_start",
            "TextDelta",
            "llm_end",
            "done",
        ]
        assert [record.index for record in records] == [0, 1, 2, 3, 4]
        assert isinstance(records[2], durable_stream.StreamEvent)
        assert records[2].data == {"chunk": "hi"}
        assert [record.type for record in replay] == ["TextDelta", "llm_end", "done"]

    asyncio.run(run())


def test_durable_stream_resolves_negative_start_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEAL_DURABLE_STREAMS_DIR", str(tmp_path))

    async def run() -> list[str]:
        stream = durable_stream.get_writable("run-1", reset=True)
        await stream.write(durable_stream.StreamStart())
        await stream.write(durable_stream.LLMEnd())
        await stream.write(durable_stream.StreamDone())
        return [
            record.type
            async for record in durable_stream.get_readable(
                "run-1",
                start_index=-2,
                poll_interval=0.001,
            )
        ]

    assert asyncio.run(run()) == ["llm_end", "done"]


def test_durable_stream_reset_truncates_existing_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEAL_DURABLE_STREAMS_DIR", str(tmp_path))

    async def run() -> list[str]:
        first = durable_stream.get_writable("run-1", reset=True)
        await first.write(durable_stream.StreamStart())
        await first.write(durable_stream.StreamDone())

        second = durable_stream.get_writable("run-1", reset=True)
        await second.write(durable_stream.StreamDone())

        return [
            record.type
            async for record in durable_stream.get_readable(
                "run-1", poll_interval=0.001
            )
        ]

    assert asyncio.run(run()) == ["done"]


def test_durable_stream_tails_until_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEAL_DURABLE_STREAMS_DIR", str(tmp_path))

    async def run() -> list[str]:
        async def collect() -> list[str]:
            return [
                record.type
                async for record in durable_stream.get_readable(
                    "run-1",
                    poll_interval=0.001,
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.01)

        stream = durable_stream.get_writable("run-1", reset=True)
        await stream.write(durable_stream.StreamStart())
        await stream.write(durable_stream.StreamDone())

        return await task

    assert asyncio.run(run()) == ["start", "done"]


def test_durable_session_state_is_small_snapshot_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEAL_DURABLE_SESSIONS_DIR", str(tmp_path))

    async def run() -> tuple[durable_session.DurableSessionState, list[str]]:
        state = durable_session.DurableSessionState(
            session_id="session-1",
            stream_key="stream-1",
        )
        state = await durable_session.write(
            state,
            durable_session.DurableSessionSnapshot(
                messages=[{"role": "system", "parts": []}],
            ),
            reset=True,
        )
        state = await durable_session.write(
            state,
            durable_session.DurableSessionSnapshot(
                messages=[
                    {"role": "system", "parts": []},
                    {"role": "user", "parts": []},
                ],
                output="ok",
            ),
        )
        snapshot = await durable_session.read(state)
        return state, [message["role"] for message in snapshot.messages]

    state, roles = asyncio.run(run())

    assert state.model_dump(mode="json") == {
        "version": 1,
        "session_id": "session-1",
        "stream_key": "stream-1",
        "output": "ok",
    }
    assert roles == ["system", "user"]
