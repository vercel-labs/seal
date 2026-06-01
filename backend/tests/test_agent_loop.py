from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import ai
import pytest

from agent import durable_stream, main, server, trigger


def test_agent_loop_dispatches_tool_calls_from_final_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assistant = ai.messages.Message(
        role="assistant",
        parts=[
            ai.messages.ToolCallPart(
                tool_call_id="call-1",
                tool_name="bash",
                tool_args='{"command":"pwd"}',
            )
        ],
    )
    final = ai.messages.Message(
        role="assistant",
        parts=[ai.messages.TextPart(text="done")],
    )
    llm_calls: list[list[dict[str, object]]] = []

    async def fake_stream_llm(
        stream_key: str,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]] | None = None,
        model_id: str = main.MODEL_ID,
    ) -> dict[str, object]:
        del stream_key, tool_schemas, model_id
        llm_calls.append(messages)
        if len(llm_calls) == 1:
            return assistant.model_dump(mode="json")
        return final.model_dump(mode="json")

    async def fake_bash(command: str, timeout: int | None = None) -> str:
        del timeout
        return f"ran {command}"

    monkeypatch.setattr(main, "stream_llm", fake_stream_llm)
    monkeypatch.setattr(main, "_bash", fake_bash)

    async def run() -> tuple[str, list[ai.events.AgentEvent]]:
        model = ai.Model("placeholder", provider=main._WorkflowModelProvider())
        agent = main.SealAgent()
        async with agent.run(
            model,
            [ai.user_message("run pwd")],
            params={"stream_key": "stream-1", "model_id": "test-model"},
        ) as stream:
            events = [event async for event in stream]
            return stream.output, events

    output, events = asyncio.run(run())

    assert output == "done"
    assert len(llm_calls) == 2
    assert llm_calls[1][-1]["role"] == "tool"
    parts = cast(list[dict[str, object]], llm_calls[1][-1]["parts"])
    assert parts[0]["result"] == "ran pwd"
    assert any(isinstance(event, ai.events.ToolCallResult) for event in events)


def test_session_turn_retries_until_hook_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleeps: list[float] = []

    class FakeTurn:
        def __init__(self, prompt: str | None = None, close: bool = False) -> None:
            del prompt, close

        async def resume(self, token: str) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError(f"Hook with token {token!r} not found")

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(cast(Any, server).agent_main, "SessionTurn", FakeTurn)
    monkeypatch.setattr(cast(Any, server).asyncio, "sleep", fake_sleep)

    async def run() -> server.TurnResponse:
        return await server.session_turn(
            "s1",
            server.TurnRequest(
                prompt="hello",
                continuation_token="seal-session:s1:1",
            ),
        )

    response = asyncio.run(run())

    assert attempts == 2
    assert sleeps == [0.05]
    assert response.continuation_token == "seal-session:s1:2"


def test_tail_events_streams_all_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEAL_DURABLE_STREAMS_DIR", str(tmp_path))

    async def run() -> list[str]:
        stream = durable_stream.get_writable("run-1", reset=True)
        await stream.write(durable_stream.StreamStart())
        await stream.write(
            durable_stream.StreamEvent(
                type="TextDelta",
                data={"chunk": "hi"},
            )
        )
        await stream.write(
            durable_stream.StreamEvent(
                type="ReasoningDelta",
                data={"chunk": "thinking"},
            )
        )
        await stream.write(durable_stream.StreamDone())
        return [chunk async for chunk in trigger._tail_events("run-1")]

    chunks = asyncio.run(run())
    payloads: list[dict[str, Any]] = []
    for chunk in chunks:
        data_line = next(
            line for line in chunk.splitlines() if line.startswith("data: ")
        )
        payloads.append(json.loads(data_line.removeprefix("data: ")))

    assert [chunk.splitlines()[0] for chunk in chunks] == [
        "event: start",
        "event: TextDelta",
        "event: ReasoningDelta",
        "event: done",
    ]
    assert payloads[2]["scope"] == "llm"
    assert payloads[2]["data"]["chunk"] == "thinking"
