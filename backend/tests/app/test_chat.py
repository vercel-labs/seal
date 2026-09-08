"""The durable-stream → AI SDK UI bridge: where duplicate/missing-message
regressions live.

``active_run_start_index`` decides where a reload resumes (wrong answer =
duplicated assistant message in the UI), ``to_sse`` decides when a stream
terminates (wrong answer = hang or truncated turn), and ``bundle_to_wire``
is the single source of truth for the nested subagent shape.

All tests run against real workflow run streams (on the local world) and the
real ai SDK UI adapter; writes go through the world directly, as the step
writers do in the app.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import ai
import ai.types.events as events_
import ai.types.messages as messages_
import stream_codec
import vercel.workflow._internal.serialization as wf_serialization
import vercel.workflow._internal.streams as wf_streams
import vercel.workflow._internal.world as wf_world

from agent import proto, stream
from app import chat


async def _run_id(session_id: str) -> str:
    existing = await stream.session_run_id(session_id)
    if existing is not None:
        return existing

    world = wf_world.get_world()
    created = await world.events_create(
        None,
        wf_world.RunCreatedEventData.model_validate(
            {
                "deployment_id": "",
                "workflow_name": "test-chat",
                "input": wf_serialization.PayloadEncoder().encode([]),
            }
        ).into_event(),
    )
    assert created.run is not None
    run_id = created.run.run_id
    await world.events_create(
        run_id,
        wf_world.HookCreatedEventData(
            token=proto.turn_hook_token(session_id)
        ).into_event("hook_test_" + session_id.replace(":", "_")),
    )
    return run_id


async def _write_run(run_id: str, *events: proto.StreamEvent) -> None:
    world = wf_world.get_world()
    name = wf_streams.workflow_run_stream_id(run_id)
    for event in events:
        data = stream_codec.adapter.dump_python(event, mode="json")
        await world.streams_write(run_id, name, wf_streams.encode_value(data))


async def _write(session_id: str, *events: proto.StreamEvent) -> None:
    await _write_run(await _run_id(session_id), *events)


def _text_events(text: str, *, block: str = "b") -> list[proto.StreamEvent]:
    message = ai.messages.Message(
        role="assistant", parts=[messages_.TextPart(text=text)]
    )
    return [
        events_.StreamStart(),
        events_.TextStart(block_id=block),
        events_.TextDelta(block_id=block, chunk=text),
        events_.TextEnd(block_id=block),
        events_.StreamEnd(message=message),
    ]


# --- active_run_start_index -------------------------------------------------------


async def test_no_stream_means_nothing_to_resume() -> None:
    assert await chat.active_run_start_index("s1") is None


async def test_completed_run_is_not_resumable() -> None:
    await _write(
        "s1",
        stream.session_started(),
        stream.turn_started(turn_index=0),
        *_text_events("hi"),
        stream.session_waiting(turn_index=0),
    )
    assert await chat.active_run_start_index("s1") is None


async def test_in_flight_run_resumes_from_its_opener() -> None:
    await _write(
        "s1",
        stream.session_started(),  # 0
        stream.turn_started(turn_index=0),  # 1
        *_text_events("hi"),  # 2-6
        stream.session_waiting(turn_index=0),  # 7
        stream.turn_started(turn_index=1),  # 8 ← in-flight run opens here
        events_.StreamStart(),  # 9
    )
    assert await chat.active_run_start_index("s1") == 8


async def test_multi_turn_run_resumes_from_run_start_not_inner_turn() -> None:
    # a subagent round trip spans several driver turns but is ONE ui run; resuming
    # from the inner turn.started would replay a partial message and duplicate.
    await _write(
        "s1",
        stream.session_started(),  # 0 ← run opens here
        stream.turn_started(turn_index=0),  # 1
        *_text_events("delegating"),  # 2-6
        stream.subagent_called(
            tool_call_id="tc-1",
            child_session_id="s1:child:tc-1",
            child_run_id="wrun-test-child-tc-1",
            name="helper",
        ),  # 7
        stream.subagent_completed(tool_call_id="tc-1", is_error=False),  # 8
        stream.turn_started(turn_index=1),  # 9 (inner turn, same run)
        events_.StreamStart(),  # 10
    )
    assert await chat.active_run_start_index("s1") == 0


async def test_run_parked_on_approval_is_resumable_from_run_start() -> None:
    # a turn parked on an approval is mid-run, so a cold reload re-tails from the
    # opener to rebuild the same UI message (the live POST resume folds instead).
    await _write(
        "s1",
        stream.session_started(),
        stream.turn_started(turn_index=0),
        *_text_events("need approval"),
        stream.tool_approval_requested(),
    )
    assert await chat.active_run_start_index("s1") == 0


# --- bundle_to_wire ---------------------------------------------------------------


def test_bundle_to_wire_folds_all_assistant_turns_into_one_ui_message() -> None:
    transcript = [
        ai.system_message("you are a subagent"),
        ai.user_message("task"),
        ai.messages.Message(
            role="assistant",
            parts=[
                messages_.TextPart(text="first"),
                messages_.ToolCallPart(
                    tool_call_id="tc-1", tool_name="web_fetch", tool_args="{}"
                ),
            ],
        ),
        ai.tool_message(tool_call_id="tc-1", tool_name="web_fetch", result="page"),
        ai.messages.Message(
            role="assistant", parts=[messages_.TextPart(text="second")]
        ),
    ]
    nested = chat.bundle_to_wire(transcript)
    assert nested is not None
    assert nested["role"] == "assistant"
    parts = cast(list[dict[str, Any]], nested["parts"])
    texts = [part["text"] for part in parts if part.get("type") == "text"]
    assert texts == ["first", "second"]


def test_bundle_to_wire_with_no_assistant_message_yet() -> None:
    assert chat.bundle_to_wire([ai.user_message("task")]) is None


# --- to_sse ----------------------------------------------------------------------


async def _collect_sse(session_id: str, start_index: int = 0) -> list[str]:
    async def drain() -> list[str]:
        return [line async for line in chat.to_sse(session_id, start_index)]

    return await asyncio.wait_for(drain(), timeout=5)


def _sse_payloads(lines: list[str]) -> list[dict[str, Any]]:
    payloads = []
    for line in lines:
        body = line.removeprefix("data: ").strip()
        if body and body != "[DONE]":
            payloads.append(json.loads(body))
    return payloads


async def test_to_sse_streams_one_turn_and_terminates_at_waiting() -> None:
    # the durable stream stays OPEN (session still alive); termination must come
    # from the session.waiting boundary — getting this wrong is a client hang.
    await _write(
        "s1",
        stream.session_started(),
        stream.turn_started(turn_index=0),
        *_text_events("hello world"),
        stream.session_waiting(turn_index=0),
    )
    lines = await _collect_sse("s1")

    deltas = [
        payload
        for payload in _sse_payloads(lines)
        if payload.get("type") == "text-delta"
    ]
    assert [delta["delta"] for delta in deltas] == ["hello world"]
    assert lines[-1].startswith("data:")
    assert "[DONE]" in lines[-1]


async def test_to_sse_parks_at_a_deferred_approval() -> None:
    hook: messages_.HookPart[Any] = messages_.HookPart(
        hook_id="approve_tc-1",
        hook_type="ToolApproval",
        tool_call_id="tc-1",
        status="pending",
        metadata={"tool": "bash", "kwargs": {"command": "rm -rf /tmp/x"}},
    )
    tool_call = ai.messages.Message(
        role="assistant",
        parts=[
            messages_.ToolCallPart(
                tool_call_id="tc-1",
                tool_name="bash",
                tool_args='{"command": "rm -rf /tmp/x"}',
            )
        ],
    )
    await _write(
        "s1",
        stream.session_started(),
        stream.turn_started(turn_index=0),
        events_.StreamStart(),
        events_.ToolStart(tool_call_id="tc-1", tool_name="bash"),
        events_.ToolEnd(tool_call_id="tc-1", tool_call=tool_call.tool_calls[0]),
        events_.StreamEnd(message=tool_call),
        events_.HookEvent(
            message=ai.messages.Message(role="internal", parts=[hook]), hook=hook
        ),
        stream.tool_approval_requested(),
    )
    lines = await _collect_sse("s1")

    kinds = [payload.get("type") for payload in _sse_payloads(lines)]
    assert "tool-approval-request" in kinds
    assert "[DONE]" in lines[-1]


async def test_approval_resume_continuation_opens_id_less() -> None:
    # an approval resume tails the continuation, which opens with the tool result
    # (run_turn writes results before the answer). Its message has no turn_id, so
    # the adapter emits an id-less ``start`` and the client folds the output +
    # answer into the assistant message it resubmitted (no new message).
    tool_msg = ai.tool_message(tool_call_id="tc-1", tool_name="bash", result="out")
    await _write(
        "s1",
        events_.ToolCallResult(message=tool_msg, results=tool_msg.tool_results),
        *_text_events("done"),
        stream.session_waiting(turn_index=0),
    )

    payloads = _sse_payloads(await _collect_sse("s1"))
    starts = [p for p in payloads if p.get("type") == "start"]
    assert starts and "messageId" not in starts[0]
    assert any(p.get("type") == "tool-output-available" for p in payloads)
    assert any(p.get("type") == "text-delta" for p in payloads)


async def test_to_sse_reloads_the_current_step_and_continues() -> None:
    # a retried llm_step wipes its own aborted attempt's events and asks the
    # client to discard the current step. The stream stays open so events from
    # the retry can continue over the same connection.
    await _write(
        "s1",
        stream.session_started(),
        stream.turn_started(turn_index=0),
        events_.StreamStart(),
        events_.ToolStart(tool_call_id="tc-aborted", tool_name="bash"),
        stream.reload_requested(),
        events_.ToolStart(tool_call_id="tc-retry", tool_name="bash"),
        stream.session_waiting(turn_index=0),
    )
    lines = await _collect_sse("s1")

    payloads = _sse_payloads(lines)
    assert any(p.get("type") == "data-reload" for p in payloads)
    tool_starts = [p for p in payloads if p.get("type") == "tool-input-start"]
    assert [p["toolCallId"] for p in tool_starts] == ["tc-aborted", "tc-retry"]
    assert "[DONE]" in lines[-1]


async def test_to_sse_replays_reload_marker() -> None:
    # ``reload.requested`` is durable history, so a fresh connection sees the
    # marker too. It forwards it and continues with the retried step's events.

    await _write(
        "s1",
        stream.session_started(),
        stream.turn_started(turn_index=0),
        stream.reload_requested(),
        *_text_events("done"),
        stream.session_waiting(turn_index=0),
    )
    lines = await _collect_sse("s1")

    payloads = _sse_payloads(lines)
    assert any(p.get("type") == "data-reload" for p in payloads)
    deltas = [p for p in payloads if p.get("type") == "text-delta"]
    assert [delta["delta"] for delta in deltas] == ["done"]
    assert "[DONE]" in lines[-1]


async def test_to_sse_interleaves_live_subagent_progress() -> None:
    child_run_id = "wrun-test-child-tc-1"
    child_message = ai.messages.Message(
        role="assistant", parts=[messages_.TextPart(text="child says hi")]
    )
    await _write_run(
        child_run_id,
        events_.StreamStart(),
        events_.StreamEnd(message=child_message),
    )
    await wf_world.get_world().streams_close(
        child_run_id, wf_streams.workflow_run_stream_id(child_run_id)
    )

    parent_call = messages_.ToolCallPart(
        tool_call_id="tc-1", tool_name="subagent", tool_args='{"prompt":"go"}'
    )
    await _write(
        "s1",
        stream.session_started(),
        stream.turn_started(turn_index=0),
        events_.StreamStart(),
        events_.ToolStart(tool_call_id="tc-1", tool_name="subagent"),
        events_.ToolEnd(tool_call_id="tc-1", tool_call=parent_call),
        events_.StreamEnd(
            message=ai.messages.Message(role="assistant", parts=[parent_call])
        ),
        stream.subagent_called(
            tool_call_id="tc-1",
            child_session_id="s1:child:tc-1",
            child_run_id=child_run_id,
            name="helper",
        ),
    )

    # consume until the preliminary child output arrives, then finish the turn —
    # mirrors the live ordering (child runs while the parent turn is open).
    gen = chat.to_sse("s1", 0)
    lines: list[str] = []

    async def read_until_preliminary() -> None:
        async for line in gen:
            lines.append(line)
            if "child says hi" in line:
                return

    await asyncio.wait_for(read_until_preliminary(), timeout=5)

    await _write(
        "s1",
        stream.subagent_completed(tool_call_id="tc-1", is_error=False),
        stream.turn_started(turn_index=1),
        *_text_events("all done"),
        stream.session_waiting(turn_index=1),
    )

    async def drain() -> None:
        async for line in gen:
            lines.append(line)

    await asyncio.wait_for(drain(), timeout=5)

    payloads = _sse_payloads(lines)
    preliminary = [
        payload
        for payload in payloads
        if payload.get("type") == "tool-output-available" and payload.get("preliminary")
    ]
    assert preliminary, "no preliminary subagent output reached the client"
    nested = preliminary[-1]["output"]
    assert nested["role"] == "assistant"
    assert any(
        part.get("type") == "text" and part.get("text") == "child says hi"
        for part in nested["parts"]
    )
    # the parent turn's own text still arrives exactly once afterwards
    deltas = [p["delta"] for p in payloads if p.get("type") == "text-delta"]
    assert deltas.count("all done") == 1
    assert "[DONE]" in lines[-1]
