"""End-to-end driver tests: the real durable engine, in-process.

The harness (``tests/harness.py``) runs ``run_session`` on the real workflow
engine; the only test double is the scripted model (``scripted_model``).

These are the regression net for the failure modes that matter here:
duplicated or missing messages after replay, unanswered tool calls, and
deadlocks (every wait is bounded, so a deadlock is a fast red test).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

import ai
import ai.types.messages as messages_
import pytest
from conftest import (
    MockProvider,
    assert_message_invariants,
    assert_stream_invariants,
    text_msg,
    tool_call_msg,
)
from harness import (
    InProcessWorld,
)
from harness import (
    lifecycle as _lifecycle,
)
from harness import (
    resume_session as _resume,
)
from harness import (
    start_session as _start,
)
from harness import (
    wait_for_lifecycle as _wait_for_lifecycle,
)
from harness import (
    wait_run as _wait_run,
)

from agent import proto, session, storage


async def test_single_turn_suspends_then_closes(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [[text_msg("hello there")]]

    run = await _start("s1", "hi")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    assert [m.role for m in state.messages] == ["system", "user", "assistant"]
    assert state.messages[-1].text == "hello there"
    assert_message_invariants(state.messages)

    await _resume("seal-session:s1:0", proto.NewUserMessage(close=True))
    output = proto.SessionOutput.model_validate(await _wait_run(run))
    assert output.output == "hello there"
    assert not output.is_error

    assert await _lifecycle("s1") == [
        proto.SESSION_STARTED,
        proto.TURN_STARTED,
        proto.TURN_COMPLETED,
        proto.SESSION_WAITING,
        proto.SESSION_COMPLETED,
    ]
    _, closed = await storage.store().info("s1", "default")
    assert closed


async def test_resume_appends_user_message_without_duplicating_history(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [[text_msg("first answer")], [text_msg("second answer")]]

    await _start("s1", "one")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    await _resume("seal-session:s1:0", proto.NewUserMessage(prompt="two"))
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING, count=2)

    state = await session.read_session("s1")
    assert state is not None
    assert [m.role for m in state.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [m.text for m in state.messages if m.role == "user"] == ["one", "two"]
    assert state.messages[-1].text == "second answer"
    assert_message_invariants(state.messages)
    assert scripted_model.call_count == 2


async def test_gated_tool_approval_replays_the_turn_without_duplication(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-1",
                name="bash",
                args='{"command": "echo approved-run"}',
                text="running it",
            )
        ],
        [text_msg("done")],
    ]

    await _start("s1", "run it")
    await _wait_for_lifecycle("s1", proto.TOOL_APPROVAL_REQUESTED)

    # the turn parked before executing the tool: exactly one model call so far,
    # and the approval request carries the gated call's identity.
    assert scripted_model.call_count == 1
    state = await session.read_session("s1")
    assert state is not None
    assert state.pending is not None
    [request] = state.pending.tool_approval_requests
    assert request.tool_call_id == "tc-1"
    assert request.tool_name == "bash"

    await _resume(
        "seal-session:s1:0",
        proto.ToolApprovals(
            tool_approvals=[
                proto.ToolApprovalResponse(tool_call_id="tc-1", granted=True)
            ]
        ),
    )
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    assert [m.role for m in state.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    # the bash subprocess really ran, exactly once, on the replayed turn
    [tool_message] = [m for m in state.messages if m.role == "tool"]
    [result] = tool_message.tool_results
    assert result.tool_call_id == "tc-1"
    assert result.result == "approved-run\n"
    assert_message_invariants(state.messages)
    # the replayed turn called the model once more (for the final answer) —
    # the gated turn itself was replayed from cache, not re-generated.
    assert scripted_model.call_count == 2
    # consumed approvals don't leak into the next turn
    assert state.tool_approvals == []
    assert state.pending is None
    await assert_stream_invariants("s1", state.messages)
    await world.drain()
    await world.check_step_serialization()

    assert await _lifecycle("s1") == [
        proto.SESSION_STARTED,
        proto.TURN_STARTED,
        proto.TURN_COMPLETED,
        proto.TOOL_APPROVAL_REQUESTED,
        proto.TOOL_APPROVAL_RESOLVED,
        proto.TURN_STARTED,
        proto.TURN_COMPLETED,
        proto.SESSION_WAITING,
    ]


async def test_subagent_result_lands_on_the_trailing_tool_message(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-sub",
                name="subagent",
                args='{"prompt": "say hi", "name": "helper"}',
                text="delegating",
            )
        ],
        [text_msg("child answer")],  # the child session's single turn
        [text_msg("final answer")],  # the parent's follow-up turn
    ]

    await _start("s1", "delegate")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    assert [m.role for m in state.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert_message_invariants(state.messages)
    assert state.messages[-1].text == "final answer"

    # the child's full transcript (a MessageBundle) is the tool result
    [tool_message] = [m for m in state.messages if m.role == "tool"]
    [result] = tool_message.tool_results
    assert result.tool_call_id == "tc-sub"
    bundle = ai.agents.MessageBundle.model_validate(result.result)
    assert [m.role for m in bundle.messages] == ["system", "user", "assistant"]
    assert bundle.messages[-1].text == "child answer"
    await assert_stream_invariants("s1", state.messages)
    await world.drain()
    await world.check_step_serialization()

    assert await _lifecycle("s1") == [
        proto.SESSION_STARTED,
        proto.TURN_STARTED,
        proto.TURN_COMPLETED,
        proto.SUBAGENT_CALLED,
        proto.SUBAGENT_COMPLETED,
        proto.TURN_STARTED,
        proto.TURN_COMPLETED,
        proto.SESSION_WAITING,
    ]
    # the child ran as its own session on its own stream, and completed
    assert await _lifecycle("s1:child:tc-sub") == [
        proto.SESSION_STARTED,
        proto.TURN_STARTED,
        proto.TURN_COMPLETED,
        proto.SESSION_COMPLETED,
    ]
    assert scripted_model.call_count == 3


# --- schedule-order coverage matrix -----------------------------------------------
#
# Every combination of tools that can share one assistant turn, in the
# schedule orders that previously only passed (or failed) by luck in the real
# delivery topology. Each scenario runs to completion and checks history,
# stream, and (where current code already satisfies it) engine health.


def _assistant(text: str, *calls: tuple[str, str, str]) -> messages_.Message:
    parts: list[messages_.Part] = [messages_.TextPart(text=text)]
    for tc_id, name, args in calls:
        parts.append(
            messages_.ToolCallPart(tool_call_id=tc_id, tool_name=name, tool_args=args)
        )
    return messages_.Message(role="assistant", parts=parts)


@pytest.fixture
async def http_url() -> AsyncGenerator[str]:
    """A local one-response HTTP server so web_fetch needs no network."""

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        while True:
            line = await reader.readline()
            if not line or line == b"\r\n":
                break
        body = b"fetched-ok"
        writer.write(
            b"HTTP/1.1 200 OK\r\ncontent-length: %d\r\n"
            b"connection: close\r\n\r\n%s" % (len(body), body)
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/"
    server.close()
    await server.wait_closed()


async def _run_two_granted_bash(
    scripted_model: MockProvider, *, timeout: float = 15
) -> proto.SessionState:
    """Both commands approved — two tool steps become live in the same turn."""
    scripted_model.responses = [
        [
            _assistant(
                "running both",
                ("tc-a", "bash", '{"command": "echo alpha-out"}'),
                ("tc-b", "bash", '{"command": "echo beta-out"}'),
            )
        ],
        [text_msg("both done")],
    ]

    await _start("s1", "run both")
    await _wait_for_lifecycle("s1", proto.TOOL_APPROVAL_REQUESTED, timeout=timeout)
    await _resume(
        "seal-session:s1:0",
        proto.ToolApprovals(
            tool_approvals=[
                proto.ToolApprovalResponse(tool_call_id="tc-a", granted=True),
                proto.ToolApprovalResponse(tool_call_id="tc-b", granted=True),
            ]
        ),
    )
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING, timeout=timeout)

    state = await session.read_session("s1")
    assert state is not None
    return state


def _results_by_id(
    state: proto.SessionState,
) -> dict[str, ai.messages.ToolResultPart]:
    return {
        part.tool_call_id: part
        for message in state.messages
        for part in message.tool_results
    }


@pytest.mark.replay_divergent
async def test_two_granted_bash_commands_both_run(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    state = await _run_two_granted_bash(scripted_model)

    assert_message_invariants(state.messages)
    results = _results_by_id(state)
    assert results["tc-a"].result == "alpha-out\n"
    assert results["tc-b"].result == "beta-out\n"
    assert state.messages[-1].text == "both done"


@pytest.mark.replay_divergent
@pytest.mark.xfail(
    strict=True,
    reason="run_turn dispatches both _bash steps (and monitor/loop write_event "
    "steps) concurrently, and writes results in runner-completion order, which "
    "is not replay-stable — a diverged replay drops or duplicates stream "
    "results. The hook-dispatch refactor must issue all durable steps from "
    "one coroutine in a replay-stable order.",
)
async def test_two_granted_bash_engine_health(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    state = await _run_two_granted_bash(scripted_model)
    await world.drain()
    await world.check_step_serialization()
    world.check_replay_determinism()
    await assert_stream_invariants("s1", state.messages, in_order=True)


async def test_bash_scheduled_before_subagent(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [
        [
            _assistant(
                "run then delegate",
                ("tc-cmd", "bash", '{"command": "echo cmd-out"}'),
                ("tc-sub", "subagent", '{"prompt": "task-delta", "name": "delta"}'),
            )
        ],
        [text_msg("merged")],
    ]
    scripted_model.keyed_responses = {"task-delta": [text_msg("delta report")]}

    await _start("s1", "run and delegate")
    await _wait_for_lifecycle("s1", proto.TOOL_APPROVAL_REQUESTED)
    await _resume(
        "seal-session:s1:0",
        proto.ToolApprovals(
            tool_approvals=[
                proto.ToolApprovalResponse(tool_call_id="tc-cmd", granted=True)
            ]
        ),
    )
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    assert_message_invariants(state.messages)
    results = _results_by_id(state)
    assert results["tc-cmd"].result == "cmd-out\n"
    bundle = ai.agents.MessageBundle.model_validate(results["tc-sub"].result)
    assert bundle.messages[-1].text == "delta report"
    await assert_stream_invariants("s1", state.messages)


async def test_subagent_scheduled_before_bash(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [
        [
            _assistant(
                "delegate then run",
                ("tc-sub", "subagent", '{"prompt": "task-edge", "name": "edge"}'),
                ("tc-cmd", "bash", '{"command": "echo cmd-out"}'),
            )
        ],
        [text_msg("merged")],
    ]
    scripted_model.keyed_responses = {"task-edge": [text_msg("edge report")]}

    await _start("s1", "delegate and run")
    await _wait_for_lifecycle("s1", proto.TOOL_APPROVAL_REQUESTED)
    await _resume(
        "seal-session:s1:0",
        proto.ToolApprovals(
            tool_approvals=[
                proto.ToolApprovalResponse(tool_call_id="tc-cmd", granted=True)
            ]
        ),
    )
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    assert_message_invariants(state.messages)
    results = _results_by_id(state)
    assert results["tc-cmd"].result == "cmd-out\n"
    bundle = ai.agents.MessageBundle.model_validate(results["tc-sub"].result)
    assert bundle.messages[-1].text == "edge report"
    await assert_stream_invariants("s1", state.messages)


async def test_web_fetch_scheduled_before_gated_bash(
    world: InProcessWorld, scripted_model: MockProvider, http_url: str
) -> None:
    # web_fetch is ungated and runs while bash parks for approval; on the
    # replayed turn its result must come back from cache, not a second fetch.
    scripted_model.responses = [
        [
            _assistant(
                "fetch then run",
                ("tc-web", "web_fetch", json.dumps({"url": http_url})),
                ("tc-cmd", "bash", '{"command": "echo cmd-out"}'),
            )
        ],
        [text_msg("wrapped")],
    ]

    await _start("s1", "fetch and run")
    await _wait_for_lifecycle("s1", proto.TOOL_APPROVAL_REQUESTED)
    await _resume(
        "seal-session:s1:0",
        proto.ToolApprovals(
            tool_approvals=[
                proto.ToolApprovalResponse(tool_call_id="tc-cmd", granted=True)
            ]
        ),
    )
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    assert_message_invariants(state.messages)
    results = _results_by_id(state)
    assert "fetched-ok" in str(results["tc-web"].result)
    assert not results["tc-web"].is_error
    assert results["tc-cmd"].result == "cmd-out\n"
    await assert_stream_invariants("s1", state.messages)


async def test_one_of_everything_in_a_single_turn(
    world: InProcessWorld, scripted_model: MockProvider, http_url: str
) -> None:
    scripted_model.responses = [
        [
            _assistant(
                "everything at once",
                ("tc-web", "web_fetch", json.dumps({"url": http_url})),
                ("tc-sub", "subagent", '{"prompt": "task-omega", "name": "omega"}'),
                ("tc-cmd", "bash", '{"command": "echo all-out"}'),
            )
        ],
        [text_msg("done with everything")],
    ]
    scripted_model.keyed_responses = {"task-omega": [text_msg("omega report")]}

    await _start("s1", "do everything")
    await _wait_for_lifecycle("s1", proto.TOOL_APPROVAL_REQUESTED)
    await _resume(
        "seal-session:s1:0",
        proto.ToolApprovals(
            tool_approvals=[
                proto.ToolApprovalResponse(tool_call_id="tc-cmd", granted=True)
            ]
        ),
    )
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    assert_message_invariants(state.messages)
    results = _results_by_id(state)
    assert "fetched-ok" in str(results["tc-web"].result)
    assert results["tc-cmd"].result == "all-out\n"
    bundle = ai.agents.MessageBundle.model_validate(results["tc-sub"].result)
    assert bundle.messages[-1].text == "omega report"
    assert state.messages[-1].text == "done with everything"
    await assert_stream_invariants("s1", state.messages)


@pytest.mark.xfail(
    strict=True,
    reason="run_turn dispatches overlapping steps; see "
    "test_two_granted_bash_engine_health",
)
async def test_one_of_everything_steps_never_overlap(
    world: InProcessWorld, scripted_model: MockProvider, http_url: str
) -> None:
    await test_one_of_everything_in_a_single_turn(world, scripted_model, http_url)
    await world.drain()
    await world.check_step_serialization()


@pytest.mark.hostile_delivery
@pytest.mark.xfail(
    strict=True,
    reason="overlapping step dispatch lets the delivery layer drop one step "
    "between step_started and any terminal event; with max_retries=0 and the "
    "already-started redelivery guard the run wedges forever (the production "
    "failure). Goes green once run_turn never has two steps in flight.",
)
async def test_session_survives_a_hostile_delivery_layer(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    state = await _run_two_granted_bash(scripted_model, timeout=5)
    results = _results_by_id(state)
    assert results["tc-a"].result == "alpha-out\n"
    assert results["tc-b"].result == "beta-out\n"
