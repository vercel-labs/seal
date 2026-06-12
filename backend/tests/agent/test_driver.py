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
from pathlib import Path
from typing import Any

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
# Every combination of tools that can share one assistant turn, in both
# schedule orders. Each scenario runs to completion and checks history and
# stream; the ``world`` fixture's teardown adds the engine laws (nothing left
# hanging, every replay pass requests the same things in the same order).
# Steps of one run overlapping is expected — tools are meant to run
# concurrently — so the rendezvous tests below assert that they actually do.


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


def _results_by_id(
    state: proto.SessionState,
) -> dict[str, ai.messages.ToolResultPart]:
    return {
        part.tool_call_id: part
        for message in state.messages
        for part in message.tool_results
    }


async def test_two_granted_bash_commands_run_concurrently(
    world: InProcessWorld, scripted_model: MockProvider, tmp_path: Path
) -> None:
    """The happy path of "run these two commands": both approved in one turn.

    Each command parks until the other's marker file appears, so the results
    only come out clean if the engine really runs them side by side —
    serialized execution makes the first command exit 1.
    """

    def rendezvous(mine: str, other: str) -> str:
        return (
            f"cd {tmp_path} && touch {mine} && for _ in $(seq 100); do "
            f"[ -f {other} ] && echo {mine}-out && exit 0; sleep 0.05; done; exit 1"
        )

    scripted_model.responses = [
        [
            _assistant(
                "running both",
                ("tc-a", "bash", json.dumps({"command": rendezvous("alpha", "beta")})),
                ("tc-b", "bash", json.dumps({"command": rendezvous("beta", "alpha")})),
            )
        ],
        [text_msg("both done")],
    ]

    await _start("s1", "run both")
    await _wait_for_lifecycle("s1", proto.TOOL_APPROVAL_REQUESTED)
    await _resume(
        "seal-session:s1:0",
        proto.ToolApprovals(
            tool_approvals=[
                proto.ToolApprovalResponse(tool_call_id="tc-a", granted=True),
                proto.ToolApprovalResponse(tool_call_id="tc-b", granted=True),
            ]
        ),
    )
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    assert_message_invariants(state.messages)
    results = _results_by_id(state)
    assert results["tc-a"].result == "alpha-out\n"
    assert results["tc-b"].result == "beta-out\n"
    assert state.messages[-1].text == "both done"
    await assert_stream_invariants("s1", state.messages)


async def test_parallel_subagents_run_concurrently(
    world: InProcessWorld,
    scripted_model: MockProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two delegated tasks must overlap, not run one after the other.

    Each child's model call blocks until the other child has also reached the
    model; sequential dispatch times that wait out and errors the first child,
    so its report assertion reds.
    """
    scripted_model.responses = [
        [
            _assistant(
                "fanning out",
                ("tc-a", "subagent", '{"prompt": "task-a", "name": "a"}'),
                ("tc-b", "subagent", '{"prompt": "task-b", "name": "b"}'),
            )
        ],
        [text_msg("combined")],
    ]
    scripted_model.keyed_responses = {
        "task-a": [text_msg("report-a")],
        "task-b": [text_msg("report-b")],
    }

    reached = {"task-a": asyncio.Event(), "task-b": asyncio.Event()}
    inner_stream = scripted_model.stream

    def rendezvous_stream(model: Any, messages: Any, **kwargs: Any) -> Any:
        events = inner_stream(model, messages, **kwargs)
        last_user = next((m.text for m in reversed(messages) if m.role == "user"), "")
        keys = [key for key in reached if key in last_user]
        if not keys:
            return events  # the parent's own calls pass through ungated

        async def gated() -> AsyncGenerator[Any]:
            reached[keys[0]].set()
            others = [event for key, event in reached.items() if key != keys[0]]
            await asyncio.gather(*(asyncio.wait_for(e.wait(), 10) for e in others))
            async for event in events:
                yield event

        return gated()

    monkeypatch.setattr(scripted_model, "stream", rendezvous_stream)

    await _start("s1", "fan out")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    assert_message_invariants(state.messages)
    results = _results_by_id(state)
    for tool_call_id, report in (("tc-a", "report-a"), ("tc-b", "report-b")):
        bundle = ai.agents.MessageBundle.model_validate(results[tool_call_id].result)
        assert bundle.messages[-1].text == report
    assert state.messages[-1].text == "combined"
    await assert_stream_invariants("s1", state.messages)


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
