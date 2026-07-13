"""End-to-end driver tests: the real durable engine, in-process.

The harness (``tests/harness.py``) runs ``run_session`` on the real workflow
engine; the only test double is the scripted model (``scripted_model``).

These are the regression net for the failure modes that matter here:
duplicated or missing messages after replay, unanswered tool calls, and
deadlocks (every wait is bounded, so a deadlock is a fast red test).
"""

from __future__ import annotations

import os

import ai
import pytest
from conftest import MockProvider, assert_message_invariants, text_msg, tool_call_msg
from harness import (
    InProcessWorld,
)
from harness import (
    lifecycle as _lifecycle,
)
from harness import (
    resume_approval as _resume_approval,
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


async def test_gated_tool_approval_runs_in_one_turn(
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
    # the turn parks on the approval hook and emits tool_approval.requested; the
    # gated tool has not run yet, so the model was called exactly once.
    await _wait_for_lifecycle("s1", proto.TOOL_APPROVAL_REQUESTED)
    assert scripted_model.call_count == 1

    await _resume_approval(
        "s1", proto.ToolApprovalResponse(tool_call_id="tc-1", granted=True)
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
    # the bash subprocess really ran, exactly once, after the approval landed
    [tool_message] = [m for m in state.messages if m.role == "tool"]
    [result] = tool_message.tool_results
    assert result.tool_call_id == "tc-1"
    assert result.result == "approved-run\n"
    assert_message_invariants(state.messages)
    # one model call for the gated turn, one more for the final answer
    assert scripted_model.call_count == 2

    # the whole exchange is a single turn that parked once on the approval.
    assert await _lifecycle("s1") == [
        proto.SESSION_STARTED,
        proto.TURN_STARTED,
        proto.TOOL_APPROVAL_REQUESTED,
        proto.TURN_COMPLETED,
        proto.SESSION_WAITING,
    ]


async def test_parallel_gated_tools_park_then_run(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [
        [
            ai.messages.Message(
                role="assistant",
                parts=[
                    ai.messages.TextPart(text="running both"),
                    ai.messages.ToolCallPart(
                        tool_call_id="tc-a",
                        tool_name="bash",
                        tool_args='{"command": "echo a"}',
                    ),
                    ai.messages.ToolCallPart(
                        tool_call_id="tc-b",
                        tool_name="bash",
                        tool_args='{"command": "echo b"}',
                    ),
                ],
            )
        ],
        [text_msg("done")],
    ]

    await _start("s2", "run both")
    # both gated calls park on their own hook before the turn parks.
    await _wait_for_lifecycle("s2", proto.TOOL_APPROVAL_REQUESTED)
    assert scripted_model.call_count == 1

    await _resume_approval(
        "s2", proto.ToolApprovalResponse(tool_call_id="tc-a", granted=True)
    )
    await _resume_approval(
        "s2", proto.ToolApprovalResponse(tool_call_id="tc-b", granted=True)
    )
    await _wait_for_lifecycle("s2", proto.SESSION_WAITING)

    state = await session.read_session("s2")
    assert state is not None
    [tool_message] = [m for m in state.messages if m.role == "tool"]
    results = {r.tool_call_id: r.result for r in tool_message.tool_results}
    assert results == {"tc-a": "a\n", "tc-b": "b\n"}
    assert_message_invariants(state.messages)
    assert scripted_model.call_count == 2
    assert proto.TOOL_APPROVAL_REQUESTED in await _lifecycle("s2")


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
    assert [m.role for m in bundle.messages] == ["assistant"]
    assert bundle.messages[-1].text == "child answer"

    assert await _lifecycle("s1") == [
        proto.SESSION_STARTED,
        proto.TURN_STARTED,
        proto.SUBAGENT_CALLED,
        proto.SUBAGENT_COMPLETED,
        proto.TURN_COMPLETED,
        proto.SESSION_WAITING,
    ]
    # the child ran as a single turn on its own stream (no session wrapper)
    assert await _lifecycle("s1:child:tc-sub") == []
    assert scripted_model.call_count == 3


# How many times to repeat the parallel-subagent stress. Kept low by default
# (each iteration re-imports `ai` per delivery, which is slow); bump for a
# heavier determinism sweep, e.g. SEAL_PARALLEL_SUBAGENT_ITERS=24.
_PARALLEL_SUBAGENT_ITERS = int(os.environ.get("SEAL_PARALLEL_SUBAGENT_ITERS", "2"))


@pytest.mark.parametrize("iteration", range(_PARALLEL_SUBAGENT_ITERS))
async def test_parallel_subagents_land_deterministically(
    world: InProcessWorld, scripted_model: MockProvider, iteration: int
) -> None:
    # Two subagents scheduled from one assistant turn run concurrently: their
    # tool coroutines and the agent loop all issue durable ``write_event`` steps,
    # so the engine must deliver recorded completions one at a time (fully
    # draining each before the next) or the two coroutines interleave their
    # writes differently across replays -> NondeterminismError. Repeated to catch
    # the flaky ordering.
    session_id = f"s{iteration}"
    scripted_model.responses = [
        [
            ai.messages.Message(
                role="assistant",
                parts=[
                    ai.messages.TextPart(text="delegating both"),
                    ai.messages.ToolCallPart(
                        tool_call_id="tc-a",
                        tool_name="subagent",
                        tool_args='{"prompt": "task-alpha", "name": "alpha"}',
                    ),
                    ai.messages.ToolCallPart(
                        tool_call_id="tc-b",
                        tool_name="subagent",
                        tool_args='{"prompt": "task-beta", "name": "beta"}',
                    ),
                ],
            )
        ],
        [text_msg("wrapped up")],  # parent's follow-up turn after both children
    ]
    scripted_model.keyed_responses = {
        "task-alpha": [text_msg("alpha-report")],
        "task-beta": [text_msg("beta-report")],
    }

    await _start(session_id, "delegate both")
    await _wait_for_lifecycle(session_id, proto.SESSION_WAITING)

    state = await session.read_session(session_id)
    assert state is not None
    assert [m.role for m in state.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert state.messages[-1].text == "wrapped up"
    assert_message_invariants(state.messages)

    [tool_message] = [m for m in state.messages if m.role == "tool"]
    results = {r.tool_call_id: r for r in tool_message.tool_results}
    assert set(results) == {"tc-a", "tc-b"}
    bundle_a = ai.agents.MessageBundle.model_validate(results["tc-a"].result)
    bundle_b = ai.agents.MessageBundle.model_validate(results["tc-b"].result)
    assert bundle_a.messages[-1].text == "alpha-report"
    assert bundle_b.messages[-1].text == "beta-report"
    # parent: 1 turn issuing both calls + 1 follow-up; each child: 1 turn
    assert scripted_model.call_count == 4
