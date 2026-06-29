"""End-to-end driver tests: the real durable engine, in-process.

The harness (``tests/harness.py``) runs ``run_session`` on the real workflow
engine; the only test double is the scripted model (``scripted_model``).

These are the regression net for the failure modes that matter here:
duplicated or missing messages after replay, unanswered tool calls, and
deadlocks (every wait is bounded, so a deadlock is a fast red test).
"""

from __future__ import annotations

import ai
from conftest import MockProvider, assert_message_invariants, text_msg, tool_call_msg
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
