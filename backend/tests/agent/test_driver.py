"""End-to-end driver tests: the real durable engine, in-process.

The harness (``tests/harness.py``) runs ``run_session`` on the real workflow
engine; the only test double is the scripted model (``scripted_model``).

These are the regression net for the failure modes that matter here:
duplicated or missing messages after replay, unanswered tool calls, and
deadlocks (every wait is bounded, so a deadlock is a fast red test).
"""

from __future__ import annotations

import base64
import os

import ai
import pytest
from conftest import MockProvider, assert_message_invariants, text_msg, tool_call_msg
from harness import (
    InProcessWorld,
    read_state,
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
    wait_for_event as _wait_for_event,
)
from harness import (
    wait_for_hook as _wait_for_hook,
)
from harness import (
    wait_for_lifecycle as _wait_for_lifecycle,
)

from agent import proto


async def test_single_turn_suspends(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [[text_msg("hello there")]]

    await _start("s1", "hi")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await read_state("s1")
    assert state is not None
    assert [m.role for m in state.messages] == ["system", "user", "assistant"]
    assert state.messages[-1].text == "hello there"
    assert_message_invariants(state.messages)


async def test_failed_turn_parks_and_accepts_another_message(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = []

    run = await _start("s1", "fail")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)
    assert await run.status() not in ("completed", "failed", "cancelled")

    scripted_model.responses = [[text_msg("recovered")]]
    await _resume(proto.session_hook_token("s1"), proto.NewUserMessage(prompt="retry"))
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING, count=2)

    state = await read_state("s1")
    assert state is not None
    assert [m.role for m in state.messages] == [
        "system",
        "user",
        "user",
        "assistant",
    ]
    assert state.messages[-1].text == "recovered"
    assert_message_invariants(state.messages)


async def test_resume_appends_user_message_without_duplicating_history(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [[text_msg("first answer")], [text_msg("second answer")]]

    run = await _start("s1", "one")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)
    first_hook = await _wait_for_hook(proto.turn_hook_token("s1"))
    first_session_hook = await _wait_for_hook(proto.session_hook_token("s1"))
    assert first_hook.run_id == run.run_id
    assert first_session_hook.run_id == run.run_id

    await _resume(proto.session_hook_token("s1"), proto.NewUserMessage(prompt="two"))
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING, count=2)
    second_hook = await _wait_for_hook(proto.turn_hook_token("s1"))
    second_session_hook = await _wait_for_hook(proto.session_hook_token("s1"))
    assert second_hook.hook_id == first_hook.hook_id
    assert second_session_hook.hook_id == first_session_hook.hook_id

    state = await read_state("s1")
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
    # The turn parks on the approval hook; the gated tool has not run yet, so
    # the model was called exactly once.
    await _wait_for_event("s1", ai.events.RunBlocked)
    assert scripted_model.call_count == 1

    await _resume_approval(
        "s1", proto.ToolApprovalResponse(tool_call_id="tc-1", granted=True)
    )
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await read_state("s1")
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
    await _wait_for_event("s2", ai.events.RunBlocked)
    assert scripted_model.call_count == 1

    await _resume_approval(
        "s2", proto.ToolApprovalResponse(tool_call_id="tc-a", granted=True)
    )
    await _resume_approval(
        "s2", proto.ToolApprovalResponse(tool_call_id="tc-b", granted=True)
    )
    await _wait_for_lifecycle("s2", proto.SESSION_WAITING)

    state = await read_state("s2")
    assert state is not None
    [tool_message] = [m for m in state.messages if m.role == "tool"]
    results = {r.tool_call_id: r.result for r in tool_message.tool_results}
    assert results == {"tc-a": "a\n", "tc-b": "b\n"}
    assert_message_invariants(state.messages)
    assert scripted_model.call_count == 2


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

    state = await read_state("s1")
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
        proto.SESSION_WAITING,
    ]
    # the child ran as a single turn on its own stream (no session wrapper)
    assert await _lifecycle("s1:child:tc-sub") == []
    assert scripted_model.call_count == 3


async def test_generate_image_returns_multipart_result(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    png_b64 = base64.b64encode(b"\x89PNG fake image bytes").decode()
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-img",
                name="generate_image",
                args='{"prompt": "a cat in cherry blossoms"}',
                text="drawing it",
            )
        ],
        # the image model's turn: an inline image alongside text
        [
            ai.messages.Message(
                role="assistant",
                parts=[
                    ai.messages.TextPart(text="here it is"),
                    ai.messages.FilePart(data=png_b64, media_type="image/png"),
                ],
            )
        ],
        [text_msg("done drawing")],
    ]

    await _start("s1", "draw a cat")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await read_state("s1")
    assert state is not None
    assert_message_invariants(state.messages)
    assert state.messages[-1].text == "done drawing"

    # the tool result is a multipart ContentOutput carrying the image, so the
    # model (and the UI adapter) sees the actual media.
    [tool_message] = [m for m in state.messages if m.role == "tool"]
    [result] = tool_message.tool_results
    assert result.tool_call_id == "tc-img"
    assert result.result_kind == "special"
    content = result.result
    assert isinstance(content, ai.messages.ContentOutput)
    [text_part, file_part] = content.value
    assert isinstance(text_part, ai.messages.TextPart)
    assert text_part.text == "here it is"
    assert isinstance(file_part, ai.messages.FilePart)
    assert file_part.media_type == "image/png"
    assert file_part.data == png_b64
    assert scripted_model.call_count == 3

    # what the follow-up model call actually saw: the tool result's
    # model-facing value must still be typed after the step JSON round-trip,
    # or providers JSON-encode it and the image goes up as base64 text.
    final_call = scripted_model.calls[-1]
    [seen_result] = [part for m in final_call for part in m.tool_results]
    assert isinstance(seen_result.get_model_input(), ai.messages.ContentOutput)


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

    state = await read_state(session_id)
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
