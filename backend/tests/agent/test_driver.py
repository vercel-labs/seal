from __future__ import annotations

import asyncio
from typing import cast

import ai
import temporalio.client
from conftest import MockProvider, assert_message_invariants, text_msg, tool_call_msg

from agent import proto, session_workflow_id, stream, temporal


async def _start(
    client: temporalio.client.Client, session_id: str, prompt: str
) -> temporalio.client.WorkflowHandle[None, None]:
    await temporal.start_or_resume(session_id, prompt)
    return cast(
        temporalio.client.WorkflowHandle[None, None],
        client.get_workflow_handle(session_workflow_id(session_id)),
    )


async def _wait_for_event(
    session_id: str, event_type: type[object] | str, *, count: int = 1
) -> object:
    stream_id = session_workflow_id(session_id)
    async with asyncio.timeout(10):
        while True:
            events = [event async for event in stream.replay(stream_id)]
            matches = [
                event
                for event in events
                if (
                    isinstance(event_type, str)
                    and isinstance(event, proto.LifecycleEvent)
                    and event.type == event_type
                )
                or (isinstance(event_type, type) and isinstance(event, event_type))
            ]
            if len(matches) >= count:
                return matches[count - 1]
            await asyncio.sleep(0.02)


async def test_session_runs_and_resumes(
    scripted_model: MockProvider, temporal_client: temporalio.client.Client
) -> None:
    scripted_model.responses = [[text_msg("first")], [text_msg("second")]]
    handle = await _start(temporal_client, "s1", "one")
    try:
        await _wait_for_event("s1", proto.SESSION_WAITING)
        await temporal.start_or_resume("s1", "two")
        await _wait_for_event("s1", proto.SESSION_WAITING, count=2)

        state = await temporal.session_state("s1")
        assert state is not None
        assert [message.role for message in state.messages] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert state.messages[-1].text == "second"
        assert_message_invariants(state.messages)
    finally:
        await handle.terminate("test complete")


async def test_approval_signal_resumes_gated_tool(
    scripted_model: MockProvider, temporal_client: temporalio.client.Client
) -> None:
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-1",
                name="bash",
                args='{"command":"echo approved"}',
                text="running",
            )
        ],
        [text_msg("done")],
    ]
    handle = await _start(temporal_client, "s2", "run it")
    try:
        blocked = await _wait_for_event("s2", ai.events.RunBlocked)
        assert isinstance(blocked, ai.events.RunBlocked)
        hook = blocked.hooks[0]
        await temporal.submit_approvals(
            "s2",
            [
                proto.ToolApprovalResponse(
                    hook_id=hook.hook_id,
                    tool_call_id="tc-1",
                    granted=True,
                )
            ],
        )
        await _wait_for_event("s2", proto.SESSION_WAITING)

        state = await temporal.session_state("s2")
        assert state is not None
        results = [
            result for message in state.messages for result in message.tool_results
        ]
        assert results[0].result == "approved\n"
        assert state.messages[-1].text == "done"
        assert_message_invariants(state.messages)
    finally:
        await handle.terminate("test complete")


async def test_subagent_runs_as_child_workflow(
    scripted_model: MockProvider, temporal_client: temporalio.client.Client
) -> None:
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-sub",
                name="subagent",
                args='{"prompt":"child task","name":"helper"}',
            )
        ],
        [text_msg("parent done")],
    ]
    scripted_model.keyed_responses = {"child task": [text_msg("child answer")]}
    handle = await _start(temporal_client, "s3", "delegate")
    try:
        await _wait_for_event("s3", proto.SESSION_WAITING)
        state = await temporal.session_state("s3")
        assert state is not None
        result = next(
            result
            for message in state.messages
            for result in message.tool_results
            if result.tool_call_id == "tc-sub"
        )
        bundle = ai.agents.MessageBundle.model_validate(result.result)
        assert bundle.messages[-1].text == "child answer"
        assert state.messages[-1].text == "parent done"
        assert_message_invariants(state.messages)
    finally:
        await handle.terminate("test complete")
