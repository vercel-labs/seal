from __future__ import annotations

import asyncio
from typing import cast

import ai
import httpx2
import pytest
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


async def test_llm_activity_retries_and_requests_reload(
    scripted_model: MockProvider, temporal_client: temporalio.client.Client
) -> None:
    scripted_model.failures_remaining = 1
    scripted_model.responses = [[text_msg("recovered")]]
    handle = await _start(temporal_client, "s4", "retry")
    try:
        await _wait_for_event("s4", proto.SESSION_WAITING)
        events = [event async for event in stream.replay(session_workflow_id("s4"))]
        assert scripted_model.call_count == 2
        assert any(
            isinstance(event, proto.LifecycleEvent)
            and event.type == proto.RELOAD_REQUESTED
            for event in events
        )

        state = await temporal.session_state("s4")
        assert state is not None
        assert state.messages[-1].text == "recovered"
    finally:
        await handle.terminate("test complete")


async def test_interrupt_cancels_llm_activity(
    scripted_model: MockProvider, temporal_client: temporalio.client.Client
) -> None:
    model_blocked = asyncio.Event()
    cancellation_observed = asyncio.Event()
    MockProvider.wait_after_tool = model_blocked
    MockProvider.cancellation_observed = cancellation_observed
    scripted_model.responses = [
        [tool_call_msg(tc_id="tc-cancel", name="bash", args='{"command":"true"}')]
    ]
    handle = await _start(temporal_client, "s5", "cancel")
    try:
        await _wait_for_event("s5", ai.events.ToolEnd)
        await temporal.interrupt("s5")
        await _wait_for_event("s5", proto.SESSION_INTERRUPTED)
        async with asyncio.timeout(10):
            await cancellation_observed.wait()
    finally:
        await handle.terminate("test complete")


async def test_web_fetch_starts_before_model_stream_finishes(
    monkeypatch: pytest.MonkeyPatch,
    scripted_model: MockProvider,
    temporal_client: temporalio.client.Client,
) -> None:
    tool_started = asyncio.Event()
    request_count = 0
    MockProvider.wait_after_tool = tool_started
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-eager",
                name="web_fetch",
                args='{"url":"https://example.test"}',
            )
        ],
        [text_msg("done")],
    ]

    class Response:
        status_code = 200
        headers: dict[str, str] = {}
        text = "eager result"

    async def request(
        _client: httpx2.AsyncClient,
        method: str,
        url: str,
        **_kwargs: object,
    ) -> Response:
        nonlocal request_count
        assert method == "GET"
        assert url == "https://example.test"
        request_count += 1
        tool_started.set()
        return Response()

    monkeypatch.setattr(httpx2.AsyncClient, "request", request)
    handle = await _start(temporal_client, "s6", "fetch")
    try:
        await _wait_for_event("s6", proto.SESSION_WAITING)
        assert tool_started.is_set()
        assert request_count == 1

        state = await temporal.session_state("s6")
        assert state is not None
        result = next(
            result
            for message in state.messages
            for result in message.tool_results
            if result.tool_call_id == "tc-eager"
        )
        assert "eager result" in str(result.result)
        assert state.messages[-1].text == "done"
    finally:
        await handle.terminate("test complete")
