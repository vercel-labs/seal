from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from ai import messages as ai_messages
from ai.agents.hooks import TOOL_APPROVAL_HOOK_TYPE

from core import durable_agent, sessions


def test_workflow_module_imports_inside_workflow_sandbox() -> None:
    import vercel._internal.workflow.py_sandbox as py_sandbox

    previous = sys.modules.pop("core.durable_agent", None)
    try:
        with py_sandbox.workflow_sandbox(random_seed="workflow-import"):
            module = importlib.import_module("core.durable_agent")
    finally:
        if previous is not None:
            sys.modules["core.durable_agent"] = previous

    assert module.workflow_app is not None


def test_legacy_agent_module_imports_inside_workflow_sandbox() -> None:
    import vercel._internal.workflow.py_sandbox as py_sandbox

    importlib.import_module("agent.main")
    previous = sys.modules.pop("agent.main", None)
    try:
        with py_sandbox.workflow_sandbox(random_seed="legacy-agent-import"):
            module = importlib.import_module("agent.main")
    finally:
        if previous is not None:
            sys.modules["agent.main"] = previous

    assert module.workflow is not None


def test_workflow_approval_tool_names_match_agent_tools() -> None:
    approval_required = frozenset(
        tool.name for tool in durable_agent.get_tools() if tool.require_approval
    )

    assert approval_required == durable_agent.APPROVAL_REQUIRED_TOOLS


def test_vercel_dev_local_world_uses_release_local_deployment_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vercel._internal.workflow.world as workflow_world
    import vercel._internal.workflow.worlds.local as local_world

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def send_async(*args: Any, **kwargs: Any) -> dict[str, str]:
        calls.append((args, kwargs))
        return {"messageId": "msg_1"}

    monkeypatch.delenv("VERCEL_QUEUE_TOKEN", raising=False)
    monkeypatch.delenv("VERCEL_WORKERS_IN_PROCESS", raising=False)
    vqs_client = cast(Any, local_world).vqs_client
    monkeypatch.setattr(vqs_client, "send_async", send_async)

    message_id = asyncio.run(
        local_world.LocalWorld().queue(
            "__wkf_workflow_test",
            workflow_world.WorkflowInvokePayload(runId="run_1"),
        )
    )

    assert message_id == "msg_1"
    assert calls[0][1]["deployment_id"] == "<local>"
    assert calls[0][0][1]["deploymentId"] == "<local>"


def test_workflow_pending_approval_message_uses_ai_sdk_shape() -> None:
    assistant = ai_messages.Message(
        id="assistant-1",
        role="assistant",
        parts=[
            ai_messages.ToolCallPart(
                id="call-part-1",
                tool_call_id="call-1",
                tool_name="bash",
                tool_args='{"command":"pwd"}',
            )
        ],
    )

    message = durable_agent._hook_message(
        tool_call=assistant.tool_calls[0],
        assistant_message=assistant,
        status="pending",
    )
    hook = message.parts[0]

    assert message.role == "internal"
    assert message.turn_id == "assistant-1"
    assert isinstance(hook, ai_messages.HookPart)
    assert hook.hook_id == "approve_call-1"
    assert hook.hook_type == TOOL_APPROVAL_HOOK_TYPE
    assert hook.status == "pending"


def test_workflow_records_all_pending_approvals_before_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = ai_messages.Message(
        id="user-1",
        role="user",
        parts=[ai_messages.TextPart(id="user-text-1", text="run two tools")],
    )
    assistant = ai_messages.Message(
        id="assistant-1",
        role="assistant",
        parts=[
            ai_messages.ToolCallPart(
                id="call-part-1",
                tool_call_id="call-1",
                tool_name="bash",
                tool_args='{"command":"pwd"}',
            ),
            ai_messages.ToolCallPart(
                id="call-part-2",
                tool_call_id="call-2",
                tool_name="web_fetch",
                tool_args='{"url":"https://example.com"}',
            ),
            ai_messages.ToolCallPart(
                id="call-part-3",
                tool_call_id="call-3",
                tool_name="bash",
                tool_args='{"command":"ls"}',
            ),
        ],
    )
    persisted: list[list[dict[str, Any]]] = []
    statuses: list[str] = []

    class Suspended(Exception):
        pass

    async def fake_llm_step(
        stream_id: str,
        messages_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return durable_agent.dump_message(assistant)

    async def fake_persist(
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        persisted.append([*messages])

    async def fake_set_stream_status_step(stream_id: str, status: str) -> None:
        statuses.append(status)

    async def fake_load_tool_approvals_step(
        stream_id: str,
        approval_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        return {}

    async def fake_sleep(delay: float) -> None:
        raise Suspended

    monkeypatch.setattr(durable_agent, "llm_step", fake_llm_step)
    monkeypatch.setattr(durable_agent, "_persist", fake_persist)
    monkeypatch.setattr(
        durable_agent,
        "set_stream_status_step",
        fake_set_stream_status_step,
    )
    monkeypatch.setattr(
        durable_agent,
        "load_tool_approvals_step",
        fake_load_tool_approvals_step,
    )
    monkeypatch.setattr(cast(Any, durable_agent).workflow, "sleep", fake_sleep)

    async def run() -> None:
        with pytest.raises(Suspended):
            await asyncio.wait_for(
                cast(Any, durable_agent.run_agent).func(
                    "session-1",
                    "stream-1",
                    [durable_agent.dump_message(user)],
                ),
                timeout=1,
            )

    asyncio.run(run())

    pending_hooks = [
        message["parts"][0]["hook_id"]
        for message in persisted[-1]
        if message["role"] == "internal"
    ]

    assert pending_hooks == [
        "approve_call-1",
        "approve_call-2",
        "approve_call-3",
    ]
    assert statuses == ["waiting"]


def test_session_active_run_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEAL_SESSIONS_DIR", str(tmp_path))

    async def run() -> None:
        await sessions.create_session("s1")
        await sessions.set_active_run("s1", "wrun_1")
        session = await sessions.get_session("s1")
        assert session is not None
        assert session.active_run_id == "wrun_1"

        await sessions.set_active_run("s1", None)
        session = await sessions.get_session("s1")
        assert session is not None
        assert session.active_run_id is None

    asyncio.run(run())
