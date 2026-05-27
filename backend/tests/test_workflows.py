from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any, cast

import pytest
from ai import messages as ai_messages
from ai.agents.hooks import TOOL_APPROVAL_HOOK_TYPE

import sessions
import workflows


def test_workflow_module_imports_inside_workflow_sandbox() -> None:
    import vercel._internal.workflow.py_sandbox as py_sandbox

    with py_sandbox.workflow_sandbox(random_seed="workflow-import"):
        module = importlib.import_module("workflows")

    assert module.workflow_app is not None


def test_workflow_approval_tool_names_match_agent_tools() -> None:
    import agent

    approval_required = frozenset(
        tool.name for tool in agent.TOOLS if tool.require_approval
    )

    assert approval_required == workflows.APPROVAL_REQUIRED_TOOLS


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

    message = workflows._hook_message(
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
