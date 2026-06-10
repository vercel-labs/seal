"""End-to-end driver tests: the real durable engine, in-process.

``run_session`` executes under the workflow SDK's LocalWorld with one change:
``world.queue`` dispatches to the workflow/step handlers as asyncio tasks
instead of going through the queue service. Everything else is real — replay,
suspensions, workflow hooks, the jsonl store, the bash subprocess. The only
test double is the scripted model (``mock_llm``).

These are the regression net for the failure modes that matter here:
duplicated or missing messages after replay, unanswered tool calls, and
deadlocks (every wait is bounded, so a deadlock is a fast red test).
"""

from __future__ import annotations

import asyncio
import gc
import itertools
from typing import Any

import pytest
import vercel._internal.workflow.py_sandbox as py_sandbox

# mirror worker.py: these modules must come from the host inside the sandbox.
py_sandbox._PASSTHROUGHS.update({"ai", "pathlib"})

import ai  # noqa: E402
import ai.models as models  # noqa: E402
import vercel._internal.workflow.runtime as wf_runtime  # noqa: E402
import vercel._internal.workflow.world as wf_world  # noqa: E402
import vercel._internal.workflow.worlds.local as wf_local  # noqa: E402
import vercel.workflow  # noqa: E402
from conftest import (  # noqa: E402
    MockProvider,
    assert_message_invariants,
    text_msg,
    tool_call_msg,
)

import agent  # noqa: E402
import agent.driver as driver  # noqa: E402
from agent import proto, session, storage, stream  # noqa: E402


class InProcessWorld(wf_local.LocalWorld):
    """LocalWorld with the queue bridged to in-process handler dispatch."""

    def __init__(self, registry: Any) -> None:
        super().__init__()
        self._registry = registry
        self._tasks: set[asyncio.Task[None]] = set()
        self._locks: dict[str, asyncio.Lock] = {}
        self._ids = itertools.count()
        self.errors: list[BaseException] = []

    async def queue(
        self,
        queue_name: str,
        message: Any,
        *,
        deployment_id: str | None = None,
        idempotency_key: str | None = None,
        delay_seconds: float | None = None,
        **kwargs: Any,
    ) -> str:
        message_id = f"msg_{next(self._ids)}"
        task = asyncio.create_task(
            self._deliver(queue_name, message, delay_seconds or 0, message_id)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return message_id

    async def _deliver(
        self, queue_name: str, message: Any, delay: float, message_id: str
    ) -> None:
        try:
            if delay:
                await asyncio.sleep(min(delay, 1.0))
            if queue_name.startswith("__wkf_workflow_"):
                handler = wf_runtime.workflow_handler
            elif queue_name.startswith("__wkf_step_"):
                handler = wf_runtime.step_handler
            else:
                raise RuntimeError(f"unexpected queue: {queue_name}")
            run_id = getattr(message, "run_id", None) or message.workflow_run_id
            lock = self._locks.setdefault(run_id, asyncio.Lock())
            attempt = 1
            while True:
                async with lock:
                    retry = await handler(
                        message.model_dump(),
                        attempt=attempt,
                        queue_name=queue_name,
                        message_id=message_id,
                        registry=self._registry,
                    )
                    # a suspended workflow abandons its in-flight agent.run
                    # generator. in production the invocation's process dies
                    # with it; here its finalizer must run (it clears global
                    # ai-sdk hook state) before the next invocation replays
                    # the body, or it clobbers the replay's pending hooks.
                    for _ in range(3):
                        gc.collect()
                        await asyncio.sleep(0)
                if retry is None:
                    return
                attempt += 1
                await asyncio.sleep(min(retry, 0.5))
        except asyncio.CancelledError:
            raise
        except BaseException as error:  # noqa: BLE001 — surfaced via fixture teardown
            self.errors.append(error)


@pytest.fixture
async def world() -> Any:
    bridged = InProcessWorld(agent.workflow)
    wf_world.set_world(bridged)
    yield bridged
    for task in list(bridged._tasks):
        task.cancel()
    wf_world.set_world(None)
    assert bridged.errors == [], f"workflow delivery errors: {bridged.errors}"


@pytest.fixture
def scripted_model(
    monkeypatch: pytest.MonkeyPatch, mock_llm: MockProvider
) -> MockProvider:
    model = models.Model(id="mock-model", provider=mock_llm)
    monkeypatch.setattr(ai, "get_model", lambda model_id=None, **kwargs: model)
    return mock_llm


async def _start(session_id: str, prompt: str) -> Any:
    return await vercel.workflow.start(
        driver.run_session,
        proto.SessionInput(session_id=session_id, prompt=prompt).model_dump(
            mode="json"
        ),
    )


async def _wait_for_lifecycle(
    session_id: str, type_: str, *, count: int = 1, timeout: float = 15
) -> None:
    async def watch() -> None:
        while True:
            seen = 0
            async for event in stream.replay(session_id):
                if isinstance(event, proto.LifecycleEvent) and event.type == type_:
                    seen += 1
                    if seen >= count:
                        return
            await asyncio.sleep(0.02)

    await asyncio.wait_for(watch(), timeout)


async def _resume(token: str, payload: proto.ResumePayload) -> None:
    hook = proto.SessionHook(payload=payload)
    for attempt in range(100):
        try:
            await hook.resume(token)
            return
        except RuntimeError as error:
            if attempt == 99 or "not found" not in str(error).lower():
                raise
            await asyncio.sleep(0.05)


async def _wait_run(run: Any, timeout: float = 20) -> Any:
    async def poll() -> Any:
        while await run.status() not in ("completed", "failed", "cancelled"):
            await asyncio.sleep(0.05)
        return await run.return_value()

    return await asyncio.wait_for(poll(), timeout)


async def _lifecycle(session_id: str) -> list[str]:
    return [
        event.type
        async for event in stream.replay(session_id)
        if isinstance(event, proto.LifecycleEvent)
    ]


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
    assert [m.role for m in bundle.messages] == ["system", "user", "assistant"]
    assert bundle.messages[-1].text == "child answer"

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
