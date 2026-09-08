"""In-process runner for the durable workflow engine.

``run_session`` executes under the workflow SDK's LocalWorld with one change:
``world.queue`` dispatches to the workflow/step handlers directly instead of
going through the queue service. Deliveries run as plain asyncio tasks on the
test's event loop; the engine gives each workflow invocation its own dedicated
loop internally (``run_workflow`` runs the body under ``asyncio.run``), so the
``loop._ready`` check stays sound without the harness managing loops itself.
Dispatching on one shared loop also keeps cross-delivery ordering deterministic
(cooperative FIFO), so e.g. two parallel subagents don't race. Everything else
is real — replay, suspensions, workflow hooks, run streams, the bash
subprocess.
"""

from __future__ import annotations

import asyncio
import contextvars
import itertools
import random
from typing import Any

import ai
import vercel.workflow  # noqa: E402
import vercel.workflow._internal.runtime as wf_runtime  # noqa: E402
import vercel.workflow._internal.worlds.local as wf_local  # noqa: E402

import agent.driver as driver  # noqa: E402
from agent import proto, stream, util  # noqa: E402


class InProcessWorld(wf_local.LocalWorld):
    """LocalWorld with the queue bridged to in-process handler dispatch."""

    def __init__(self, registry: Any) -> None:
        super().__init__()
        self._registry = registry
        # Deliveries run as tasks on the test's loop; the engine spins up a
        # dedicated loop per workflow invocation internally, so the harness
        # doesn't manage loops. One asyncio.Lock per run_id serializes that run's
        # deliveries.
        self._tasks: set[asyncio.Task[None]] = set()
        self._locks: dict[str, asyncio.Lock] = {}
        self._ids = itertools.count()
        self.errors: list[BaseException] = []
        # Deterministic run ids so the workflow sandbox's seeded
        # `random` (the seed is the run id) draws the same
        # message/part ids on every run.  This isn't needed for test
        # correctness (the tests that care canonicalize ids when
        # comparing), but should reduce churn when using
        # UPDATE_FIXTURES=1.
        self._ulid = itertools.count()
        self.monotonic_ulid = self._det_ulid  # type: ignore[assignment]
        # One seeded rng shared by every step body (see `_deliver`). Fresh per
        # world, so each test draws the same sequence.
        self._step_rng = random.Random("seal-test-step-ids")

    def _det_ulid(self, _timestamp_ms: int | None = None) -> str:
        # Fixed time prefix + monotonic counter: a valid 26-char Crockford-base32
        # ULID that's stable across runs (the real one is time + os.urandom).
        return f"0000000000{next(self._ulid):016d}"

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
        # Fire-and-forget, like send_async to the queue service: schedule the
        # delivery as a task and return the message id without awaiting it.
        # A fresh contextvars context, because a real delivery is a fresh
        # request: create_task would otherwise copy the enqueuing step's
        # context (e.g. its stream-writer state) into the handler.
        message_id = f"msg_{next(self._ids)}"
        task = asyncio.create_task(
            self._deliver(queue_name, message, delay_seconds or 0, message_id),
            context=contextvars.Context(),
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
            # workflow_handler dispatches both runs and steps (via step_id).
            if not queue_name.startswith("__wkf_workflow_"):
                raise RuntimeError(f"unexpected queue: {queue_name}")
            handler = wf_runtime.workflow_handler
            run_id = getattr(message, "run_id", None) or message.workflow_run_id
            lock = self._locks.setdefault(run_id, asyncio.Lock())
            attempt = 1
            while True:
                async with (
                    lock,
                    # Install a seeded RNG for steps so they get
                    # deterministic IDs also.
                    ai.messages.use_random(self._step_rng),
                ):
                    retry = await handler(
                        message.model_dump(),
                        attempt=attempt,
                        queue_name=queue_name,
                        message_id=message_id,
                        registry=self._registry,
                    )
                if retry is None:
                    return
                attempt += 1
                await asyncio.sleep(min(retry.delay_seconds, 0.5))
        except asyncio.CancelledError:
            raise
        except BaseException as error:  # noqa: BLE001 — surfaced via fixture teardown
            self.errors.append(error)


async def start_session(session_id: str, prompt: str) -> vercel.workflow.Run[None]:
    run = await vercel.workflow.start(
        driver.run_session,
        proto.SessionInput(session_id=session_id, prompt=prompt),
    )
    async for _ in util.hook_retries(retry_count=100):
        if await stream.session_run_id(session_id) is not None:
            return run
    raise RuntimeError(f"session workflow did not register its turn hook: {session_id}")


async def read_state(session_id: str) -> proto.SessionState | None:
    """The latest session snapshot, resolved through its stable turn hook."""
    run_id = await stream.session_run_id(session_id)
    if run_id is None:
        return None
    return await stream.read_session(run_id)


async def wait_for_lifecycle(
    session_id: str, type_: str, *, count: int = 1, timeout: float = 30
) -> None:
    async def watch() -> None:
        while True:
            run_id = await stream.session_run_id(session_id)
            if run_id is not None:
                seen = 0
                async for event in stream.replay(run_id):
                    if isinstance(event, proto.LifecycleEvent) and event.type == type_:
                        seen += 1
                        if seen >= count:
                            return
            await asyncio.sleep(0.02)

    await asyncio.wait_for(watch(), timeout)


async def wait_for_event(
    session_id: str, event_type: type[Any], *, count: int = 1, timeout: float = 30
) -> None:
    async def watch() -> None:
        while True:
            run_id = await stream.session_run_id(session_id)
            if run_id is not None:
                seen = 0
                async for event in stream.replay(run_id):
                    if isinstance(event, event_type):
                        seen += 1
                        if seen >= count:
                            return
            await asyncio.sleep(0.02)

    await asyncio.wait_for(watch(), timeout)


async def resume_session(token: str, payload: proto.NewUserMessage) -> None:
    await _resume_hook(token, proto.SessionHook(payload=payload))


async def resume_approval(
    session_id: str, response: proto.ToolApprovalResponse
) -> None:
    await _resume_hook(
        proto.approval_hook_token(session_id, response.tool_call_id),
        proto.ApprovalHook(response=response),
    )


async def _resume_hook(token: str, hook: vercel.workflow.BaseHook) -> None:
    async for last_attempt in util.hook_retries(retry_count=100):
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if last_attempt:
                raise


async def wait_for_hook(token: str) -> vercel.workflow.Hook:
    async for last_attempt in util.hook_retries(retry_count=100):
        try:
            return await vercel.workflow.get_hook_by_token(token)
        except vercel.workflow.HookNotFoundError:
            if last_attempt:
                raise
    raise AssertionError("unreachable")


async def wait_run[T](run: vercel.workflow.Run[T], timeout: float = 20) -> T:
    async def poll() -> T:
        while await run.status() not in ("completed", "failed", "cancelled"):
            await asyncio.sleep(0.05)
        return await run.return_value()

    return await asyncio.wait_for(poll(), timeout)


async def lifecycle(session_id: str) -> list[str]:
    # a completed child has already lost its hook and no longer advertises a run.
    run_id = await stream.session_run_id(session_id)
    if run_id is None:
        return []
    return [
        event.type
        async for event in stream.replay(run_id)
        if isinstance(event, proto.LifecycleEvent)
    ]
