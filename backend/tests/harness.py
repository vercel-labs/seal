"""In-process runner for the durable workflow engine.

``run_session`` executes under the workflow SDK's LocalWorld with one change:
``world.queue`` dispatches to the workflow/step handlers as asyncio tasks
instead of going through the queue service. Everything else is real — replay,
suspensions, workflow hooks, the jsonl store, the bash subprocess.
"""

from __future__ import annotations

import asyncio
import itertools
import random
from typing import Any

import ai
import vercel._internal.workflow.runtime as wf_runtime  # noqa: E402
import vercel._internal.workflow.worlds.local as wf_local  # noqa: E402
import vercel.workflow  # noqa: E402

import agent.driver as driver  # noqa: E402
from agent import proto, stream  # noqa: E402


class InProcessWorld(wf_local.LocalWorld):
    """LocalWorld with the queue bridged to in-process handler dispatch."""

    def __init__(self, registry: Any) -> None:
        super().__init__()
        self._registry = registry
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
        self.monotonic_ulid = self._det_ulid
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
            # Install a seeded RNG for steps so they get deterministic IDs also.
            rng_ctx = ai.messages.use_random(self._step_rng)
            attempt = 1
            while True:
                async with lock:
                    with rng_ctx:
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


async def start_session(session_id: str, prompt: str) -> Any:
    return await vercel.workflow.start(
        driver.run_session,
        proto.SessionInput(session_id=session_id, prompt=prompt).model_dump(
            mode="json"
        ),
    )


async def wait_for_lifecycle(
    session_id: str, type_: str, *, count: int = 1, timeout: float = 30
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


async def resume_session(token: str, payload: proto.ResumePayload) -> None:
    hook = proto.SessionHook(payload=payload)
    for attempt in range(100):
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if attempt == 99:
                raise
            await asyncio.sleep(0.05)


async def wait_run(run: Any, timeout: float = 20) -> Any:
    async def poll() -> Any:
        while await run.status() not in ("completed", "failed", "cancelled"):
            await asyncio.sleep(0.05)
        return await run.return_value()

    return await asyncio.wait_for(poll(), timeout)


async def lifecycle(session_id: str) -> list[str]:
    return [
        event.type
        async for event in stream.replay(session_id)
        if isinstance(event, proto.LifecycleEvent)
    ]
