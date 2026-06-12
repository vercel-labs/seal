"""In-process runner for the durable workflow engine.

``run_session`` executes under the workflow SDK's LocalWorld with one change:
``world.queue`` dispatches to the workflow/step handlers as asyncio tasks
instead of going through the queue service. Everything else is real — replay,
suspensions, workflow hooks, the jsonl store, the bash subprocess.

Workflow invocations of one run stay serialized (each invocation replays the
whole body, and the previous invocation's abandoned generators must be
finalized first). Step invocations are deliberately NOT serialized:
production delivery runs them concurrently, and serializing them here once
masked a production-only deadlock.

The harness also watches engine health (``check_*`` methods, run from the
``world`` fixture's teardown or called explicitly by tests):

* ``check_settled`` — every step reached a terminal state and every run
  either finished or is parked on a hook. A step left ``running`` is the
  signature of the production wedge: a dropped step invocation can never be
  retried with ``max_retries = 0``.
* ``check_replay_determinism`` — every replay pass of a run must request the
  same steps/hooks in the same order. The engine matches results to requests
  purely by position, so a reordered replay silently wires results to the
  wrong calls (and the colliding step_created is swallowed as a conflict).

Concurrent steps of one run are expected and healthy (that's what lets two
tools execute side by side); only the two checks above are laws.
"""

from __future__ import annotations

import asyncio
import contextvars
import gc
import itertools
import json
import re
from typing import Any

import vercel._internal.workflow.py_sandbox as py_sandbox

# mirror worker.py: these modules must come from the host inside the sandbox.
py_sandbox._PASSTHROUGHS.update({"ai", "pathlib"})

import vercel._internal.workflow.runtime as wf_runtime  # noqa: E402
import vercel._internal.workflow.world as wf_world  # noqa: E402
import vercel._internal.workflow.worlds.local as wf_local  # noqa: E402
import vercel.workflow  # noqa: E402

import agent.driver as driver  # noqa: E402
from agent import proto, stream  # noqa: E402

RUN_TERMINAL = ("completed", "failed", "cancelled")
STEP_TERMINAL = ("completed", "failed")

# ids the app generates fresh inside the workflow body on every replay pass.
_VOLATILE_ID = re.compile(r"\b(?:msg|part|turn)_[0-9a-f]+\b")


def _canon_pass(entries: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Replace generated ids with stable placeholders, in encounter order."""
    mapping: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        return mapping.setdefault(match.group(0), f"id-{len(mapping)}")

    return [
        tuple(_VOLATILE_ID.sub(replace, field) for field in entry) for entry in entries
    ]


# the (world, run_id) of the delivery currently executing in this task; lets
# the orchestrator patches below attribute suspension requests to a run.
_DELIVERY: contextvars.ContextVar[tuple[InProcessWorld, str] | None] = (
    contextvars.ContextVar("seal_delivery", default=None)
)


class InProcessWorld(wf_local.LocalWorld):
    """LocalWorld with the queue bridged to in-process handler dispatch."""

    def __init__(self, registry: Any) -> None:
        super().__init__()
        self._registry = registry
        self._tasks: set[asyncio.Task[None]] = set()
        self._locks: dict[str, asyncio.Lock] = {}
        self._ids = itertools.count()
        self.errors: list[BaseException] = []
        # run_id -> one entry per replay pass, each the ordered list of
        # suspension requests (steps/hooks/waits) that pass made.
        self.suspension_traces: dict[str, list[list[tuple[str, ...]]]] = {}

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
            run_id = getattr(message, "run_id", None) or message.workflow_run_id
            _DELIVERY.set((self, run_id))
            if queue_name.startswith("__wkf_workflow_"):
                handler = wf_runtime.workflow_handler
                # workflow replays of one run must not interleave: every
                # invocation replays the same body from scratch.
                lock = self._locks.setdefault(run_id, asyncio.Lock())
            elif queue_name.startswith("__wkf_step_"):
                handler = wf_runtime.step_handler
                # steps are intentionally unserialized — see module docstring.
                lock = None
            else:
                raise RuntimeError(f"unexpected queue: {queue_name}")
            attempt = 1
            while True:
                if lock is None:
                    retry = await handler(
                        message.model_dump(),
                        attempt=attempt,
                        queue_name=queue_name,
                        message_id=message_id,
                        registry=self._registry,
                    )
                else:
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

    async def drain(self, timeout: float = 15) -> None:
        """Wait until every delivery (and the ones it spawns) has finished."""

        async def settle() -> None:
            while self._tasks:
                await asyncio.wait(list(self._tasks))

        try:
            await asyncio.wait_for(settle(), timeout)
        except TimeoutError:
            pending = [str(task) for task in self._tasks]
            for task in list(self._tasks):
                task.cancel()
            raise AssertionError(
                f"deliveries still in flight after {timeout}s: {pending}"
            ) from None

    # --- engine health checks (read the per-test .workflow-data store) ---------

    def check_settled(self) -> None:
        """No work left hanging: steps terminal, runs finished or on a hook."""
        failures = []
        for step in self._steps():
            if step.status not in STEP_TERMINAL:
                failures.append(
                    f"step {step.step_name!r} ({step.step_id}) of run "
                    f"{step.run_id} is stuck in {step.status!r}"
                )
        parked = self._runs_with_hooks()
        for run in self._runs():
            if run.status in RUN_TERMINAL or run.run_id in parked:
                continue
            failures.append(
                f"run {run.run_id} ({run.workflow_name}) is {run.status!r} "
                "with no hook to wake it — this is a wedged run"
            )
        assert not failures, "engine left work hanging:\n" + "\n".join(failures)

    def check_replay_determinism(self) -> None:
        """Consecutive replay passes must request the same things in order.

        The engine correlates a pass's requests with the event log purely by
        position; a divergent pass attaches logged results to the wrong
        requests without any error.

        Generated ids (msg_*/part_*) are canonicalized before comparing: the
        workflow body mints fresh ones on every pass, and the engine discards
        re-issued inputs for already-logged requests, so they are noise.
        """
        for run_id, passes in self.suspension_traces.items():
            canonical = [_canon_pass(entries) for entries in passes]
            for index in range(1, len(canonical)):
                previous, current = canonical[index - 1], canonical[index]
                for pos in range(min(len(previous), len(current))):
                    assert previous[pos] == current[pos], (
                        f"run {run_id} replay diverged at request #{pos}: "
                        f"pass {index - 1} made {previous[pos]!r}, "
                        f"pass {index} made {current[pos]!r}"
                    )

    # --- store readers ----------------------------------------------------------

    def _runs(self) -> list[Any]:
        return self._read_dir("runs", wf_world.WorkflowRunAdaptor)

    def _steps(self) -> list[Any]:
        return self._read_dir("steps", wf_world.WorkflowStepAdaptor)

    def _runs_with_hooks(self) -> set[str]:
        hooks = self._read_dir("hooks", wf_world.Hook)
        return {hook.run_id for hook in hooks}

    def _read_dir(self, name: str, schema: Any) -> list[Any]:
        directory = self.data_dir / name
        if not directory.exists():
            return []
        records = [
            wf_local.read_json(path, schema)
            for path in sorted(directory.iterdir())
            if path.suffix == ".json"
        ]
        return [record for record in records if record is not None]


# --- replay tracing ---------------------------------------------------------------
# The orchestrator gives no visibility into the order a replay pass requests
# its steps/hooks in, and order bugs leave no trace in the event log (the
# colliding step_created is swallowed as an EntityConflictError). Record every
# pass's requests here so check_replay_determinism can compare them.

_original_init = wf_runtime.WorkflowOrchestratorContext.__init__
_original_run_step = wf_runtime.WorkflowOrchestratorContext.run_step
_original_create_hook = wf_runtime.WorkflowOrchestratorContext.create_hook
_original_run_wait = wf_runtime.WorkflowOrchestratorContext.run_wait


def _tracing_init(self: Any, *args: Any, **kwargs: Any) -> None:
    _original_init(self, *args, **kwargs)
    delivery = _DELIVERY.get()
    if delivery is not None:
        world, run_id = delivery
        trace: list[tuple[str, ...]] = []
        world.suspension_traces.setdefault(run_id, []).append(trace)
        self._seal_trace = trace


async def _tracing_run_step(self: Any, step: Any, *args: Any, **kwargs: Any) -> Any:
    trace = getattr(self, "_seal_trace", None)
    if trace is not None:
        trace.append(("step", step.name, json.dumps((args, kwargs), sort_keys=True)))
    return await _original_run_step(self, step, *args, **kwargs)


def _tracing_create_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
    event = _original_create_hook(self, *args, **kwargs)
    trace = getattr(self, "_seal_trace", None)
    if trace is not None:
        trace.append(("hook", event._token))
    return event


async def _tracing_run_wait(self: Any, param: Any) -> None:
    trace = getattr(self, "_seal_trace", None)
    if trace is not None:
        trace.append(("wait", str(param)))
    return await _original_run_wait(self, param)


for _name, _wrapper in (
    ("__init__", _tracing_init),
    ("run_step", _tracing_run_step),
    ("create_hook", _tracing_create_hook),
    ("run_wait", _tracing_run_wait),
):
    setattr(wf_runtime.WorkflowOrchestratorContext, _name, _wrapper)


async def start_session(session_id: str, prompt: str) -> Any:
    return await vercel.workflow.start(
        driver.run_session,
        proto.SessionInput(session_id=session_id, prompt=prompt).model_dump(
            mode="json"
        ),
    )


async def wait_for_lifecycle(
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


async def resume_session(token: str, payload: proto.ResumePayload) -> None:
    hook = proto.SessionHook(payload=payload)
    for attempt in range(100):
        try:
            await hook.resume(token)
            return
        except RuntimeError as error:
            if attempt == 99 or "not found" not in str(error).lower():
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
