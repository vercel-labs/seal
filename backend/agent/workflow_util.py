from __future__ import annotations

import abc
import asyncio
import dataclasses
import functools
import inspect
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, Protocol, TypeVar, cast

import vercel.workflow
from vercel.workflow._internal import core

HookFunction = TypeVar("HookFunction", bound=Callable[..., Awaitable[None]])
HookToken = Callable[[Any], str]


class HookLabels[**Params](Protocol):
    def __call__(
        self, run_id: str, *args: Params.args, **kwargs: Params.kwargs
    ) -> list[str]: ...


_HOOK_TYPE = "__workflow_hook_type__"
_HOOK_TOKEN = "__workflow_hook_token__"
_WORKFLOW_INIT = "__workflow_init__"


def hook(
    hook_type: type[vercel.workflow.BaseHook],
    token: HookToken | None = None,
) -> Callable[[HookFunction], HookFunction]:
    """Mark a hook handler, optionally setting its token from the instance."""
    if not inspect.isclass(hook_type) or not issubclass(
        hook_type, vercel.workflow.BaseHook
    ):
        raise TypeError("hook() expects a BaseHook subclass")
    if token is not None and not callable(token):
        raise TypeError("hook token must be a callable")

    def decorate(func: HookFunction) -> HookFunction:
        if not inspect.iscoroutinefunction(func):
            raise TypeError("hook handlers must be async functions")
        setattr(func, _HOOK_TYPE, hook_type)
        setattr(func, _HOOK_TOKEN, token)
        return cast(HookFunction, func)

    return decorate


def init[F: Callable[..., None]](func: F) -> F:
    """Mark the real initializer to receive the workflow arguments."""
    if func.__name__ != "__init__":  # ty: ignore[unresolved-attribute]
        raise TypeError("@init may only decorate __init__")
    if inspect.iscoroutinefunction(func):
        raise TypeError("@init must decorate a synchronous function")
    setattr(func, _WORKFLOW_INIT, True)
    return func


def hook_token(run_id: str, hook_name: str) -> str:
    return run_id + "$signal$" + hook_name


@dataclasses.dataclass(frozen=True)
class WorkflowWithHooks[**Params, Result]:
    workflow: core.Workflow[Params, Result]
    hook_labels: HookLabels[Params]
    timeout: float | None = 30


type WorkflowLike[**Params, Result] = (
    core.Workflow[Params, Result] | WorkflowWithHooks[Params, Result]
)


def with_hooks[**Params, Result](
    workflow: WorkflowLike[Params, Result],
    hooks: HookLabels[Params] | list[str],
    *,
    timeout: float | None = 30,
) -> WorkflowWithHooks[Params, Result]:
    """Describe the hooks that must exist before a started run is ready."""
    new_hook_labels: HookLabels[Params]
    if isinstance(hooks, list):
        labels = [
            cast(str, label)  # type: ignore[redundant-cast]
            for label in hooks
        ]

        def static_hook_labels(
            run_id: str, *args: Params.args, **kwargs: Params.kwargs
        ) -> list[str]:
            return list(labels)

        new_hook_labels = static_hook_labels
    else:
        new_hook_labels = hooks

    hook_labels: HookLabels[Params]
    if isinstance(workflow, WorkflowWithHooks):
        existing_hook_labels = workflow.hook_labels

        def combined(
            run_id: str, *args: Params.args, **kwargs: Params.kwargs
        ) -> list[str]:
            return existing_hook_labels(run_id, *args, **kwargs) + new_hook_labels(
                run_id, *args, **kwargs
            )

        registered = workflow.workflow
        hook_labels = combined
    else:
        registered = workflow
        hook_labels = new_hook_labels

    return WorkflowWithHooks(
        workflow=registered, hook_labels=hook_labels, timeout=timeout
    )


async def start[**Params, Result](
    workflow: WorkflowLike[Params, Result],
    *args: Params.args,
    **kwargs: Params.kwargs,
) -> vercel.workflow.Run[Result]:
    """Start a workflow, waiting until its declared hooks are available."""
    if isinstance(workflow, WorkflowWithHooks):
        registered = workflow.workflow
        hook_labels = workflow.hook_labels
        hook_timeout = workflow.timeout
    else:
        registered = workflow
        hook_labels = None
        hook_timeout = None

    run = await vercel.workflow.start(registered, *args, **kwargs)
    if hook_labels is None:
        return run

    pending = set(hook_labels(run.run_id, *args, **kwargs))
    try:
        async with asyncio.timeout(hook_timeout):
            while pending:
                found: list[str] = []
                for label in pending:
                    try:
                        hook = await vercel.workflow.get_hook_by_token(label)
                    except vercel.workflow.HookNotFoundError:
                        continue
                    if hook.run_id == run.run_id:
                        found.append(label)
                pending.difference_update(found)
                if pending:
                    if await run.status() in ("completed", "failed", "cancelled"):
                        return run
                    await asyncio.sleep(0.05)
    except TimeoutError:
        labels = ", ".join(sorted(pending))
        raise TimeoutError(
            f"workflow {run.run_id} did not create hooks within "
            f"{hook_timeout}s: {labels}"
        ) from None
    return run


def _hook_methods(
    cls: type[Any],
) -> tuple[
    tuple[str, type[vercel.workflow.BaseHook], HookToken | None],
    ...,
]:
    methods: list[tuple[str, type[vercel.workflow.BaseHook], HookToken | None]] = []
    seen: set[str] = set()
    for base in cls.__mro__:
        for name, member in vars(base).items():
            if name in seen:
                continue
            seen.add(name)
            hook_type = getattr(member, _HOOK_TYPE, None)
            if hook_type is not None:
                methods.append(
                    (
                        name,
                        cast(type[vercel.workflow.BaseHook], hook_type),
                        cast(HookToken | None, getattr(member, _HOOK_TOKEN, None)),
                    )
                )
    return tuple(methods)


class HasRun[**Params, Result_co](Protocol):
    workflow: object
    _workflow_with_hooks: object

    def run(
        self, *args: Params.args, **kwargs: Params.kwargs
    ) -> Coroutine[Any, Any, Result_co]: ...


class WorkflowClass(abc.ABC):
    workflow: object
    _workflow_with_hooks: object
    _active_hooks: list[vercel.workflow.HookEvent[Any]]

    def __init_subclass__(
        cls, *, registry: vercel.workflow.Workflows, **kwargs: Any
    ) -> None:
        super().__init_subclass__(**kwargs)
        prototype = object.__new__(cls)
        bound_run = prototype.run
        init_accepts_workflow_args = hasattr(cls.__init__, _WORKFLOW_INIT)
        hook_methods = _hook_methods(cls)

        def instantiate(*args: Any, **kwargs: Any) -> WorkflowClass:
            return cls(*args, **kwargs) if init_accepts_workflow_args else cls()

        def labels_for(instance: WorkflowClass, run_id: str) -> list[str]:
            labels: list[str] = []
            for func_name, _, token_for in hook_methods:
                token = (
                    token_for(instance)
                    if token_for is not None
                    else hook_token(run_id, func_name)
                )
                if not isinstance(token, str):
                    raise TypeError(f"hook token for {func_name} must be a string")
                labels.append(token)
            return labels

        @functools.wraps(bound_run)
        async def generated_workflow(*args: Any, **kwargs: Any) -> Any:
            instance = instantiate(*args, **kwargs)
            instance._active_hooks = []
            run_id = vercel.workflow.get_workflow_metadata().run_id

            async with asyncio.TaskGroup() as tasks:
                labels = labels_for(instance, run_id)
                for (func_name, hook_type, _), token in zip(
                    hook_methods, labels, strict=True
                ):
                    events = hook_type.wait(
                        token=token,
                        metadata={
                            "kind": "hook",
                            "handler": func_name,
                            "name": func_name,
                        },
                    )
                    instance._active_hooks.append(events)
                    tasks.create_task(
                        instance._listen_for_hook(func_name, events),
                        name=token,
                    )

                try:
                    return await instance.run(*args, **kwargs)
                finally:
                    instance.dispose_all()

        generated_workflow.__name__ = "workflow"
        generated_workflow.__qualname__ = f"{cls.__qualname__}.workflow"
        registered = registry.workflow(generated_workflow)

        def registered_hook_labels(run_id: str, *args: Any, **kwargs: Any) -> list[str]:
            return labels_for(instantiate(*args, **kwargs), run_id)

        cls.workflow = registered
        cls._workflow_with_hooks = with_hooks(registered, registered_hook_labels)

    async def _listen_for_hook(
        self,
        func_name: str,
        events: vercel.workflow.HookEvent[Any],
    ) -> None:
        async for payload in events:
            handler = cast(
                Callable[[vercel.workflow.BaseHook], Awaitable[None]],
                getattr(self, func_name),
            )
            await handler(payload)

    def dispose_all(self) -> None:
        """Dispose every registered hook."""
        for events in self._active_hooks:
            events.dispose()
        self._active_hooks.clear()

    @classmethod
    def registered_workflow[**Params, Result](
        cls: type[HasRun[Params, Result]],
    ) -> WorkflowWithHooks[Params, Result]:
        return cast(WorkflowWithHooks[Params, Result], cls._workflow_with_hooks)

    @abc.abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
