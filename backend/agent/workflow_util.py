from __future__ import annotations

import asyncio
import dataclasses
from typing import Protocol, cast

import vercel.workflow
from vercel.workflow._internal import core


class HookLabels[**Params](Protocol):
    def __call__(
        self, run_id: str, *args: Params.args, **kwargs: Params.kwargs
    ) -> list[str]: ...


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
