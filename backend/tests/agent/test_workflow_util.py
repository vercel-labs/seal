from __future__ import annotations

import types
from typing import Any, cast

import pytest
import vercel.workflow

import agent.driver as driver
import agent.proto as proto
import agent.workflow_util as workflow_util


def fake_run(run_id: str, status: str = "running") -> vercel.workflow.Run[None]:
    async def get_status() -> str:
        return status

    return cast(
        vercel.workflow.Run[None],
        types.SimpleNamespace(run_id=run_id, status=get_status),
    )


def test_with_hooks_accepts_static_or_generated_labels() -> None:
    static = workflow_util.with_hooks(driver.run_session, ["one", "two"])
    generated = workflow_util.with_hooks(
        driver.run_session,
        lambda run_id, session_input: [f"{run_id}:{session_input.session_id}"],
    )
    session_input = proto.SessionInput(session_id="s1", prompt="hello")

    assert static.hook_labels("ignored", session_input) == ["one", "two"]
    assert generated.hook_labels("run-1", session_input) == ["run-1:s1"]


def test_with_hooks_collapses_existing_hook_wrapper() -> None:
    first = workflow_util.with_hooks(driver.run_session, ["one"])
    combined = workflow_util.with_hooks(
        first,
        lambda run_id, session_input: [f"{run_id}:{session_input.prompt}"],
        timeout=5,
    )
    session_input = proto.SessionInput(session_id="s1", prompt="hello")

    assert combined.workflow is driver.run_session
    assert combined.hook_labels("run-1", session_input) == ["one", "run-1:hello"]
    assert combined.timeout == 5


async def test_start_passes_through_plain_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = fake_run("run-1")

    async def fake_start(*args: Any, **kwargs: Any) -> vercel.workflow.Run[None]:
        return expected

    monkeypatch.setattr(vercel.workflow, "start", fake_start)

    run = await workflow_util.start(
        driver.run_session,
        proto.SessionInput(session_id="s1", prompt="hello"),
    )

    assert run is expected


async def test_start_retries_until_all_declared_hooks_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = fake_run("run-1")
    attempts: dict[str, int] = {}

    async def fake_start(*args: Any, **kwargs: Any) -> vercel.workflow.Run[None]:
        return expected

    async def fake_get_hook(label: str) -> vercel.workflow.Hook:
        attempts[label] = attempts.get(label, 0) + 1
        run_id = "old-run" if attempts[label] == 1 else "run-1"
        return cast(
            vercel.workflow.Hook,
            types.SimpleNamespace(run_id=run_id),
        )

    monkeypatch.setattr(vercel.workflow, "start", fake_start)
    monkeypatch.setattr(vercel.workflow, "get_hook_by_token", fake_get_hook)
    wrapped = workflow_util.with_hooks(
        driver.run_session,
        lambda run_id, session_input: [
            f"{run_id}:{session_input.session_id}:one",
            f"{run_id}:{session_input.session_id}:two",
        ],
    )

    run = await workflow_util.start(
        wrapped,
        proto.SessionInput(session_id="s1", prompt="hello"),
    )

    assert run is expected
    assert attempts == {"run-1:s1:one": 2, "run-1:s1:two": 2}


async def test_start_times_out_waiting_for_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = fake_run("run-1")

    async def fake_start(*args: Any, **kwargs: Any) -> vercel.workflow.Run[None]:
        return expected

    monkeypatch.setattr(vercel.workflow, "start", fake_start)
    wrapped = workflow_util.with_hooks(driver.run_session, ["missing"], timeout=0)

    with pytest.raises(TimeoutError, match=r"run-1.*missing"):
        await workflow_util.start(
            wrapped,
            proto.SessionInput(session_id="s1", prompt="hello"),
        )


@pytest.mark.parametrize("status", ["completed", "failed"])
async def test_start_stops_waiting_when_workflow_finishes(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    expected = fake_run("run-1", status=status)

    async def fake_start(*args: Any, **kwargs: Any) -> vercel.workflow.Run[None]:
        return expected

    async def missing_hook(label: str) -> vercel.workflow.Hook:
        raise vercel.workflow.HookNotFoundError(token=label)

    monkeypatch.setattr(vercel.workflow, "start", fake_start)
    monkeypatch.setattr(vercel.workflow, "get_hook_by_token", missing_hook)
    wrapped = workflow_util.with_hooks(driver.run_session, ["missed"])

    run = await workflow_util.start(
        wrapped,
        proto.SessionInput(session_id="s1", prompt="hello"),
    )

    assert run is expected
