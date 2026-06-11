"""Test-only workflow that runs a single ``run_turn`` and returns its output.

``run_turn`` resumes a workflow hook when it finishes, so it cannot be run
alone: someone must own that hook or the resume retries time out. In the app
that someone is ``run_session``; this wrapper is the minimal stand-in, giving
turn-level tests a direct seam without driving a whole session around it.

This module is imported (and registered on the shared workflow registry) only
by tests; the production worker never sees it.
"""

from __future__ import annotations

from typing import Any

import agent.driver as driver
from agent import proto, workflow


@workflow.workflow
async def capture_turn(turn_input: dict[str, Any]) -> dict[str, Any]:
    hook = proto.TurnHook.wait(token=turn_input["turn_hook_token"])
    await driver.spawn_turn_workflow(turn_input)
    resolution = await hook
    hook.dispose()
    assert resolution is not None
    return resolution.output.model_dump(mode="json")
