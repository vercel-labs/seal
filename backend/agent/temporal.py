from __future__ import annotations

import asyncio
import os

import temporalio.client
import temporalio.contrib.pydantic
import temporalio.exceptions
import temporalio.service

from agent import TASK_QUEUE, driver, proto, session_workflow_id

_client: temporalio.client.Client | None = None
_client_lock = asyncio.Lock()
DATA_CONVERTER = temporalio.contrib.pydantic.pydantic_data_converter


async def get_client() -> temporalio.client.Client:
    global _client
    async with _client_lock:
        if _client is None:
            _client = await temporalio.client.Client.connect(
                os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"),
                namespace=os.environ.get("TEMPORAL_NAMESPACE", "default"),
                data_converter=DATA_CONVERTER,
            )
    return _client


async def start_or_resume(session_id: str, prompt: str) -> None:
    client = await get_client()
    workflow_id = session_workflow_id(session_id)
    try:
        await client.start_workflow(
            driver.SessionWorkflow.run,
            proto.SessionInput(session_id=session_id, prompt=prompt),
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
    except temporalio.exceptions.WorkflowAlreadyStartedError:
        await client.get_workflow_handle(workflow_id).signal(
            driver.SessionWorkflow.new_message,
            proto.NewUserMessage(prompt=prompt),
        )


async def submit_approvals(
    session_id: str, approvals: list[proto.ToolApprovalResponse]
) -> None:
    await (
        (await get_client())
        .get_workflow_handle(session_workflow_id(session_id))
        .signal(
            driver.SessionWorkflow.approvals,
            proto.ApprovalSignal(responses=approvals),
        )
    )


async def interrupt(session_id: str) -> None:
    await (
        (await get_client())
        .get_workflow_handle(session_workflow_id(session_id))
        .signal(driver.SessionWorkflow.interrupt)
    )


async def terminate(session_id: str) -> None:
    await (
        (await get_client())
        .get_workflow_handle(session_workflow_id(session_id))
        .terminate("session deleted")
    )


async def session_state(session_id: str) -> proto.SessionState | None:
    try:
        return await (
            (await get_client())
            .get_workflow_handle(session_workflow_id(session_id))
            .query(driver.SessionWorkflow.get_state)
        )
    except temporalio.service.RPCError as error:
        if error.status == temporalio.service.RPCStatusCode.NOT_FOUND:
            return None
        raise
