"""Durable Seal agent workflow."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any, Literal, cast

import vercel.workflow as workflow

if TYPE_CHECKING:
    import ai


TOOL_APPROVAL_HOOK_TYPE = "ToolApproval"
APPROVAL_REQUIRED_TOOLS = frozenset({"bash", "web_fetch"})

workflow_app = workflow.Workflows()


@dataclasses.dataclass
class ToolApprovalHook(workflow.BaseHook):
    granted: bool
    reason: str | None = None


def dump_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], message.model_dump(mode="json")) for message in messages
    ]


def load_messages(data: list[dict[str, Any]]) -> list[ai.messages.Message]:
    import ai

    return [ai.messages.Message.model_validate(item) for item in data]


def dump_message(message: Any) -> dict[str, Any]:
    return cast(dict[str, Any], message.model_dump(mode="json"))


def load_message(data: dict[str, Any]) -> ai.messages.Message:
    import ai

    return ai.messages.Message.model_validate(data)


def tool_requires_approval(tool_name: str) -> bool:
    return tool_name in APPROVAL_REQUIRED_TOOLS


def _message_role(message: dict[str, Any]) -> str:
    return str(message["role"])


def _message_id(message: dict[str, Any]) -> str:
    return str(message["id"])


def _turn_id_for_data(assistant_message: dict[str, Any]) -> str:
    turn_id = assistant_message.get("turn_id")
    return str(turn_id or _message_id(assistant_message))


def _tool_call_parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        part
        for part in message.get("parts", [])
        if isinstance(part, dict) and part.get("kind") == "tool_call"
    ]


def _tool_call_id(tool_call: dict[str, Any]) -> str:
    return str(tool_call["tool_call_id"])


def _tool_name(tool_call: dict[str, Any]) -> str:
    return str(tool_call["tool_name"])


def _tool_args(tool_call: dict[str, Any]) -> str:
    raw = tool_call.get("tool_args")
    return raw if isinstance(raw, str) else "{}"


def _approval_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        tool_call_id = _tool_call_id(tool_call)
    else:
        tool_call_id = str(tool_call.tool_call_id)
    return f"approve_{tool_call_id}"


def _tool_call_data(tool_call: Any) -> dict[str, Any]:
    return (
        tool_call
        if isinstance(tool_call, dict)
        else tool_call.model_dump(mode="json")
    )


def _message_data(message: Any) -> dict[str, Any]:
    return message if isinstance(message, dict) else message.model_dump(mode="json")


def _hook_message_data(
    *,
    tool_call: dict[str, Any],
    assistant_message: dict[str, Any],
    status: Literal["pending", "resolved", "cancelled"],
    resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    approval_id = _approval_id(tool_call)
    try:
        kwargs = json.loads(_tool_args(tool_call) or "{}")
    except json.JSONDecodeError:
        kwargs = {}
    return {
        "id": f"internal_{approval_id}_{status}",
        "turn_id": _turn_id_for_data(assistant_message),
        "role": "internal",
        "parts": [
            {
                "id": f"hook_{approval_id}_{status}",
                "hook_id": approval_id,
                "hook_type": TOOL_APPROVAL_HOOK_TYPE,
                "status": status,
                "metadata": {
                    "tool": _tool_name(tool_call),
                    "kwargs": kwargs,
                },
                "resolution": resolution,
                "kind": "hook",
            }
        ],
        "usage": None,
        "provider_metadata": None,
    }


def _hook_message(
    *,
    tool_call: ai.messages.ToolCallPart,
    assistant_message: ai.messages.Message,
    status: Literal["pending", "resolved", "cancelled"],
    resolution: dict[str, Any] | None = None,
) -> ai.messages.Message:
    import ai

    return ai.messages.Message.model_validate(
        _hook_message_data(
            tool_call=_tool_call_data(tool_call),
            assistant_message=_message_data(assistant_message),
            status=status,
            resolution=resolution,
        )
    )


def _denied_tool_result(
    tool_call: Any,
    reason: str | None,
) -> dict[str, Any]:
    tool_call_data = _tool_call_data(tool_call)
    tool_call_id = _tool_call_id(tool_call_data)
    tool_name = _tool_name(tool_call_data)
    return {
        "kind": "tool_result",
        "id": f"result_{tool_call_id}_denied",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "is_error": True,
        "is_hook_pending": False,
        "provider_metadata": None,
        "result": f"Rejected: {reason or 'not approved'}",
    }


def _tool_message_data(
    *,
    assistant_message: dict[str, Any],
    parts_data: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": f"tool_{_message_id(assistant_message)}",
        "turn_id": _turn_id_for_data(assistant_message),
        "role": "tool",
        "parts": parts_data,
        "usage": None,
        "provider_metadata": None,
    }


def _tool_message(
    *,
    assistant_message: ai.messages.Message,
    parts_data: list[dict[str, Any]],
) -> ai.messages.Message:
    import ai

    parts: list[ai.messages.Part] = [
        ai.messages.ToolResultPart.model_validate(item) for item in parts_data
    ]
    return ai.messages.Message.model_validate(
        _tool_message_data(
            assistant_message=_message_data(assistant_message),
            parts_data=[part.model_dump(mode="json") for part in parts],
        )
    )


@workflow_app.step
async def llm_step(messages_data: list[dict[str, Any]]) -> dict[str, Any]:
    import ai

    import agent
    import attachments

    messages = [
        ai.messages.Message.model_validate(message)
        for message in messages_data
        if _message_role(message) not in ("system", "internal")
    ]
    messages = await attachments.inline_file_parts(messages)
    full_messages = [ai.system_message(agent.SYSTEM), *messages]

    async with ai.stream(
        agent.get_model(),
        full_messages,
        tools=[tool.tool for tool in agent.TOOLS],
    ) as stream:
        async for _ in stream:
            pass
        return dump_message(stream.message)


@workflow_app.step
async def execute_tool_step(tool_call_data: dict[str, Any]) -> dict[str, Any]:
    import ai

    import agent

    tool_call = ai.messages.ToolCallPart.model_validate(tool_call_data)
    tool = next(
        (tool for tool in agent.TOOLS if tool.name == tool_call.tool_name),
        None,
    )
    if tool is None:
        return ai.messages.ToolResultPart(
            id=f"result_{tool_call.tool_call_id}_missing",
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            result=f"Unknown tool: {tool_call.tool_name}",
            is_error=True,
        ).model_dump(mode="json")

    try:
        raw_kwargs = json.loads(tool_call.tool_args or "{}")
        kwargs = raw_kwargs
        if tool.validator is not None:
            validated = tool.validator.model_validate(raw_kwargs)
            kwargs = {
                field: getattr(validated, field)
                for field in type(validated).model_fields
            }
        result = await tool.fn(**kwargs)
        return ai.messages.ToolResultPart(
            id=f"result_{tool_call.tool_call_id}",
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            result=result,
            is_error=False,
        ).model_dump(mode="json")
    except Exception as exc:
        return ai.messages.ToolResultPart(
            id=f"result_{tool_call.tool_call_id}_error",
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            result=f"{type(exc).__name__}: {exc}",
            is_error=True,
        ).model_dump(mode="json")


execute_tool_step.max_retries = 0


@workflow_app.step
async def persist_messages_step(
    session_id: str,
    messages_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import sessions

    messages = load_messages(messages_data)
    await sessions.save_messages(session_id, messages)
    return messages_data


@workflow_app.step
async def clear_active_run_step(session_id: str) -> None:
    import sessions

    await sessions.set_active_run(session_id, None)


async def _persist(
    session_id: str,
    messages: list[dict[str, Any]],
) -> None:
    await persist_messages_step(session_id, messages)


@workflow_app.workflow
async def run_agent(
    session_id: str,
    messages_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages = messages_data

    while messages and _message_role(messages[-1]) != "assistant":
        assistant_message = await llm_step(messages)
        messages.append(assistant_message)
        await _persist(session_id, messages)

        tool_results: list[dict[str, Any]] = []
        for tool_call in _tool_call_parts(assistant_message):
            if tool_requires_approval(_tool_name(tool_call)):
                approval_id = _approval_id(tool_call)
                approval_event = ToolApprovalHook.wait(token=approval_id)
                messages.append(
                    _hook_message_data(
                        tool_call=tool_call,
                        assistant_message=assistant_message,
                        status="pending",
                    )
                )
                await _persist(session_id, messages)

                approval = await approval_event
                granted = bool(approval and approval.granted)
                reason = approval.reason if approval else "cancelled"
                resolution = {
                    "granted": granted,
                    "reason": reason,
                }
                messages.append(
                    _hook_message_data(
                        tool_call=tool_call,
                        assistant_message=assistant_message,
                        status="resolved",
                        resolution=resolution,
                    )
                )
                await _persist(session_id, messages)

                if not granted:
                    tool_results.append(_denied_tool_result(tool_call, reason))
                    continue

            tool_results.append(await execute_tool_step(tool_call))

        if not tool_results:
            break

        messages.append(
            _tool_message_data(
                assistant_message=assistant_message,
                parts_data=tool_results,
            )
        )
        await _persist(session_id, messages)

    await clear_active_run_step(session_id)
    return messages


async def resume_tool_approval(
    approval_id: str,
    *,
    granted: bool,
    reason: str | None,
) -> None:
    await ToolApprovalHook(granted=granted, reason=reason).resume(approval_id)
