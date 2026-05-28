"""Durable Seal agent and workflow entrypoint."""

from __future__ import annotations

import collections.abc
import json
from typing import TYPE_CHECKING, Any, Literal, cast

import vercel.workflow as workflow

if TYPE_CHECKING:
    import ai


TOOL_APPROVAL_HOOK_TYPE = "ToolApproval"
APPROVAL_REQUIRED_TOOLS = frozenset({"bash", "web_fetch"})

SYSTEM = "You are a helpful assistant with access to a bash shell and the internet."

_TITLE_PROMPT = (
    "Generate a concise 3-6 word title for a conversation that starts with "
    "the following message. Reply with ONLY the title, no quotes or punctuation."
)

workflow_app = workflow.Workflows()
_tools: list[Any] | None = None
_model_event_adapter: Any | None = None


async def bash(command: str, timeout: int | None = None) -> str:
    """Execute a bash command.

    Use timeout (seconds) to limit long-running commands.
    """
    import asyncio

    proc = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        return f"Command timed out after {timeout}s."

    output = stdout.decode() if stdout else ""
    if proc.returncode != 0:
        return f"[exit code {proc.returncode}]\n{output}"
    return output


async def web_fetch(
    url: str,
    method: str = "GET",
    headers: str = "",
    body: str = "",
) -> str:
    """Fetch a URL and return the response.

    Args:
        url: The URL to fetch.
        method: HTTP method (GET, POST, PUT, DELETE, etc.).
        headers: Optional headers as newline-separated "Key: Value" pairs.
        body: Optional request body for POST/PUT.
    """
    import httpx

    parsed_headers: dict[str, str] = {}
    for line in headers.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            parsed_headers[key.strip()] = value.strip()

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.request(
            method,
            url,
            headers=parsed_headers or None,
            content=body or None,
        )

    parts = [
        f"HTTP {response.status_code}",
        *(f"{key}: {value}" for key, value in response.headers.items()),
        "",
        response.text[:50_000],
    ]
    return "\n".join(parts)


def get_tools() -> list[ai.AgentTool]:
    """Return the agent tools."""
    import ai

    global _tools
    if _tools is None:
        _tools = [
            ai.tool(require_approval=True)(bash),
            ai.tool(require_approval=True)(web_fetch),
        ]
    return cast(list[ai.AgentTool], _tools)


def get_model() -> ai.Model:
    """Create the primary LLM instance."""
    import ai

    return ai.get_model("anthropic/claude-opus-4.6")


def _get_fast_model() -> ai.Model:
    """Cheap / fast model for lightweight tasks like title generation."""
    import ai

    return ai.get_model("anthropic/claude-sonnet-4.6")


async def generate_title(first_message: str) -> str:
    """Generate a short title for a session using a cheap LLM call."""
    import ai

    messages = [
        ai.system_message(_TITLE_PROMPT),
        ai.user_message(first_message),
    ]
    async with ai.stream(_get_fast_model(), messages) as stream:
        async for _ in stream:
            pass
        return stream.text.strip()


def dump_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], message.model_dump(mode="json")) for message in messages
    ]


def load_messages(data: list[dict[str, Any]]) -> list[ai.messages.Message]:
    import ai

    return [ai.messages.Message.model_validate(item) for item in data]


def dump_message(message: Any) -> dict[str, Any]:
    return cast(dict[str, Any], message.model_dump(mode="json"))


def dump_agent_event(event: Any) -> dict[str, Any]:
    return cast(dict[str, Any], event.model_dump(mode="json"))


def load_agent_event(data: dict[str, object]) -> ai.events.AgentEvent:
    import pydantic
    from ai import events as ai_events

    global _model_event_adapter
    if _model_event_adapter is None:
        _model_event_adapter = pydantic.TypeAdapter(ai_events.DiscriminatedEvent)

    kind = data.get("kind")
    if kind == "tool_call_result":
        return ai_events.ToolCallResult.model_validate(data)
    if kind == "hook":
        return ai_events.HookEvent.model_validate(data)
    if kind == "partial_tool_call_result":
        return ai_events.PartialToolCallResult.model_validate(data)
    return cast(ai_events.AgentEvent, _model_event_adapter.validate_python(data))


def new_ui_stream_state() -> Any:
    import ai.agents.ui.ai_sdk.outbound_stream as ai_sdk_outbound_stream

    return ai_sdk_outbound_stream._StreamState()


async def to_supported_sse(
    events: collections.abc.AsyncIterable[ai.events.AgentEvent],
) -> collections.abc.AsyncGenerator[str]:
    """Convert events to SSE chunks supported by the installed JS AI SDK."""
    state = new_ui_stream_state()
    denied_tool_call_ids: set[str] = set()

    async for event in events:
        for chunk in ui_sse_chunks_for_agent_event(
            state,
            event,
            denied_tool_call_ids,
        ):
            yield chunk

    for chunk in finish_ui_sse_chunks(state, denied_tool_call_ids):
        yield chunk

    yield done_sse()


def ui_sse_chunks_for_agent_event(
    state: Any,
    event_or_data: ai.events.AgentEvent | dict[str, object],
    denied_tool_call_ids: set[str],
) -> list[str]:
    event = (
        load_agent_event(event_or_data)
        if isinstance(event_or_data, dict)
        else event_or_data
    )
    chunks: list[str] = []
    for ui_event in _ui_events_for_agent_event(state, event):
        chunk = _format_supported_sse_event(ui_event, denied_tool_call_ids)
        if chunk is not None:
            chunks.append(chunk)
    return chunks


def finish_ui_sse_chunks(
    state: Any,
    denied_tool_call_ids: set[str],
) -> list[str]:
    chunks: list[str] = []
    for ui_event in state.finish():
        chunk = _format_supported_sse_event(ui_event, denied_tool_call_ids)
        if chunk is not None:
            chunks.append(chunk)
    return chunks


def done_sse() -> str:
    import ai.agents.ui.ai_sdk.outbound_stream as ai_sdk_outbound_stream

    return ai_sdk_outbound_stream.format_done_sse()


def _ui_events_for_agent_event(
    state: Any,
    event: ai.events.AgentEvent,
) -> list[Any]:
    from ai import events as ai_events

    if isinstance(event, ai_events.ToolCallResult):
        return cast(list[Any], state.on_tool_result(event))
    if isinstance(event, ai_events.PartialToolCallResult):
        return cast(list[Any], state.on_partial_tool_result(event))
    if isinstance(event, ai_events.HookEvent):
        return cast(list[Any], state.on_hook(event))
    return cast(list[Any], state.on_event(event))


def _format_supported_sse_event(
    event: Any,
    denied_tool_call_ids: set[str],
) -> str | None:
    import ai.agents.ui.ai_sdk.outbound_stream as ai_sdk_outbound_stream

    event_type = getattr(event, "type", None)
    tool_call_id = getattr(event, "tool_call_id", None)

    if event_type == "tool-approval-response":
        return None

    if isinstance(tool_call_id, str) and event_type == "tool-output-denied":
        denied_tool_call_ids.add(tool_call_id)

    if (
        isinstance(tool_call_id, str)
        and event_type == "tool-output-error"
        and tool_call_id in denied_tool_call_ids
    ):
        return None

    return ai_sdk_outbound_stream.format_sse(event)


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
        tool_call if isinstance(tool_call, dict) else tool_call.model_dump(mode="json")
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
async def llm_step(
    stream_id: str,
    messages_data: list[dict[str, Any]],
) -> dict[str, Any]:
    import ai

    from core import attachments, stream_store

    messages = [
        ai.messages.Message.model_validate(message)
        for message in messages_data
        if _message_role(message) not in ("system", "internal")
    ]
    messages = await attachments.inline_file_parts(messages)
    full_messages = [ai.system_message(SYSTEM), *messages]

    async with ai.stream(
        get_model(),
        full_messages,
        tools=[tool.tool for tool in get_tools()],
    ) as stream:
        emitted_approval_ids: set[str] = set()
        async for event in stream:
            await stream_store.append_event(stream_id, dump_agent_event(event))
            if not isinstance(event, ai.events.ToolEnd):
                continue
            tool_call = event.tool_call
            if not tool_requires_approval(tool_call.tool_name):
                continue
            approval_id = _approval_id(tool_call)
            if approval_id in emitted_approval_ids:
                continue
            emitted_approval_ids.add(approval_id)

            hook_message = _hook_message(
                tool_call=tool_call,
                assistant_message=event.message,
                status="pending",
            )
            hook = next(
                part
                for part in hook_message.parts
                if isinstance(part, ai.messages.HookPart)
            )
            await stream_store.append_event(
                stream_id,
                dump_agent_event(ai.events.HookEvent(message=hook_message, hook=hook)),
            )
        if stream.message is None:
            raise RuntimeError("LLM stream ended without a final message")
        return dump_message(stream.message)


llm_step.max_retries = 0


@workflow_app.step
async def execute_tool_step(tool_call_data: dict[str, Any]) -> dict[str, Any]:
    import ai

    tool_call = ai.messages.ToolCallPart.model_validate(tool_call_data)
    tool = next(
        (tool for tool in get_tools() if tool.name == tool_call.tool_name),
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
async def append_hook_event_step(
    stream_id: str,
    message_data: dict[str, Any],
) -> None:
    import ai

    from core import stream_store

    message = load_message(message_data)
    hook = next(
        (part for part in message.parts if isinstance(part, ai.messages.HookPart)),
        None,
    )
    if hook is None:
        return
    event = ai.events.HookEvent(message=message, hook=hook)
    await stream_store.append_event(stream_id, dump_agent_event(event))


append_hook_event_step.max_retries = 0


@workflow_app.step
async def append_tool_result_event_step(
    stream_id: str,
    message_data: dict[str, Any],
) -> None:
    import ai

    from core import stream_store

    event = ai.tool_result(load_message(message_data))
    await stream_store.append_event(stream_id, dump_agent_event(event))


append_tool_result_event_step.max_retries = 0


@workflow_app.step
async def load_tool_approvals_step(
    stream_id: str,
    approval_ids: list[str],
) -> dict[str, dict[str, Any]]:
    from core import stream_store

    return await stream_store.list_tool_approvals(stream_id, approval_ids)


load_tool_approvals_step.max_retries = 0


@workflow_app.step
async def set_stream_status_step(
    stream_id: str,
    status: Literal["idle", "running", "waiting", "completed", "failed"],
) -> None:
    from core import stream_store

    await stream_store.set_status(stream_id, status)


@workflow_app.step
async def persist_messages_step(
    session_id: str,
    messages_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from core import sessions

    messages = load_messages(messages_data)
    await sessions.save_messages(session_id, messages)
    return messages_data


@workflow_app.step
async def clear_active_run_step(session_id: str) -> None:
    from core import sessions

    await sessions.set_active_run(session_id, None)


async def _persist(
    session_id: str,
    messages: list[dict[str, Any]],
) -> None:
    await persist_messages_step(session_id, messages)


class SealDurableAgent:
    """Durable custom loop for the Seal chat agent."""

    def __init__(self, *, session_id: str, stream_id: str) -> None:
        self.session_id = session_id
        self.stream_id = stream_id

    async def run(self, messages_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages = messages_data

        while messages and _message_role(messages[-1]) != "assistant":
            assistant_message = await llm_step(self.stream_id, messages)
            messages.append(assistant_message)
            await _persist(self.session_id, messages)

            tool_calls = _tool_call_parts(assistant_message)
            approval_ids_by_call: dict[str, str] = {}
            for tool_call in tool_calls:
                if tool_requires_approval(_tool_name(tool_call)):
                    approval_id = _approval_id(tool_call)
                    approval_ids_by_call[_tool_call_id(tool_call)] = approval_id
                    hook_message = _hook_message_data(
                        tool_call=tool_call,
                        assistant_message=assistant_message,
                        status="pending",
                    )
                    messages.append(hook_message)
                    await _persist(self.session_id, messages)

            approval_resolutions: dict[str, tuple[bool, str | None]] = {}
            approval_ids = list(approval_ids_by_call.values())
            while len(approval_resolutions) < len(approval_ids_by_call):
                await set_stream_status_step(self.stream_id, "waiting")
                approvals = await load_tool_approvals_step(self.stream_id, approval_ids)
                for tool_call_id, approval_id in approval_ids_by_call.items():
                    if tool_call_id in approval_resolutions:
                        continue
                    approval = approvals.get(approval_id)
                    if approval is None:
                        continue
                    approval_resolutions[tool_call_id] = (
                        bool(approval["granted"]),
                        cast(str | None, approval.get("reason")),
                    )
                if len(approval_resolutions) < len(approval_ids_by_call):
                    await workflow.sleep(0.5)

            if approval_resolutions:
                await set_stream_status_step(self.stream_id, "running")

            tool_results: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                tool_call_id = _tool_call_id(tool_call)
                if tool_call_id in approval_resolutions:
                    granted, reason = approval_resolutions[tool_call_id]
                    resolution = {
                        "granted": granted,
                        "reason": reason,
                    }
                    hook_message = _hook_message_data(
                        tool_call=tool_call,
                        assistant_message=assistant_message,
                        status="resolved",
                        resolution=resolution,
                    )
                    messages.append(hook_message)
                    await _persist(self.session_id, messages)
                    await append_hook_event_step(self.stream_id, hook_message)

                    if not granted:
                        tool_results.append(_denied_tool_result(tool_call, reason))
                        continue

                tool_results.append(await execute_tool_step(tool_call))

            if not tool_results:
                break

            tool_message = _tool_message_data(
                assistant_message=assistant_message,
                parts_data=tool_results,
            )
            messages.append(tool_message)
            await _persist(self.session_id, messages)
            await append_tool_result_event_step(self.stream_id, tool_message)

        await clear_active_run_step(self.session_id)
        await set_stream_status_step(self.stream_id, "completed")
        return messages


@workflow_app.workflow
async def run_agent(
    session_id: str,
    stream_id: str,
    messages_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    agent = SealDurableAgent(session_id=session_id, stream_id=stream_id)
    return await agent.run(messages_data)
