import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, ClassVar, cast

import ai
import vercel.workflow

from durable_agent import proto, stream

workflow = vercel.workflow.Workflows()

MODEL_ID = "gateway:anthropic/claude-sonnet-4.6"
SYSTEM_PROMPT = (
    "You are Seal, a coding assistant. Use bash, web_fetch, and subagent to "
    "inspect the environment, gather information, and delegate focused work."
)
SUBAGENT_SYSTEM_PROMPT = (
    "You are a focused Seal subagent. Use bash and web_fetch when useful, then "
    "answer the delegated task directly."
)


# hack: agent.run requires model, but can't actually pass it to ai.stream
# that's wrapped in a step
class _WorkflowModelProvider(ai.Provider[Any]):
    def __init__(self) -> None:
        super().__init__(name="workflow-placeholder", base_url="")


def _workflow_model() -> ai.Model:
    return ai.Model("workflow-placeholder", provider=_WorkflowModelProvider())


@workflow.step
async def llm_step(
    model_id: str,
    messages_data: list[dict[str, object]],
    tools_data: list[dict[str, object]],
    session_id: str | None,
) -> dict[str, object]:
    model = ai.get_model(model_id)
    messages = [
        ai.messages.Message.model_validate(message) for message in messages_data
    ]
    tools = [ai.Tool.model_validate(tool) for tool in tools_data]
    writer = await stream.get_writable(session_id) if session_id else None

    message: ai.messages.Message | None = None

    async with ai.stream(model, messages, tools=tools) as model_stream:
        async for e in model_stream:
            if writer is not None:
                await writer.write(e)
            if isinstance(e, ai.events.StreamEnd):
                message = e.message

    assert message is not None
    return message.model_dump(mode="json")


llm_step.max_retries = 0


@workflow.step
async def emit_event(
    # writes one agent event to the durable stream
    session_id: str,
    event_data: dict[str, object],
) -> None:
    writer = await stream.get_writable(session_id)
    await writer.write(event_data)


emit_event.max_retries = 0


# closes a durable event stream once the owning session is terminal.
@workflow.step
async def close_stream(session_id: str) -> None:
    writer = await stream.get_writable(session_id)
    await writer.close()


close_stream.max_retries = 0


@workflow.step
async def _bash(command: str, timeout: int | None = None) -> str:
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


_bash.max_retries = 0


@ai.tool
async def bash(command: str, timeout: int | None = None) -> str:
    """Execute a bash command. Use timeout in seconds to limit long-running commands."""
    return await _bash(command, timeout)


@workflow.step
async def _web_fetch(
    url: str,
    method: str = "GET",
    headers: str = "",
    body: str = "",
) -> str:
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


_web_fetch.max_retries = 0


@ai.tool
async def web_fetch(
    url: str,
    method: str = "GET",
    headers: str = "",
    body: str = "",
) -> str:
    """Fetch a URL and return the response."""
    return await _web_fetch(url, method, headers, body)


# we only need schema off of this tool
@ai.tool
async def subagent(prompt: str, name: str | None = None) -> str:
    """Delegate a focused task to a child agent and return its answer."""
    raise RuntimeError("subagent is dispatched by the durable driver")


class DurableAgent(ai.Agent):
    TOOLS: ClassVar[list[ai.AgentTool]] = [bash, web_fetch]

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        params = context.params if isinstance(context.params, dict) else {}
        model_id = str(params.get("model_id") or MODEL_ID)

        control = cast(dict[str, object], params["control"])
        session_id = cast(str | None, params.get("session_id"))

        while context.keep_running():
            result = await llm_step(
                model_id,
                [message.model_dump(mode="json") for message in context.messages],
                [tool.model_dump(mode="json") for tool in context.tools],
                session_id,
            )

            assistant_message = ai.messages.Message.model_validate(result)
            context.add(assistant_message)

            pending_subagents: list[proto.SubagentRequest] = []

            async with ai.ToolRunner() as runner:
                for tool_call in assistant_message.tool_calls:
                    if tool_call.tool_name == "subagent":
                        # we're not processing this inside the loop
                        # we'll return that and have session driver dispatch
                        # a separate agent session for this
                        args = json.loads(tool_call.tool_args or "{}")
                        pending_subagents.append(
                            proto.SubagentRequest(
                                tool_call_id=tool_call.tool_call_id,
                                name=str(args.get("name") or "subagent"),
                                prompt=str(args["prompt"]),
                            )
                        )
                    else:
                        runner.schedule(context.resolve(tool_call))

                async for event in runner.events():
                    if session_id is not None:
                        await emit_event(
                            session_id,
                            event.model_dump(mode="json"),
                        )
                    yield event

                tool_message = runner.get_tool_message()

            if tool_message is not None:
                context.add(tool_message)
            elif pending_subagents:
                # guarantee a trailing tool message for the driver to extend
                context.add(ai.messages.Message(role="tool", parts=[]))

            if pending_subagents:
                control["pending_subagents"] = [
                    request.model_dump(mode="json") for request in pending_subagents
                ]
                break

        # Keeps this method an async generator when the loop exits without yielding.
        if False:
            yield


# runs one agent turn, maybe requests subagents
@workflow.workflow
async def run_turn(turn_input: dict[str, Any]) -> None:
    _turn_input = proto.TurnInput.model_validate(turn_input)

    # messages should already contain either the user message
    # or the tool result message, so no need to do anything

    extra_tools = [] if _turn_input.mode == "task" else [subagent]
    agent = DurableAgent(tools=extra_tools)

    # side-channel for exfiltrating subagent requests out of the loop
    control: dict[str, object] = {
        "pending_subagents": [],
    }

    async with agent.run(
        _workflow_model(),
        _turn_input.messages,
        params={
            "model_id": MODEL_ID,
            "session_id": _turn_input.session_id,
            "control": control,
        },
    ) as run:
        async for _event in run:
            pass

        messages = run.messages

    subagent_requests = [
        proto.SubagentRequest.model_validate(item)
        for item in cast(list[dict[str, object]], control["pending_subagents"])
    ]

    if _turn_input.mode == "infinite":
        output_kind = "subagents" if subagent_requests else "suspend"
    else:
        output_kind = "done"

    output = proto.TurnOutput(
        kind=output_kind, messages=messages, subagent_requests=subagent_requests
    )

    # notify the driver session that parked on this turn
    await proto.TurnHook(output=output).resume(_turn_input.turn_hook_token)
