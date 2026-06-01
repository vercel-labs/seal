import asyncio
import collections.abc
import typing

import ai
import vercel.workflow

import durable_agent.stream

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
class _WorkflowModelProvider(ai.Provider[typing.Any]):
    def __init__(self) -> None:
        super().__init__(name="workflow-placeholder", base_url="")


def _workflow_model() -> ai.Model:
    return ai.Model("workflow-placeholder", provider=_WorkflowModelProvider())


@workflow.step
async def llm_step(
    model_id: str,
    messages_data: list[dict[str, object]],
    tools_data: list[dict[str, object]],
    writable_data: dict[str, object] | None,
) -> dict[str, object]:
    model = ai.get_model(model_id)
    messages = [
        ai.messages.Message.model_validate(message) for message in messages_data
    ]
    tools = [ai.Tool.model_validate(tool) for tool in tools_data]
    writer = (
        await durable_agent.stream.get_writable(writable_data)
        if writable_data is not None
        else None
    )

    message: ai.messages.Message | None = None

    async with ai.stream(model, messages, tools=tools) as stream:
        async for e in stream:
            if writer is not None:
                await writer.write(e)
            if isinstance(e, ai.events.StreamEnd):
                message = e.message

    assert message is not None
    return message.model_dump(mode="json")


llm_step.max_retries = 0


@workflow.step
async def emit_event(
    writable_data: dict[str, object],
    event_data: dict[str, object],
) -> None:
    writer = await durable_agent.stream.get_writable(writable_data)
    await writer.write(event_data)


emit_event.max_retries = 0


@workflow.step
async def close_stream(writable_data: dict[str, object]) -> None:
    writer = await durable_agent.stream.get_writable(writable_data)
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
    """Execute a bash command.

    Use timeout in seconds to limit long-running commands.
    """
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


@ai.tool
async def subagent(prompt: str, name: str | None = None) -> ai.SubAgentTool:
    """Delegate a focused task to a child agent and return its answer."""
    label = f" You are acting as {name}." if name else ""
    messages = [
        ai.system_message(f"{SUBAGENT_SYSTEM_PROMPT}{label}"),
        ai.user_message(prompt),
    ]

    child = DurableAgent(tools=[bash, web_fetch])
    async with child.run(
        _workflow_model(),
        messages,
        params={"model_id": MODEL_ID},
    ) as stream:
        async for event in stream:
            yield event


class DurableAgent(ai.Agent):
    TOOLS: typing.ClassVar[list[ai.AgentTool]] = [bash, web_fetch, subagent]

    async def loop(
        self,
        context: ai.Context,
    ) -> collections.abc.AsyncGenerator[ai.events.AgentEvent]:
        params = context.params if isinstance(context.params, dict) else {}
        model_id = str(params.get("model_id") or MODEL_ID)
        writable = params.get("writable")
        writable_data = (
            dict(writable) if isinstance(writable, collections.abc.Mapping) else None
        )

        while context.keep_running():
            result = await llm_step(
                model_id,
                [message.model_dump(mode="json") for message in context.messages],
                [tool.model_dump(mode="json") for tool in context.tools],
                writable_data,
            )

            assistant_message = ai.messages.Message.model_validate(result)
            context.add(assistant_message)

            async with ai.ToolRunner() as runner:
                for tool_call in assistant_message.tool_calls:
                    runner.schedule(context.resolve(tool_call))

                async for event in runner.events():
                    if writable_data is not None:
                        await emit_event(writable_data, event.model_dump(mode="json"))
                    yield event

                context.add(runner.get_tool_message())

        # hack: needs to be a generator
        if False:
            yield


@workflow.workflow
async def run_agent(prompt: str) -> str:
    agent = DurableAgent()
    messages = [
        ai.system_message(SYSTEM_PROMPT),
        ai.user_message(prompt),
    ]
    writable = await durable_agent.stream.get_writable()

    try:
        async with agent.run(
            _workflow_model(),
            messages,
            params={"model_id": MODEL_ID, "writable": writable},
        ) as run:
            async for _event in run:
                pass
            return run.output
    finally:
        await close_stream(dict(writable))
