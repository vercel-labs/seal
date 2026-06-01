from collections.abc import AsyncGenerator
from typing import Any, ClassVar, cast

import vercel._internal.workflow.py_sandbox
import vercel.workflow

# ai imports shutil; sandbox turns os.supports_dir_fd into a function.
vercel._internal.workflow.py_sandbox._PASSTHROUGHS.add("ai")

import ai  # noqa: E402

from agent import durable_stream  # noqa: E402

workflow = vercel.workflow.Workflows()
MODEL_ID = "gateway:anthropic/claude-sonnet-4.6"


class _WorkflowModelProvider(ai.Provider[Any]):
    def __init__(self) -> None:
        # Real provider/httpx construction happens inside stream_llm.
        super().__init__(name="workflow-placeholder", base_url="")


@workflow.step
async def _bash(command: str, timeout: int | None = None) -> str:
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


_bash.max_retries = 0


@ai.tool
async def bash(command: str, timeout: int | None = None) -> str:
    """Execute a bash command.

    Use timeout (seconds) to limit long-running commands.
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
    """Fetch a URL and return the response.

    Args:
        url: The URL to fetch.
        method: HTTP method (GET, POST, PUT, DELETE, etc.).
        headers: Optional headers as newline-separated "Key: Value" pairs.
        body: Optional request body for POST/PUT.
    """
    return await _web_fetch(url, method, headers, body)


@workflow.step
async def start_agent_stream(stream_key: str) -> None:
    """Initialize the side-channel once for the full agent workflow run."""
    stream = durable_stream.get_writable(stream_key, reset=True)
    await stream.write(durable_stream.StreamStart())


start_agent_stream.max_retries = 0


@workflow.step
async def finish_agent_stream(stream_key: str) -> None:
    """Terminate the side-channel once the full agent workflow run is done."""
    stream = durable_stream.get_writable(stream_key)
    await stream.write(durable_stream.StreamDone())


finish_agent_stream.max_retries = 0


@workflow.step
async def stream_llm(
    stream_key: str,
    messages: list[dict[str, object]],
    tool_schemas: list[dict[str, object]] | None = None,
    model_id: str = MODEL_ID,
) -> dict[str, object]:
    """Durable wrapper around ``ai.stream``."""
    # Provider/httpx setup belongs in the step, outside the workflow body.
    model = ai.get_model(model_id)
    parsed_messages = [ai.messages.Message.model_validate(m) for m in messages]
    tools = [
        ai.Tool(
            kind="function",
            name=cast(str, t["name"]),
            args=ai.tools.FunctionToolArgs.model_validate(t["args"]),
        )
        for t in (tool_schemas or [])
    ]

    stream = durable_stream.get_writable(stream_key)
    await stream.write(durable_stream.LLMStart())
    async with ai.stream(model, parsed_messages, tools=tools) as s:
        async for event in s:
            # Append events so later LLM calls keep the same HTTP stream open.
            await stream.write(
                durable_stream.StreamEvent(
                    type=type(event).__name__,
                    data=event.model_dump(mode="json"),
                )
            )
        if s.message is None:
            raise RuntimeError("LLM stream ended without a final message")
        await stream.write(durable_stream.LLMEnd())
        return s.message.model_dump(mode="json")


stream_llm.max_retries = 0


class SealAgent(ai.Agent):
    TOOLS: ClassVar[list[ai.AgentTool]] = [bash, web_fetch]

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        # ``stream_key`` is threaded in via ``agent.run(..., params=...)`` so
        # the durable LLM step can publish live tokens to its side-channel.
        stream_key = context.params["stream_key"]
        model_id = context.params["model_id"]
        tool_schemas: list[dict[str, object]] = [
            {"name": t.name, "args": t.args.model_dump(mode="json")}
            for t in context.tools
        ]

        while context.keep_running():
            # 1. LLM call via the durable ``stream_llm`` step. The step result
            #    is the complete message; live tokens land in the jsonl
            #    side-channel keyed by ``stream_key``.
            result = await stream_llm(
                stream_key,
                [m.model_dump(mode="json") for m in context.messages],
                tool_schemas=tool_schemas,
                model_id=model_id,
            )
            llm_msg = ai.messages.Message.model_validate(result)

            context.add(llm_msg)

            # 2. Dispatch tool calls directly from the final assistant message.
            #    Tool bodies resolve to workflow steps, and their results are
            #    folded back into context for the next LLM turn.
            async with ai.ToolRunner() as tr:
                for tool_call in llm_msg.tool_calls:
                    tr.schedule(context.resolve(tool_call))

                async for event in tr.events():
                    yield event

                context.add(tr.get_tool_message())

SYSTEM_PROMPT = (
    "You are Seal, a coding assistant. Use the bash and web_fetch tools to "
    "inspect the environment and gather information before answering."
)


@workflow.workflow
async def run_agent(prompt: str, stream_key: str) -> str:
    """Durable agent run. Returns the final assistant text.

    ``stream_key`` is supplied by the caller (outside the workflow, so it is
    stable across step retries) and selects the jsonl side-channel that
    ``stream_llm`` writes live tokens to.

    Trigger a run with the public workflow API, e.g.::

        run = await vercel.workflow.start(run_agent, prompt, stream_key)
        final = await vercel.workflow.Run(run.run_id).return_value()
    """
    # Real provider/httpx construction happens inside stream_llm.
    model = ai.Model("workflow-placeholder", provider=_WorkflowModelProvider())
    agent = SealAgent()
    messages = [ai.system_message(SYSTEM_PROMPT), ai.user_message(prompt)]
    await start_agent_stream(stream_key)  # Own truncation at the workflow-run boundary.

    async with agent.run(
        model,
        messages,
        params={"stream_key": stream_key, "model_id": MODEL_ID},
    ) as stream:
        async for _event in stream:
            pass
        # Save before the final step so return stays after stream close.
        output = stream.output
    # Send done after all LLM/tool workflow steps finish.
    await finish_agent_stream(stream_key)
    return output
