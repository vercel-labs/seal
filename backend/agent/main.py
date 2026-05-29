from collections.abc import AsyncGenerator
from typing import Any, ClassVar, cast

import vercel._internal.workflow.py_sandbox
import vercel.workflow

# ai imports shutil; sandbox turns os.supports_dir_fd into a function.
vercel._internal.workflow.py_sandbox._PASSTHROUGHS.add("ai")

import ai  # noqa: E402

workflow = vercel.workflow.Workflows()
MODEL_ID = "gateway:anthropic/claude-sonnet-4.6"
STREAM_SCOPE_AGENT = "agent"  # Mark records owned by the whole run, not one LLM call.
STREAM_SCOPE_LLM = "llm"  # Mark records emitted by one durable LLM step.


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


def _stream_path(stream_key: str) -> str:
    """Return the local side-channel path for a full agent run."""
    return f"./.streams/{stream_key}.jsonl"  # Keep every step on the same run file.


def _write_stream_record(
    stream_key: str,
    record: dict[str, object],
    *,
    mode: str = "a",
) -> None:
    """Write one JSONL stream record and flush it for the HTTP tailer."""
    import json  # Keep JSON serialization local to side-channel writes.
    import pathlib  # Keep filesystem access local to side-channel writes.

    # Reuse the run stream path so every durable step writes to one file.
    channel_path = pathlib.Path(_stream_path(stream_key))
    # Create the directory here because `/run/stream` can be the first caller.
    channel_path.parent.mkdir(parents=True, exist_ok=True)
    # Use append except for the run-start step, which intentionally resets.
    with channel_path.open(mode, encoding="utf-8") as channel:
        channel.write(json.dumps(record) + "\n")  # Store one complete record per line.
        channel.flush()  # Make records visible while the workflow keeps running.


@workflow.step
async def start_agent_stream(stream_key: str) -> None:
    """Initialize the side-channel once for the full agent workflow run."""
    # Truncate once so later LLM calls do not erase prior records.
    _write_stream_record(
        stream_key,
        {"type": "start", "scope": STREAM_SCOPE_AGENT},
        mode="w",
    )


@workflow.step
async def finish_agent_stream(stream_key: str) -> None:
    """Terminate the side-channel once the full agent workflow run is done."""
    # Emit the only terminal marker the HTTP drainer should stop on.
    _write_stream_record(
        stream_key,
        {"type": "done", "scope": STREAM_SCOPE_AGENT},
    )


@workflow.step
async def stream_llm(
    stream_key: str,
    messages: list[dict[str, object]],
    tool_schemas: list[dict[str, object]] | None = None,
    model_id: str = MODEL_ID,
) -> dict[str, object]:
    """Durable wrapper around ``ai.stream``.

    Runs the model call as a workflow step (durable, retried) while
    streaming every event to a side-channel keyed by ``stream_key`` so a
    caller can render live tokens. Returns the final assistant message as
    a JSON-serializable dict (the step's durable result).

    ``stream_key`` must be supplied by the caller (e.g. the HTTP request
    id) and threaded in via ``workflow.start`` -> workflow -> step. Do NOT
    generate it randomly inside the workflow: steps retry up to 3 times and
    the drainer must keep targeting the same channel.

    How to replace a naked ``ai.stream`` *inside a workflow*::

        # BEFORE (not durable, no side-channel):
        #   async with ai.stream(model, messages, tools=tools) as s:
        #       async for event in s:
        #           ...
        #       final = s.message
        #
        # AFTER:
        result = await stream_llm(
            stream_key,
            [m.model_dump(mode="json") for m in messages],
            tool_schemas=[
                {"name": t.name, "args": t.args.model_dump(mode="json")}
                for t in tools
            ],
        )
        final = ai.messages.Message.model_validate(result)

    The HTTP callsite drains the full run, not this one LLM step::

        # stream_key = request_id  # stable id, generated OUTSIDE the workflow
        # run = await vercel.workflow.start(run_agent, prompt, stream_key)
        #
        # # tail ./.streams/{stream_key}.jsonl line-by-line, e.g. for SSE:
        # import json, asyncio, pathlib
        # path = pathlib.Path(f"./.streams/{stream_key}.jsonl")
        # while not path.exists():
        #     await asyncio.sleep(0.05)
        # with path.open() as fh:
        #     while True:
        #         line = fh.readline()
        #         if not line:
        #             await asyncio.sleep(0.05)
        #             continue
        #         record = json.loads(line)
        #         if record["type"] == "done" and record["scope"] == "agent":
        #             break
        #         if record["type"] == "TextDelta":
        #             yield record["data"]["chunk"]  # push to client
        #
        # # durable final message, after the run-level "done":
        # final = await vercel.workflow.Run(run.run_id).return_value()
    """
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

    # ── Side-channel: local JSONL ────────────────────────────────────
    # TODO(postgres): replace `_write_stream_record` with ordered inserts.
    _write_stream_record(  # Mark an LLM turn without ending the full agent run.
        stream_key,
        {"type": "llm_start", "scope": STREAM_SCOPE_LLM},
    )
    async with ai.stream(model, parsed_messages, tools=tools) as s:
        async for event in s:
            # Append events so later LLM calls keep the same HTTP stream open.
            _write_stream_record(
                stream_key,
                {
                    "type": type(event).__name__,
                    "scope": STREAM_SCOPE_LLM,
                    "data": event.model_dump(mode="json"),
                },
            )
        if s.message is None:
            raise RuntimeError("LLM stream ended without a final message")
        _write_stream_record(  # End only this durable LLM step, not the HTTP stream.
            stream_key,
            {"type": "llm_end", "scope": STREAM_SCOPE_LLM},
        )
        return s.message.model_dump(mode="json")


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
            #    is the complete message (replayed on retry); live tokens land
            #    in the jsonl side-channel keyed by ``stream_key``.
            result = await stream_llm(
                stream_key,
                [m.model_dump(mode="json") for m in context.messages],
                tool_schemas=tool_schemas,
                model_id=model_id,
            )
            llm_msg = ai.messages.Message.model_validate(result)

            # 2. Replay the complete message into a synthetic stream so the
            #    rest of the loop is identical to ``Agent.loop``. Each tool
            #    call resolves to a ``@workflow.step`` and runs durably.
            async with (
                ai.Stream(ai.events.replay_message_events(llm_msg)) as stream,
                ai.ToolRunner() as tr,
            ):
                async for event in ai.util.merge(stream, tr.events()):
                    yield event

                    if isinstance(event, ai.events.ToolEnd):
                        tr.schedule(context.resolve(event.tool_call))

                context.add(stream.message)
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
