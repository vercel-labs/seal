import json
import os
from collections.abc import AsyncGenerator
from typing import Any, ClassVar, Literal, cast

_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault(
    "SEAL_DURABLE_STREAMS_DIR",
    os.path.join(_BACKEND_DIR, ".streams"),
)
os.environ.setdefault(
    "SEAL_DURABLE_SESSIONS_DIR",
    os.path.join(_BACKEND_DIR, ".sessions"),
)
os.environ.setdefault(
    "WORKFLOW_LOCAL_DATA_DIR",
    os.path.join(_BACKEND_DIR, ".workflow-data"),
)

import vercel._internal.workflow.py_sandbox  # noqa: E402
import vercel.workflow  # noqa: E402

# ai imports shutil; sandbox turns os.supports_dir_fd into a function.
vercel._internal.workflow.py_sandbox._PASSTHROUGHS.add("ai")

import ai  # noqa: E402
import pydantic  # noqa: E402

workflow = vercel.workflow.Workflows()
MODEL_ID = "gateway:anthropic/claude-sonnet-4.6"
type SessionSource = Literal["root", "subagent"]
type TurnAction = Literal["continue", "done", "runtime_action", "error"]


class SessionTurn(pydantic.BaseModel, vercel.workflow.BaseHook):
    prompt: str | None = None
    close: bool = False


class TurnCompletion(pydantic.BaseModel, vercel.workflow.BaseHook):
    action: TurnAction
    session: dict[str, object] | None = None
    output: str = ""
    actions: list[dict[str, object]] = pydantic.Field(default_factory=list)
    error: str | None = None


class RuntimeActionResult(pydantic.BaseModel, vercel.workflow.BaseHook):
    tool_call_id: str
    tool_name: str
    result: object
    is_error: bool = False


class _WorkflowModelProvider(ai.Provider[Any]):
    def __init__(self) -> None:
        # Real provider/httpx construction happens inside stream_llm.
        super().__init__(name="workflow-placeholder", base_url="")


async def _execute_bash(command: str, timeout: int | None = None) -> str:
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


async def _execute_web_fetch(
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


@workflow.step
async def _bash(command: str, timeout: int | None = None) -> str:
    return await _execute_bash(command, timeout)


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
    return await _execute_web_fetch(url, method, headers, body)


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


SUBAGENT_TOOL = ai.Tool(
    kind="function",
    name="subagent",
    args=ai.tools.FunctionToolArgs(
        description="Delegate a focused task to a child agent and return its answer.",
        params={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task for the child agent.",
                },
                "name": {
                    "type": "string",
                    "description": "Short display name for the child agent.",
                },
            },
            "required": ["prompt"],
        },
    ),
)


@workflow.step
async def start_agent_stream(stream_key: str) -> None:
    """Initialize the side-channel once for the full agent workflow run."""
    from agent import durable_stream

    stream = durable_stream.get_writable(stream_key, reset=True)
    await stream.write(durable_stream.StreamStart())


start_agent_stream.max_retries = 0


@workflow.step
async def finish_agent_stream(stream_key: str) -> None:
    """Terminate the side-channel once the full agent workflow run is done."""
    from agent import durable_stream

    stream = durable_stream.get_writable(stream_key)
    await stream.write(durable_stream.StreamDone())


finish_agent_stream.max_retries = 0


@workflow.step
async def write_agent_event(
    stream_key: str,
    session_id: str,
    kind: str,
    data: dict[str, object],
    turn_id: str | None = None,
    source: SessionSource = "root",
    parent: dict[str, str] | None = None,
    reset: bool = False,
) -> int:
    """Append one authoritative indexed agent event."""
    from agent import durable_stream

    stream = durable_stream.get_writable(stream_key, reset=reset)
    return await stream.write(
        durable_stream.AgentStreamEvent(
            kind=kind,
            session_id=session_id,
            turn_id=turn_id,
            source=source,
            parent=parent,
            data=data,
        )
    )


write_agent_event.max_retries = 0


@workflow.step
async def stream_llm(
    stream_key: str,
    messages: list[dict[str, object]],
    tool_schemas: list[dict[str, object]] | None = None,
    model_id: str = MODEL_ID,
    session_id: str | None = None,
    turn_id: str | None = None,
    source: SessionSource = "root",
    parent: dict[str, str] | None = None,
) -> dict[str, object]:
    """Durable wrapper around ``ai.stream``."""
    from agent import durable_stream

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
    if session_id is None:
        await stream.write(durable_stream.LLMStart())
    else:
        await stream.write(
            durable_stream.AgentStreamEvent(
                kind="model.started",
                session_id=session_id,
                turn_id=turn_id,
                source=source,
                parent=parent,
                data={"model_id": model_id},
            )
        )
    async with ai.stream(model, parsed_messages, tools=tools) as s:
        async for event in s:
            # Append events so later LLM calls keep the same HTTP stream open.
            if session_id is None:
                await stream.write(
                    durable_stream.StreamEvent(
                        type=type(event).__name__,
                        data=event.model_dump(mode="json"),
                    )
                )
            else:
                await stream.write(
                    durable_stream.AgentStreamEvent(
                        kind="model.event",
                        session_id=session_id,
                        turn_id=turn_id,
                        source=source,
                        parent=parent,
                        data={
                            "event_type": type(event).__name__,
                            "event": event.model_dump(mode="json"),
                        },
                    )
                )
        if s.message is None:
            raise RuntimeError("LLM stream ended without a final message")
        if session_id is None:
            await stream.write(durable_stream.LLMEnd())
        else:
            await stream.write(
                durable_stream.AgentStreamEvent(
                    kind="model.completed",
                    session_id=session_id,
                    turn_id=turn_id,
                    source=source,
                    parent=parent,
                    data={"message": s.message.model_dump(mode="json")},
                )
            )
        return s.message.model_dump(mode="json")


stream_llm.max_retries = 0


class SealAgent(ai.Agent):
    TOOLS: ClassVar[list[ai.AgentTool]] = [bash, web_fetch]

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        # ``stream_key`` is threaded in via ``agent.run(..., params=...)`` so
        # the durable LLM step can publish live tokens to its side-channel.
        params = context.params if isinstance(context.params, dict) else {}
        stream_key = params["stream_key"]
        model_id = params["model_id"]
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
                **(
                    {
                        "session_id": params["session_id"],
                        "turn_id": params.get("turn_id"),
                        "source": params.get("source", "root"),
                        "parent": params.get("parent"),
                    }
                    if "session_id" in params
                    else {}
                ),
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
SESSION_SYSTEM_PROMPT = (
    "You are Seal, a coding assistant. Use bash, web_fetch, and subagent to "
    "inspect the environment, gather information, and delegate focused work."
)
SUBAGENT_SYSTEM_PROMPT = (
    "You are a focused Seal subagent. Complete the delegated task directly."
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


def _dump_message(message: ai.messages.Message) -> dict[str, object]:
    return cast(dict[str, object], message.model_dump(mode="json"))


@workflow.step
async def create_session_step(
    session_id: str,
    stream_key: str,
    source: SessionSource = "root",
) -> dict[str, object]:
    """Create the durable session snapshot and return its small handle."""
    from agent import durable_session

    system_prompt = (
        SUBAGENT_SYSTEM_PROMPT if source == "subagent" else SESSION_SYSTEM_PROMPT
    )
    system_message = ai.system_message(system_prompt).model_copy(
        update={"id": f"{session_id}-system"}
    )
    state = durable_session.DurableSessionState(
        session_id=session_id,
        stream_key=stream_key,
    )
    snapshot = durable_session.DurableSessionSnapshot(
        messages=[_dump_message(system_message)]
    )
    return (await durable_session.write(state, snapshot, reset=True)).model_dump(
        mode="json"
    )


create_session_step.max_retries = 0


@workflow.step
async def read_session_step(session_state: dict[str, object]) -> dict[str, object]:
    """Read the latest durable session snapshot from its session stream."""
    from agent import durable_session

    state = durable_session.DurableSessionState.model_validate(session_state)
    return (await durable_session.read(state)).model_dump(mode="json")


read_session_step.max_retries = 0


@workflow.step
async def write_session_step(
    session_state: dict[str, object],
    messages: list[dict[str, object]],
    output: str = "",
    pending_runtime_actions: list[dict[str, object]] | None = None,
    pending_tool_results: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Append a durable session snapshot and return its small handle."""
    from agent import durable_session

    state = durable_session.DurableSessionState.model_validate(session_state)
    snapshot = durable_session.DurableSessionSnapshot(
        messages=messages,
        output=output,
        pending_runtime_actions=pending_runtime_actions or [],
        pending_tool_results=pending_tool_results or [],
    )
    return (await durable_session.write(state, snapshot)).model_dump(mode="json")


write_session_step.max_retries = 0


def _session_messages(session_state: dict[str, object]) -> list[dict[str, object]]:
    raw_messages = session_state.get("messages")
    if not isinstance(raw_messages, list):
        return []

    messages: list[dict[str, object]] = []
    for raw_message in raw_messages:
        message = ai.messages.Message.model_validate(raw_message)
        messages.append(_dump_message(message))
    return messages


def _agent_tool_schemas(source: SessionSource) -> list[dict[str, object]]:
    if source == "subagent":
        return []

    tools = [bash.tool, web_fetch.tool, SUBAGENT_TOOL]
    return [
        {"name": tool.name, "args": tool.args.model_dump(mode="json")} for tool in tools
    ]


def _parse_tool_args(tool_call: ai.messages.ToolCallPart) -> dict[str, object]:
    try:
        args = json.loads(tool_call.tool_args or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON tool args: {error}") from error
    if not isinstance(args, dict):
        raise ValueError("tool args must be a JSON object")
    return cast(dict[str, object], args)


def _tool_result_part(
    tool_call: ai.messages.ToolCallPart,
    result: object,
    *,
    turn_id: str,
    model_round: int,
    call_index: int,
    is_error: bool = False,
) -> ai.messages.ToolResultPart:
    part = ai.tool_result_part(
        tool_call.tool_call_id,
        tool_name=tool_call.tool_name,
        result=result,
        is_error=is_error,
    )
    return part.model_copy(
        update={"id": f"{turn_id}-tool-result-{model_round}-{call_index}"}
    )


def _stable_slug(value: str) -> str:
    slug = "".join(
        char.lower() if ("a" <= char.lower() <= "z" or "0" <= char <= "9") else "-"
        for char in value
    ).strip("-")
    return slug or "item"


async def _resume_hook_with_retry(
    payload: vercel.workflow.BaseHook,
    token: str,
) -> None:
    import asyncio

    for attempt in range(40):
        try:
            await payload.resume(token)
            return
        except RuntimeError as error:
            error_text = str(error)
            lowered = error_text.lower()
            missing_hook = (
                "hook" in lowered and "not found" in lowered and token in error_text
            )
            if not missing_hook or attempt == 39:
                raise
            await asyncio.sleep(0.05)


@workflow.step
async def notify_turn_completion(
    completion_token: str,
    result: dict[str, object],
) -> None:
    await _resume_hook_with_retry(
        TurnCompletion.model_validate(result),
        completion_token,
    )


notify_turn_completion.max_retries = 0


@workflow.step
async def notify_runtime_action_result(
    result_token: str,
    tool_call_id: str,
    tool_name: str,
    result: object,
    is_error: bool = False,
) -> None:
    await _resume_hook_with_retry(
        RuntimeActionResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=result,
            is_error=is_error,
        ),
        result_token,
    )


notify_runtime_action_result.max_retries = 0


@workflow.step
async def dispatch_runtime_actions_step(
    stream_key: str,
    session_id: str,
    turn_id: str,
    actions: list[dict[str, object]],
    source: SessionSource = "root",
    parent: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Start child session workflows for runtime actions."""
    from agent import durable_stream

    stream = durable_stream.get_writable(stream_key)
    started_actions: list[dict[str, object]] = []
    for action in actions:
        prompt = action.get("prompt")
        result_token = action.get("runtime_result_token")
        tool_call_id = str(action.get("tool_call_id") or "")
        name = str(action.get("name") or "subagent")
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError("subagent runtime action requires a prompt")
        if not isinstance(result_token, str) or not result_token:
            raise RuntimeError("subagent runtime action requires a result token")

        child_session_id = (
            f"{session_id}-child-{_stable_slug(turn_id)}-{_stable_slug(tool_call_id)}"
        )
        child_parent = {
            "session_id": session_id,
            "turn_id": turn_id,
            "tool_call_id": tool_call_id,
            "name": name,
            "runtime_result_token": result_token,
        }
        await stream.write(
            durable_stream.AgentStreamEvent(
                kind="subagent.called",
                session_id=session_id,
                turn_id=turn_id,
                source=source,
                parent=parent,
                data={
                    "child_session_id": child_session_id,
                    "child_stream_key": child_session_id,
                    "name": name,
                    "tool_call_id": tool_call_id,
                },
            )
        )
        started = await vercel.workflow.start(
            run_session,
            child_session_id,
            child_session_id,
            prompt,
            "subagent",
            child_parent,
        )
        started_actions.append(
            {
                "tool_call_id": tool_call_id,
                "name": name,
                "child_session_id": child_session_id,
                "child_stream_key": child_session_id,
                "run_id": started.run_id,
            }
        )
    return started_actions


dispatch_runtime_actions_step.max_retries = 0


@workflow.workflow
async def run_turn(
    completion_token: str,
    stream_key: str,
    session_id: str,
    turn_id: str,
    session_state: dict[str, object],
    prompt: str | None = None,
    runtime_results: list[dict[str, object]] | None = None,
    model_id: str = MODEL_ID,
    source: SessionSource = "root",
    parent: dict[str, str] | None = None,
) -> str:
    """Short-lived turn harness. Returns a small driver action through a hook."""
    snapshot = await read_session_step(session_state)
    messages = _session_messages(snapshot)
    output = str(snapshot.get("output") or "")
    model_round = 1
    try:
        if runtime_results is not None:
            raw_pending = snapshot.get("pending_runtime_actions")
            pending_actions = (
                [
                    cast(dict[str, object], item)
                    for item in raw_pending
                    if isinstance(item, dict)
                ]
                if isinstance(raw_pending, list)
                else []
            )
            if not pending_actions:
                raise RuntimeError("runtime results arrived without pending actions")
            raw_pending_tool_results = snapshot.get("pending_tool_results")
            pending_tool_results = (
                [
                    cast(dict[str, object], item)
                    for item in raw_pending_tool_results
                    if isinstance(item, dict)
                ]
                if isinstance(raw_pending_tool_results, list)
                else []
            )

            tool_calls: dict[str, ai.messages.ToolCallPart] = {}
            tool_call_order: list[str] = []
            for message_data in messages:
                message = ai.messages.Message.model_validate(message_data)
                message_tool_calls = message.tool_calls
                if message_tool_calls:
                    tool_call_order = [
                        tool_call.tool_call_id for tool_call in message_tool_calls
                    ]
                for tool_call in message_tool_calls:
                    tool_calls[tool_call.tool_call_id] = tool_call

            results_by_id = {
                result.tool_call_id: result
                for result in (
                    RuntimeActionResult.model_validate(item) for item in runtime_results
                )
            }
            result_parts_by_id: dict[str, ai.messages.ToolResultPart] = {}
            for item in pending_tool_results:
                tool_call_id = str(item.get("tool_call_id") or "")
                matching_call = tool_calls.get(tool_call_id)
                if matching_call is None:
                    raise RuntimeError(f"missing pending tool call: {tool_call_id}")
                call_index_raw = item.get("call_index")
                call_index = call_index_raw if type(call_index_raw) is int else 0
                model_round_raw = item.get("model_round")
                action_model_round = (
                    model_round_raw if type(model_round_raw) is int else model_round
                )
                result_parts_by_id[tool_call_id] = _tool_result_part(
                    matching_call,
                    item.get("result"),
                    turn_id=turn_id,
                    model_round=action_model_round,
                    call_index=call_index,
                    is_error=item.get("is_error") is True,
                )
            for action in pending_actions:
                tool_call_id = str(action.get("tool_call_id") or "")
                runtime_result = results_by_id.get(tool_call_id)
                matching_call = tool_calls.get(tool_call_id)
                if runtime_result is None or matching_call is None:
                    raise RuntimeError(f"missing runtime action result: {tool_call_id}")

                call_index_raw = action.get("call_index")
                call_index = call_index_raw if type(call_index_raw) is int else 0
                model_round_raw = action.get("model_round")
                action_model_round = (
                    model_round_raw if type(model_round_raw) is int else model_round
                )
                model_round = max(model_round, action_model_round + 1)
                result_parts_by_id[tool_call_id] = _tool_result_part(
                    matching_call,
                    runtime_result.result,
                    turn_id=turn_id,
                    model_round=action_model_round,
                    call_index=call_index,
                    is_error=runtime_result.is_error,
                )
                if not runtime_result.is_error:
                    await write_agent_event(
                        stream_key,
                        session_id,
                        "subagent.completed",
                        {
                            "tool_call_id": runtime_result.tool_call_id,
                            "name": runtime_result.tool_name,
                            "output": runtime_result.result,
                        },
                        turn_id=turn_id,
                        source=source,
                        parent=parent,
                    )
                await write_agent_event(
                    stream_key,
                    session_id,
                    "action.result",
                    {
                        "tool_call_id": runtime_result.tool_call_id,
                        "tool_name": runtime_result.tool_name,
                        "result": runtime_result.result,
                        "is_error": runtime_result.is_error,
                    },
                    turn_id=turn_id,
                    source=source,
                    parent=parent,
                )

            ordered_results = [
                result_parts_by_id[tool_call_id]
                for tool_call_id in tool_call_order
                if tool_call_id in result_parts_by_id
            ]
            missing_results = [
                tool_call_id
                for tool_call_id in tool_call_order
                if tool_call_id not in result_parts_by_id
            ]
            if missing_results:
                raise RuntimeError(
                    f"missing tool results after runtime action: {missing_results}"
                )
            tool_message = ai.tool_message(*ordered_results).model_copy(
                update={"id": f"{turn_id}-tool-runtime", "turn_id": turn_id}
            )
            messages.append(_dump_message(tool_message))
            await write_agent_event(
                stream_key,
                session_id,
                "message.committed",
                {"message": _dump_message(tool_message)},
                turn_id=turn_id,
                source=source,
                parent=parent,
            )
            session_state = await write_session_step(session_state, messages, output)
        elif prompt is not None:
            user_message = ai.user_message(prompt).model_copy(
                update={"id": f"{turn_id}-user", "turn_id": turn_id}
            )
            messages.append(_dump_message(user_message))
            await write_agent_event(
                stream_key,
                session_id,
                "message.committed",
                {"message": _dump_message(user_message)},
                turn_id=turn_id,
                source=source,
                parent=parent,
            )
            session_state = await write_session_step(session_state, messages, output)
        else:
            raise RuntimeError("turn requires a prompt or runtime action results")

        while True:
            result = await stream_llm(
                stream_key,
                messages,
                tool_schemas=_agent_tool_schemas(source),
                model_id=model_id,
                session_id=session_id,
                turn_id=turn_id,
                source=source,
                parent=parent,
            )
            assistant_message = ai.messages.Message.model_validate(result)
            if assistant_message.turn_id is None:
                assistant_message = assistant_message.model_copy(
                    update={"turn_id": turn_id}
                )
            messages.append(_dump_message(assistant_message))
            await write_agent_event(
                stream_key,
                session_id,
                "message.committed",
                {"message": _dump_message(assistant_message)},
                turn_id=turn_id,
                source=source,
                parent=parent,
            )
            session_state = await write_session_step(session_state, messages, output)

            if not assistant_message.tool_calls:
                output = assistant_message.text
                session_state = await write_session_step(
                    session_state, messages, output
                )
                await notify_turn_completion(
                    completion_token,
                    {
                        "action": "done",
                        "session": session_state,
                        "output": output,
                    },
                )
                return output

            await write_agent_event(
                stream_key,
                session_id,
                "actions.requested",
                {
                    "actions": [
                        {
                            "tool_call_id": tool_call.tool_call_id,
                            "tool_name": tool_call.tool_name,
                            "tool_args": tool_call.tool_args,
                        }
                        for tool_call in assistant_message.tool_calls
                    ]
                },
                turn_id=turn_id,
                source=source,
                parent=parent,
            )

            tool_results: dict[str, ai.messages.ToolResultPart] = {}
            completed_tool_results: list[dict[str, object]] = []
            runtime_actions: list[dict[str, object]] = []
            for call_index, tool_call in enumerate(
                assistant_message.tool_calls,
                start=1,
            ):
                try:
                    args = _parse_tool_args(tool_call)
                    match tool_call.tool_name:
                        case "bash":
                            command = args.get("command")
                            if not isinstance(command, str) or not command:
                                raise ValueError("bash requires command")
                            timeout_value = args.get("timeout")
                            timeout = (
                                timeout_value if type(timeout_value) is int else None
                            )
                            tool_result = await _bash(command, timeout)
                            tool_results[tool_call.tool_call_id] = _tool_result_part(
                                tool_call,
                                tool_result,
                                turn_id=turn_id,
                                model_round=model_round,
                                call_index=call_index,
                            )
                            await write_agent_event(
                                stream_key,
                                session_id,
                                "action.result",
                                {
                                    "tool_call_id": tool_call.tool_call_id,
                                    "tool_name": tool_call.tool_name,
                                    "result": tool_result,
                                    "is_error": False,
                                },
                                turn_id=turn_id,
                                source=source,
                                parent=parent,
                            )
                            completed_tool_results.append(
                                {
                                    "tool_call_id": tool_call.tool_call_id,
                                    "tool_name": tool_call.tool_name,
                                    "result": tool_result,
                                    "is_error": False,
                                    "call_index": call_index,
                                    "model_round": model_round,
                                }
                            )

                        case "web_fetch":
                            url = args.get("url")
                            if not isinstance(url, str) or not url:
                                raise ValueError("web_fetch requires url")
                            method = str(args.get("method") or "GET")
                            headers = str(args.get("headers") or "")
                            body = str(args.get("body") or "")
                            tool_result = await _web_fetch(url, method, headers, body)
                            tool_results[tool_call.tool_call_id] = _tool_result_part(
                                tool_call,
                                tool_result,
                                turn_id=turn_id,
                                model_round=model_round,
                                call_index=call_index,
                            )
                            await write_agent_event(
                                stream_key,
                                session_id,
                                "action.result",
                                {
                                    "tool_call_id": tool_call.tool_call_id,
                                    "tool_name": tool_call.tool_name,
                                    "result": tool_result,
                                    "is_error": False,
                                },
                                turn_id=turn_id,
                                source=source,
                                parent=parent,
                            )
                            completed_tool_results.append(
                                {
                                    "tool_call_id": tool_call.tool_call_id,
                                    "tool_name": tool_call.tool_name,
                                    "result": tool_result,
                                    "is_error": False,
                                    "call_index": call_index,
                                    "model_round": model_round,
                                }
                            )

                        case "subagent":
                            if source == "subagent":
                                raise ValueError("subagent tool is not available here")
                            child_prompt = args.get("prompt")
                            if not isinstance(child_prompt, str) or not child_prompt:
                                raise ValueError("subagent requires prompt")
                            runtime_actions.append(
                                {
                                    "tool_call_id": tool_call.tool_call_id,
                                    "tool_name": tool_call.tool_name,
                                    "prompt": child_prompt,
                                    "name": str(args.get("name") or "subagent"),
                                    "call_index": call_index,
                                    "model_round": model_round,
                                }
                            )

                        case _:
                            raise ValueError(f"unknown tool: {tool_call.tool_name}")
                except Exception as error:
                    error_text = str(error)
                    tool_results[tool_call.tool_call_id] = _tool_result_part(
                        tool_call,
                        error_text,
                        turn_id=turn_id,
                        model_round=model_round,
                        call_index=call_index,
                        is_error=True,
                    )
                    await write_agent_event(
                        stream_key,
                        session_id,
                        "action.result",
                        {
                            "tool_call_id": tool_call.tool_call_id,
                            "tool_name": tool_call.tool_name,
                            "result": error_text,
                            "is_error": True,
                        },
                        turn_id=turn_id,
                        source=source,
                        parent=parent,
                    )
                    completed_tool_results.append(
                        {
                            "tool_call_id": tool_call.tool_call_id,
                            "tool_name": tool_call.tool_name,
                            "result": error_text,
                            "is_error": True,
                            "call_index": call_index,
                            "model_round": model_round,
                        }
                    )

            if runtime_actions:
                session_state = await write_session_step(
                    session_state,
                    messages,
                    output,
                    pending_runtime_actions=runtime_actions,
                    pending_tool_results=completed_tool_results,
                )
                await notify_turn_completion(
                    completion_token,
                    {
                        "action": "runtime_action",
                        "session": session_state,
                        "output": output,
                        "actions": runtime_actions,
                    },
                )
                return output

            ordered_results = [
                tool_results[tool_call.tool_call_id]
                for tool_call in assistant_message.tool_calls
                if tool_call.tool_call_id in tool_results
            ]
            if ordered_results:
                tool_message = ai.tool_message(*ordered_results).model_copy(
                    update={
                        "id": f"{turn_id}-tool-{model_round}",
                        "turn_id": turn_id,
                    }
                )
                messages.append(_dump_message(tool_message))
                await write_agent_event(
                    stream_key,
                    session_id,
                    "message.committed",
                    {"message": _dump_message(tool_message)},
                    turn_id=turn_id,
                    source=source,
                    parent=parent,
                )
                session_state = await write_session_step(
                    session_state, messages, output
                )
            model_round += 1
    except Exception as error:
        await notify_turn_completion(
            completion_token,
            {
                "action": "error",
                "session": session_state,
                "output": output,
                "error": str(error),
            },
        )
        raise


@workflow.step
async def dispatch_turn_step(
    completion_token: str,
    stream_key: str,
    session_id: str,
    turn_id: str,
    session_state: dict[str, object],
    prompt: str | None = None,
    runtime_results: list[dict[str, object]] | None = None,
    model_id: str = MODEL_ID,
    source: SessionSource = "root",
    parent: dict[str, str] | None = None,
) -> str:
    """Start the child agent workflow for one turn."""
    from agent import durable_stream

    started = await vercel.workflow.start(
        run_turn,
        completion_token,
        stream_key,
        session_id,
        turn_id,
        session_state,
        prompt,
        runtime_results,
        model_id,
        source,
        parent,
    )
    stream = durable_stream.get_writable(stream_key)
    await stream.write(
        durable_stream.AgentStreamEvent(
            kind="turn.dispatched",
            session_id=session_id,
            turn_id=turn_id,
            source=source,
            parent=parent,
            data={"run_id": started.run_id},
        )
    )
    return started.run_id


dispatch_turn_step.max_retries = 0


@workflow.workflow
async def run_session(
    session_id: str,
    stream_key: str,
    initial_prompt: str | None = None,
    source: SessionSource = "root",
    parent: dict[str, str] | None = None,
) -> str:
    """Long-lived durable session workflow."""
    session_state = await create_session_step(session_id, stream_key, source)
    turn_index = 1
    pending_prompt = initial_prompt
    output = ""
    await write_agent_event(
        stream_key,
        session_id,
        "session.started",
        {"stream_key": stream_key},
        source=source,
        parent=parent,
        reset=True,
    )

    while True:
        if pending_prompt is None:
            continuation_token = f"seal-session:{session_id}:{turn_index}"
            hook = SessionTurn.wait(token=continuation_token)
            await write_agent_event(
                stream_key,
                session_id,
                "session.waiting",
                {
                    "continuation_token": continuation_token,
                    "turn_index": turn_index,
                },
                source=source,
                parent=parent,
            )
            turn = await hook
            hook.dispose()
            if turn is None:
                continue
            if turn.close:
                await write_agent_event(
                    stream_key,
                    session_id,
                    "session.completed",
                    {"turn_index": turn_index},
                    source=source,
                    parent=parent,
                )
                return output
            prompt = turn.prompt or ""
        else:
            prompt = pending_prompt
            pending_prompt = None

        turn_id = f"{session_id}-turn-{turn_index}"
        await write_agent_event(
            stream_key,
            session_id,
            "turn.started",
            {"turn_index": turn_index},
            turn_id=turn_id,
            source=source,
            parent=parent,
        )
        completion: TurnCompletion | None = None
        runtime_results: list[dict[str, object]] | None = None
        prompt_for_turn: str | None = prompt
        runtime_round = 1
        while True:
            completion_token = f"seal-turn:{session_id}:{turn_index}:{runtime_round}"
            completion_hook = TurnCompletion.wait(token=completion_token)
            try:
                await dispatch_turn_step(
                    completion_token,
                    stream_key,
                    session_id,
                    turn_id,
                    session_state,
                    prompt_for_turn,
                    runtime_results,
                    MODEL_ID,
                    source,
                    parent,
                )
                completion = await completion_hook
            except Exception as error:
                completion_hook.dispose()
                error_text = str(error)
                await write_agent_event(
                    stream_key,
                    session_id,
                    "turn.failed",
                    {"turn_index": turn_index, "error": error_text},
                    turn_id=turn_id,
                    source=source,
                    parent=parent,
                )
                await write_agent_event(
                    stream_key,
                    session_id,
                    "session.failed",
                    {"turn_index": turn_index, "error": error_text},
                    turn_id=turn_id,
                    source=source,
                    parent=parent,
                )
                if source == "subagent" and parent is not None:
                    result_token = parent.get("runtime_result_token")
                    if result_token is not None:
                        await notify_runtime_action_result(
                            result_token,
                            parent.get("tool_call_id", ""),
                            parent.get("name", "subagent"),
                            error_text,
                            is_error=True,
                        )
                raise
            completion_hook.dispose()

            if completion is None:
                break
            if completion.session is not None:
                session_state = completion.session
            output = completion.output or str(session_state.get("output") or output)
            if completion.action != "runtime_action":
                break
            if not completion.actions:
                break

            runtime_hooks: list[vercel.workflow.HookEvent[RuntimeActionResult]] = []
            dispatch_actions: list[dict[str, object]] = []
            try:
                try:
                    for action_index, action in enumerate(completion.actions, start=1):
                        result_token = (
                            f"seal-runtime:{session_id}:{turn_id}:"
                            f"{runtime_round}:{action_index}"
                        )
                        runtime_hooks.append(
                            RuntimeActionResult.wait(token=result_token)
                        )
                        action_with_token = dict(action)
                        action_with_token["runtime_result_token"] = result_token
                        dispatch_actions.append(action_with_token)
                    await dispatch_runtime_actions_step(
                        stream_key,
                        session_id,
                        turn_id,
                        dispatch_actions,
                        source=source,
                        parent=parent,
                    )
                    runtime_results = []
                    for runtime_hook in runtime_hooks:
                        runtime_result = await runtime_hook
                        if runtime_result is not None:
                            runtime_results.append(
                                runtime_result.model_dump(mode="json")
                            )
                except Exception as error:
                    error_text = str(error)
                    await write_agent_event(
                        stream_key,
                        session_id,
                        "turn.failed",
                        {"turn_index": turn_index, "error": error_text},
                        turn_id=turn_id,
                        source=source,
                        parent=parent,
                    )
                    await write_agent_event(
                        stream_key,
                        session_id,
                        "session.failed",
                        {"turn_index": turn_index, "error": error_text},
                        turn_id=turn_id,
                        source=source,
                        parent=parent,
                    )
                    if source == "subagent" and parent is not None:
                        result_token = parent.get("runtime_result_token")
                        if result_token is not None:
                            await notify_runtime_action_result(
                                result_token,
                                parent.get("tool_call_id", ""),
                                parent.get("name", "subagent"),
                                error_text,
                                is_error=True,
                            )
                    raise
            finally:
                for runtime_hook in runtime_hooks:
                    runtime_hook.dispose()
            prompt_for_turn = None
            runtime_round += 1

        if completion is None:
            continue
        if completion.action != "done":
            error_text = completion.error or f"turn ended with {completion.action}"
            await write_agent_event(
                stream_key,
                session_id,
                "turn.failed",
                {"turn_index": turn_index, "error": error_text},
                turn_id=turn_id,
                source=source,
                parent=parent,
            )
            await write_agent_event(
                stream_key,
                session_id,
                "session.failed",
                {"turn_index": turn_index, "error": error_text},
                turn_id=turn_id,
                source=source,
                parent=parent,
            )
            if source == "subagent" and parent is not None:
                result_token = parent.get("runtime_result_token")
                if result_token is not None:
                    await notify_runtime_action_result(
                        result_token,
                        parent.get("tool_call_id", ""),
                        parent.get("name", "subagent"),
                        error_text,
                        is_error=True,
                    )
            raise RuntimeError(error_text)

        await write_agent_event(
            stream_key,
            session_id,
            "turn.completed",
            {"turn_index": turn_index, "output": output},
            turn_id=turn_id,
            source=source,
            parent=parent,
        )
        if source == "subagent":
            await write_agent_event(
                stream_key,
                session_id,
                "session.completed",
                {"turn_index": turn_index},
                source=source,
                parent=parent,
            )
            if parent is not None:
                result_token = parent.get("runtime_result_token")
                if result_token is not None:
                    await notify_runtime_action_result(
                        result_token,
                        parent.get("tool_call_id", ""),
                        parent.get("name", "subagent"),
                        output,
                    )
            return output
        turn_index += 1
