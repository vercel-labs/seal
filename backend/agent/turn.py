import asyncio
import dataclasses
import json
import traceback
from collections.abc import AsyncGenerator
from typing import Any, ClassVar, cast

import ai

from agent import proto, stream, workflow

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


# end of hack


# hack: smuggle hidden Message.replay and ToolCallPart.cached_result
# through llm_step json boundary. they are stamped by agent.run before the loop.
# without this replay doesn't work.
def _dump_message(message: ai.messages.Message) -> dict[str, object]:
    data = message.model_dump(mode="json")
    data["_replay"] = message.replay
    cached: dict[str, object] = {}
    for part in message.tool_calls:
        if part.cached_result is not None:
            cached[part.tool_call_id] = part.cached_result.model_dump(mode="json")
    if cached:
        data["_cached_results"] = cached
    return data


def _load_message(data: dict[str, object]) -> ai.messages.Message:
    replay = bool(data.pop("_replay", False))
    cached_raw = cast(dict[str, object], data.pop("_cached_results", {}) or {})
    cached = {
        tool_call_id: ai.messages.ToolResultPart.model_validate(result)
        for tool_call_id, result in cached_raw.items()
    }
    message = ai.messages.Message.model_validate(data)
    if cached:
        new_parts: list[ai.messages.Part] = []
        for part in message.parts:
            if (
                isinstance(part, ai.messages.ToolCallPart)
                and part.tool_call_id in cached
            ):
                part = part.model_copy(
                    update={"cached_result": cached[part.tool_call_id]}
                )
            new_parts.append(part)
        message = message.model_copy(update={"parts": new_parts})
    if replay:
        message = message.model_copy(update={"replay": True})
    return message


# end of hack


@workflow.step
async def llm_step(
    model_id: str,
    messages_data: list[dict[str, object]],
    tools_data: list[dict[str, object]],
    session_id: str | None,
) -> dict[str, object]:
    model = ai.get_model(model_id)
    messages = [_load_message(message) for message in messages_data]
    # hack: ai.Tool erases args type so we have to reconstruct them manually
    tools: list[ai.Tool] = []
    for tool in tools_data:
        if tool.get("kind") == "function":
            tool = {
                **tool,
                "args": ai.tools.FunctionToolArgs.model_validate(tool["args"]),
            }
        tools.append(ai.Tool.model_validate(tool))
    # end of hack

    writer = await stream.get_writable(session_id) if session_id else None
    message: ai.messages.Message | None = None

    async with ai.stream(model, messages, tools=tools) as model_stream:
        async for e in model_stream:
            if writer is not None and not getattr(e, "replay", False):
                await writer.write(e)
            if isinstance(e, ai.events.StreamEnd):
                message = e.message

        if message is None:
            message = model_stream.message

    assert message is not None
    # hack: special dump to preserve replay so the loop's context.add
    # skips the replayed turn
    return _dump_message(message)


llm_step.max_retries = 0


@workflow.step
async def write_event(
    # writes one stream event (agent or lifecycle) to the durable stream
    session_id: str,
    event_data: dict[str, object],
) -> None:
    writer = await stream.get_writable(session_id)
    await writer.write(event_data)


write_event.max_retries = 0


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


@ai.tool(require_approval=True)
async def bash(command: str, timeout: int | None = None) -> str:
    """Execute a bash command. Use timeout in seconds to limit long-running commands."""
    return await _bash(command, timeout)


# subagent (task) sessions cannot surface tool approvals to a human and would
# deadlock on a gated tool, so they run an ungated copy of the same tool.
bash_ungated = dataclasses.replace(
    bash, tool=bash.tool.model_copy(update={"require_approval": False})
)


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


# the driver stores the child's full transcript (a MessageBundle) as this tool's
# result for rich UI rendering. declaring MessageAggregator lets Agent.run reduce
# that bundle to the summary string the model sees: _populate_model_inputs looks
# up this aggregator by tool name and calls to_model_input(result) each turn,
# since the model-facing value is not serialized across the durable boundary.
subagent = dataclasses.replace(subagent, aggregator=ai.agents.MessageAggregator)


class DurableAgent(ai.Agent):
    # bash is gated/ungated per mode, so it is supplied via tools=, not here.
    TOOLS: ClassVar[list[ai.AgentTool]] = [web_fetch]

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        params = context.params if isinstance(context.params, dict) else {}
        model_id = str(params.get("model_id") or MODEL_ID)

        control = cast(dict[str, object], params["control"])
        session_id = cast(str | None, params.get("session_id"))

        while context.keep_running():
            result = await llm_step(
                model_id,
                [_dump_message(message) for message in context.messages],
                # hack: have to use serialize_as_any because ai.Tool erases args type
                [
                    tool.model_dump(mode="json", serialize_as_any=True)
                    for tool in context.tools
                ],
                session_id,
            )

            assistant_message = _load_message(result)
            context.add(assistant_message)

            pending_subagents: list[proto.SubagentRequest] = []
            cached_results: list[ai.messages.ToolResultPart] = []
            tool_message: ai.messages.Message | None = None

            async with ai.ToolRunner() as runner:
                for tool_call in assistant_message.tool_calls:
                    if tool_call.cached_result is not None:
                        # hack: special treatment of replayed results
                        cached_results.append(tool_call.cached_result)
                    elif tool_call.tool_name == "subagent":
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
                        await write_event(session_id, event.model_dump(mode="json"))
                    yield event

                # hack: special treatment of replayed results
                for cached_result in cached_results:
                    tool_message = ai.tool_message(cached_result)
                    event = ai.events.ToolCallResult(
                        message=tool_message,
                        results=tool_message.tool_results,
                    )
                    if session_id is not None:
                        await write_event(session_id, event.model_dump(mode="json"))
                    yield event
                    runner.add_result(event)

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


@workflow.step
async def resume_turn_hook(token: str, output_data: dict[str, Any]) -> None:
    # resume() is a side effect, so it must run in a step. the driver may not
    # have parked on the hook yet, so retry while it is missing.
    hook = proto.TurnHook(output=proto.TurnOutput.model_validate(output_data))
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except RuntimeError as error:
            message = str(error).lower()
            if attempt == 39 or "not found" not in message:
                raise
            await asyncio.sleep(0.05)


resume_turn_hook.max_retries = 0


# runs one agent turn, maybe requests subagents
@workflow.workflow
async def run_turn(turn_input: dict[str, Any]) -> None:
    try:
        return await _run_turn(turn_input)
    except Exception:
        print(
            f"[seal] run_turn failed:\n{traceback.format_exc()}",
            flush=True,
        )
        # XXX: some sort of cleanup?
        raise


async def _run_turn(turn_input: dict[str, Any]) -> None:
    _turn_input = proto.TurnInput.model_validate(turn_input)
    messages = _turn_input.messages

    # messages should already contain either the user message
    # or the tool result message, so no need to do anything

    extra_tools = [bash_ungated] if _turn_input.mode == "task" else [bash, subagent]
    agent = DurableAgent(tools=extra_tools)

    # side-channel for exfiltrating subagent requests out of the loop
    control: dict[str, object] = {
        "pending_subagents": [],
    }

    # pre-register tool approvals
    for tool_approval in _turn_input.tool_approvals:
        ai.resolve_hook(
            f"{proto.TOOL_APPROVAL_HOOK_PREFIX}{tool_approval.tool_call_id}",
            {"granted": tool_approval.granted, "reason": tool_approval.reason},
        )

    # new tool approval requests to send to session
    tool_approval_requests: list[proto.ToolApprovalRequest] = []

    try:
        async with agent.run(
            _workflow_model(),
            messages,
            params={
                "model_id": MODEL_ID,
                "session_id": _turn_input.session_id,
                "control": control,
            },
        ) as run:
            async for event in run:
                # monitor the stream for hook events and interrupt on them.
                if (
                    isinstance(event, ai.events.HookEvent)
                    and event.hook.status == "pending"
                ):
                    hook = event.hook
                    if hook.hook_id.startswith(proto.TOOL_APPROVAL_HOOK_PREFIX):
                        # HookEvents ride the runtime queue, not runner.events(),
                        # so the loop never wrote this to the durable stream. write
                        # it here so the AI SDK UI adapter emits the approval
                        # request part (it skips the is_hook_pending tool result
                        # and waits for the pending HookEvent to drive the UI).
                        await write_event(
                            _turn_input.session_id,
                            event.model_dump(mode="json"),
                        )
                        tool_approval_requests.append(
                            proto.ToolApprovalRequest(
                                tool_call_id=hook.hook_id[
                                    len(proto.TOOL_APPROVAL_HOOK_PREFIX) :
                                ],
                                tool_name=str(hook.metadata.get("tool", "")),
                                args=cast(
                                    dict[str, Any], hook.metadata.get("kwargs", {})
                                ),
                            )
                        )
                    ai.abort_pending_hook(hook)

            messages = run.messages
    except Exception as error:
        output = proto.TurnOutput(
            kind="error",
            messages=messages,
            error=f"{type(error).__name__}: {error}",
        )
        print(
            f"[seal] error in run_turn:\n{traceback.format_exc()}",
            flush=True,
        )
    else:
        # create normal output if the run has completed successfully
        subagent_requests = [
            proto.SubagentRequest.model_validate(item)
            for item in cast(list[dict[str, object]], control["pending_subagents"])
        ]

        has_pending = bool(subagent_requests or tool_approval_requests)
        if _turn_input.mode == "infinite":
            output_kind = "pending_requests" if has_pending else "suspend"
        else:
            # task (subagent) sessions never gate; pending requests would deadlock.
            output_kind = "done"

        output = proto.TurnOutput(
            kind=output_kind,
            messages=messages,
            pending_requests=[*subagent_requests, *tool_approval_requests],
        )

    # notify session that the turn is complete
    await resume_turn_hook(_turn_input.turn_hook_token, output.model_dump(mode="json"))
