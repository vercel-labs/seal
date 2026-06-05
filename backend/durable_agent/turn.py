import asyncio
import dataclasses
import json
from collections.abc import AsyncGenerator
from typing import Any, ClassVar, cast

import ai

from durable_agent import proto, stream, workflow

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
            if writer is not None:
                await writer.write(e)
            if isinstance(e, ai.events.StreamEnd):
                message = e.message

    assert message is not None
    return message.model_dump(mode="json")


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


# subagent (task) sessions cannot surface approvals to a human and would
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
                [message.model_dump(mode="json") for message in context.messages],
                # hack: have to use serialize_as_any because ai.Tool erases args type
                [
                    tool.model_dump(mode="json", serialize_as_any=True)
                    for tool in context.tools
                ],
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
                        await write_event(
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
    _turn_input = proto.TurnInput.model_validate(turn_input)

    # messages should already contain either the user message
    # or the tool result message, so no need to do anything

    # task sessions get the ungated bash and no subagent tool; the main
    # session gets the gated bash and may delegate.
    extra_tools = [bash_ungated] if _turn_input.mode == "task" else [bash, subagent]
    agent = DurableAgent(tools=extra_tools)

    # side-channel for exfiltrating subagent requests out of the loop
    control: dict[str, object] = {
        "pending_subagents": [],
    }

    # pre-register approvals; the loop replays up to each gated tool and the
    # hook returns immediately instead of suspending, so the tool runs (or is
    # rejected) for real and overwrites the prior pending placeholder.
    for approval in _turn_input.approvals:
        ai.resolve_hook(
            f"{proto.APPROVAL_HOOK_PREFIX}{approval.tool_call_id}",
            {"granted": approval.granted, "reason": approval.reason},
        )

    # gated tool calls the loop suspended on; collected from the run's hook
    # events (the SDK carries tool name + args in the hook metadata) so we do
    # not need the control side-channel the subagent path uses.
    approval_requests: list[proto.ToolApprovalRequest] = []

    async with agent.run(
        _workflow_model(),
        _turn_input.messages,
        params={
            "model_id": MODEL_ID,
            "session_id": _turn_input.session_id,
            "control": control,
        },
    ) as run:
        async for event in run:
            # monitor the stream for hook events and interrupt on them
            # by now the event has been processed by durable stream,
            # so nothing else to do with it
            if (
                isinstance(event, ai.events.HookEvent)
                and event.hook.status == "pending"
            ):
                hook = event.hook
                if hook.hook_id.startswith(proto.APPROVAL_HOOK_PREFIX):
                    approval_requests.append(
                        proto.ToolApprovalRequest(
                            tool_call_id=hook.hook_id[
                                len(proto.APPROVAL_HOOK_PREFIX) :
                            ],
                            tool_name=str(hook.metadata.get("tool", "")),
                            args=cast(dict[str, Any], hook.metadata.get("kwargs", {})),
                        )
                    )
                ai.abort_pending_hook(hook)

        messages = run.messages

    subagent_requests = [
        proto.SubagentRequest.model_validate(item)
        for item in cast(list[dict[str, object]], control["pending_subagents"])
    ]

    has_pending = bool(subagent_requests or approval_requests)
    if _turn_input.mode == "infinite":
        output_kind = "pending_requests" if has_pending else "suspend"
    else:
        # task (subagent) sessions never gate; pending requests would deadlock.
        output_kind = "done"

    output = proto.TurnOutput(
        kind=output_kind,
        messages=messages,
        pending_requests=[*subagent_requests, *approval_requests],
    )

    # notify session that the turn is complete
    await resume_turn_hook(_turn_input.turn_hook_token, output.model_dump(mode="json"))
