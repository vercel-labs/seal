import asyncio
import dataclasses
import json
import traceback
from collections.abc import AsyncGenerator, Sequence
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
            if writer is not None and not getattr(e, "replay", False):
                await writer.write(e)
            if isinstance(e, ai.events.StreamEnd):
                message = e.message

        if message is None:
            message = model_stream.message

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

    # ``run(params=...)`` is typed inference params now, so the durable plumbing
    # (model id, stream target, subagent side-channel) lives on the instance.
    def __init__(
        self,
        *,
        tools: Sequence[ai.AgentTool | ai.Tool] | None = None,
        session_id: str | None = None,
    ) -> None:
        super().__init__(tools=tools)
        self.session_id = session_id
        # side-channel for exfiltrating subagent requests out of the loop
        self.pending_subagents: list[dict[str, object]] = []

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        model_id = context.model.id
        session_id = self.session_id

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
            tool_message: ai.messages.Message | None = None

            async with ai.ToolRunner() as runner:
                for tool_call in assistant_message.tool_calls:
                    if tool_call.cached_result is not None:
                        # hack: special treatment of replayed results
                        runner.add_result(ai.tool_result(tool_call.cached_result))
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

                tool_message = runner.get_tool_message()

            if tool_message is not None:
                context.add(tool_message)
            elif pending_subagents:
                # guarantee a trailing tool message for the driver to extend
                context.add(ai.messages.Message(role="tool", parts=[]))

            if pending_subagents:
                self.pending_subagents = [
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
    messages = _turn_input.messages

    # messages should already contain either the user message
    # or the tool result message, so no need to do anything

    extra_tools = [bash_ungated] if _turn_input.mode == "task" else [bash, subagent]
    agent = DurableAgent(
        tools=extra_tools,
        session_id=_turn_input.session_id,
    )

    # pre-register tool approvals
    for tool_approval in _turn_input.tool_approvals:
        ai.resolve_hook(
            f"{proto.TOOL_APPROVAL_HOOK_PREFIX}{tool_approval.tool_call_id}",
            {"granted": tool_approval.granted, "reason": tool_approval.reason},
        )

    # new tool approval requests to send to session
    tool_approval_requests: list[proto.ToolApprovalRequest] = []

    try:
        model = ai.get_model(MODEL_ID)
        async with agent.run(model, messages) as run:
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
            for item in agent.pending_subagents
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
