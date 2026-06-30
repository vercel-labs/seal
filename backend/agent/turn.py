import asyncio
import contextvars
import dataclasses
import random
import traceback
from collections.abc import AsyncGenerator, Sequence
from typing import Any, ClassVar, cast

import ai
import vercel.workflow

from agent import proto, stream, util, workflow

MODEL_ID = "gateway:anthropic/claude-sonnet-4.6"
SYSTEM_PROMPT = (
    "You are Seal, a coding assistant. Use bash, web_fetch, and subagent to "
    "inspect the environment, gather information, and delegate focused work."
)
SUBAGENT_SYSTEM_PROMPT = (
    "You are a focused Seal subagent. Use bash and web_fetch when useful, then "
    "answer the delegated task directly."
)


@workflow.step(max_retries=0)
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


@workflow.step(max_retries=0)
async def write_event(
    # writes one stream event (agent or lifecycle) to the durable stream
    session_id: str,
    event_data: dict[str, object],
) -> None:
    writer = await stream.get_writable(session_id)
    await writer.write(event_data)


# closes a durable event stream once the owning session is terminal.
@workflow.step(max_retries=0)
async def close_stream(session_id: str) -> None:
    writer = await stream.get_writable(session_id)
    await writer.close()


@ai.tool(require_approval=True)
@workflow.step(max_retries=0)
async def bash(command: str, timeout: int | None = None) -> str:
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


# subagent (task) sessions cannot surface tool approvals to a human and would
# deadlock on a gated tool, so they run an ungated copy of the same tool.
bash_ungated = dataclasses.replace(
    bash, tool=bash.tool.model_copy(update={"require_approval": False})
)


@ai.tool
@workflow.step(max_retries=0)
async def web_fetch(
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
async def spawn_subagent_turn(turn_input: dict[str, object]) -> dict[str, object]:
    # a subagent is just one ungated turn writing to its own stream.
    started = await vercel.workflow.start(run_turn, turn_input)
    return {"run_id": started.run_id}


# the running tool call's session and id, set by the loop around each schedule so
# a tool can reach them without smuggling args. tasks copy the context at
# creation, so each tool sees its own call.
tool_call_context: contextvars.ContextVar[tuple[str, str]] = contextvars.ContextVar(
    "tool_call_context"
)


# hack: the only way the library currently supports transforming a
# tool result before sending it to the model is by using an
# aggregator, so we use MessageAggregator without actually being a
# generator.
@ai.tool(aggregator=ai.agents.MessageAggregator)  # type: ignore
@util.print_traceback
async def subagent(prompt: str, name: str | None = None) -> ai.agents.MessageBundle:
    """Delegate a focused task to a child agent and return its answer."""
    session_id, tool_call_id = tool_call_context.get()
    name = name or "subagent"
    child_session_id = f"{session_id}:child:{tool_call_id}"
    token = f"seal-turn:{child_session_id}:0"
    await write_event(
        session_id,
        stream.subagent_called(
            tool_call_id=tool_call_id, child_session_id=child_session_id, name=name
        ),
    )
    hook = proto.TurnHook.wait(token=token)
    await spawn_subagent_turn(
        proto.TurnInput(
            session_id=child_session_id,
            messages=[
                ai.system_message(SUBAGENT_SYSTEM_PROMPT),
                ai.user_message(prompt),
            ],
            gated=False,
            turn_hook_token=token,
        ).model_dump(mode="json")
    )
    resolution = await hook
    hook.dispose()
    assert resolution is not None
    output = resolution.output
    await write_event(
        session_id,
        stream.subagent_completed(
            tool_call_id=tool_call_id, is_error=output.kind == "error"
        ),
    )
    await close_stream(child_session_id)
    return ai.agents.MessageBundle(
        messages=tuple(m for m in output.messages if m.role in ("assistant", "tool"))
    )


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

            async with ai.ToolRunner() as runner:
                for tool_call in assistant_message.tool_calls:
                    token = tool_call_context.set(
                        (session_id or "", tool_call.tool_call_id)
                    )
                    runner.schedule(context.resolve(tool_call))
                    tool_call_context.reset(token)

                async for event in runner.events():
                    yield event

                tool_message = runner.get_tool_message()

            if tool_message is not None:
                context.add(tool_message)


@workflow.step(max_retries=0)
async def resume_turn_hook(token: str, output_data: dict[str, Any]) -> None:
    # resume() is a side effect, so it must run in a step. the driver may not
    # have parked on the hook yet, so retry while it is missing.
    hook = proto.TurnHook(output=proto.TurnOutput.model_validate(output_data))
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if attempt == 39:
                raise
            await asyncio.sleep(0.05)


# runs one agent turn, maybe parks on a tool approval
@workflow.workflow
# HACK: workflow sets up `random` as a custom seeded thing...
# We ought to make it have something explicit instead
@ai.messages.use_random_async(lambda: random)  # type: ignore
async def run_turn(turn_input: dict[str, Any]) -> None:
    _turn_input = proto.TurnInput.model_validate(turn_input)
    messages = _turn_input.messages

    # messages should already contain either the user message
    # or the tool result message, so no need to do anything

    extra_tools = [bash, subagent] if _turn_input.gated else [bash_ungated]
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
                # N.B: DurableAgent.run filters out most events -- we
                # will only get tool running events and hooks.
                await write_event(
                    _turn_input.session_id,
                    event.model_dump(mode="json"),
                )
                # monitor the stream for hook events and interrupt on them.
                if (
                    isinstance(event, ai.events.HookEvent)
                    and event.hook.status == "pending"
                ):
                    hook = event.hook
                    if hook.hook_id.startswith(proto.TOOL_APPROVAL_HOOK_PREFIX):
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
        # ungated turns never raise approvals, so they fall through to suspend.
        output = proto.TurnOutput(
            kind="pending_requests" if tool_approval_requests else "suspend",
            messages=messages,
            pending_requests=tool_approval_requests,
        )

    # notify session that the turn is complete
    await resume_turn_hook(_turn_input.turn_hook_token, output.model_dump(mode="json"))
