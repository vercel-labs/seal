import asyncio
import contextvars
import dataclasses
import traceback
from collections.abc import AsyncGenerator, Sequence
from typing import Any, ClassVar

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
    turn_span_data: dict[str, object] | None = None,
) -> dict[str, object]:
    model = ai.get_model(model_id)
    messages = [
        ai.messages.Message.model_validate(message) for message in messages_data
    ]
    tools = [ai.Tool.model_validate(tool) for tool in tools_data]

    writer = await stream.get_writable(session_id) if session_id else None
    message: ai.messages.Message | None = None

    # parent this step's spans under the turn's span
    turn_span = (
        ai.experimental_telemetry.Span.model_validate(turn_span_data)
        if turn_span_data
        else None
    )
    async with (
        ai.experimental_telemetry.use_span(turn_span),
        ai.stream(model, messages, tools=tools) as model_stream,
    ):
        async for e in model_stream:
            if writer is not None and not e.replay:
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


@workflow.step(max_retries=0)
async def spawn_subagent_turn(
    turn_input: dict[str, object],
    parent_span_data: dict[str, object] | None = None,
) -> dict[str, object]:
    # a subagent is just one ungated turn writing to its own stream. its span
    payload = dict(turn_input)
    if ai.experimental_telemetry.enabled():
        # create and nest the span for the subagent turn
        parent = (
            ai.experimental_telemetry.Span.model_validate(parent_span_data)
            if parent_span_data
            else None
        )
        turn_span = ai.experimental_telemetry.create_span(
            "turn",
            {"openinference.span.kind": "AGENT"},
            parent=parent,
        ).stamp_start()
        payload["turn_span"] = turn_span.model_dump(mode="json")
    started = await vercel.workflow.start(run_turn, payload)
    return {"run_id": started.run_id}


# the running tool call's context, set by the loop around each schedule so a
# tool can reach it without smuggling args. tasks copy the contextvars at
# creation, so each tool sees its own call.
tool_call_context: contextvars.ContextVar[proto.ToolCallContext] = (
    contextvars.ContextVar("tool_call_context")
)


# hack: the only way the library currently supports transforming a
# tool result before sending it to the model is by using an
# aggregator, so we use MessageAggregator without actually being a
# generator.
@ai.tool(aggregator=ai.agents.MessageAggregator)  # type: ignore
@util.print_traceback
async def subagent(prompt: str, name: str | None = None) -> ai.agents.MessageBundle:
    """Delegate a focused task to a child agent and return its answer."""
    call = tool_call_context.get()
    session_id, tool_call_id = call.session_id, call.tool_call_id
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
        ).model_dump(mode="json"),
        # the child turn's root span nests under this turn's root span.
        call.turn_span.model_dump(mode="json") if call.turn_span else None,
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
        turn_span: ai.experimental_telemetry.Span | None = None,
    ) -> None:
        super().__init__(tools=tools)
        self.session_id = session_id
        self.turn_span = turn_span

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        model_id = context.model.id
        session_id = self.session_id
        turn_span_data = (
            self.turn_span.model_dump(mode="json") if self.turn_span else None
        )

        while context.keep_running():
            result = await llm_step(
                model_id,
                [message.model_dump(mode="json") for message in context.messages],
                [tool.model_dump(mode="json") for tool in context.tools],
                session_id,
                turn_span_data,
            )

            assistant_message = ai.messages.Message.model_validate(result)
            context.add(assistant_message)
            # llm_step streamed this turn out-of-band (straight to the durable
            # stream), so yield the final StreamEnd here for run-blocked
            # tracking, which counts the turn's tool calls from it.
            yield ai.events.StreamEnd(message=assistant_message)

            async with ai.ToolRunner() as runner:
                for tool_call in assistant_message.tool_calls:
                    token = tool_call_context.set(
                        proto.ToolCallContext(
                            session_id=session_id or "",
                            tool_call_id=tool_call.tool_call_id,
                            turn_span=self.turn_span,
                        )
                    )
                    runner.schedule(context.resolve(tool_call))
                    tool_call_context.reset(token)

                async for event in runner.events():
                    # write tool-running events from the producer side so they land
                    # in loop order (results before the next turn's answer); run_turn
                    # only writes HookEvents, which ride the runtime queue instead.
                    if session_id is not None:
                        await write_event(session_id, event.model_dump(mode="json"))
                    yield event

                tool_message = runner.get_tool_message()

            if tool_message is not None:
                context.add(tool_message)


@workflow.step(max_retries=0)
async def ship_spans(spans_data: list[dict[str, Any]]) -> None:
    # re-deliver spans collected in the workflow body to the real adapters.
    await ai.experimental_telemetry.push_all(spans_data)


@workflow.step(max_retries=0)
async def resume_turn_hook(
    token: str,
    output_data: dict[str, Any],
    turn_span_data: dict[str, Any] | None = None,
    span_attributes: dict[str, Any] | None = None,
) -> None:
    # resume() is a side effect, so it must run in a step
    output = proto.TurnOutput.model_validate(output_data)

    if turn_span_data is not None:
        turn_span = ai.experimental_telemetry.Span.model_validate(turn_span_data)
        # complete and push the turn span
        turn_span.stamp_end(
            error=ai.experimental_telemetry.SpanError(
                type="TurnError", message=output.error
            )
            if output.kind == "error" and output.error
            else None
        )
        if span_attributes:
            turn_span.set(span_attributes)
        await turn_span.push()

    # the driver may not have parked on the hook yet, so retry while it is
    # missing.
    hook = proto.TurnHook(output=output)
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if attempt == 39:
                raise
            await asyncio.sleep(0.05)


# runs one agent turn, parking on a durable hook per gated tool call
@workflow.workflow
# Draw message/part ids from the workflow's deterministic RNG so they're
# stable across replay. ``vercel.workflow.random`` is a factory resolved on
# entry (only valid inside the workflow).
@ai.messages.use_random(vercel.workflow.random)
@ai.experimental_telemetry.use_clock(vercel.workflow.time_ns)
async def run_turn(turn_input: dict[str, Any]) -> None:
    _turn_input = proto.TurnInput.model_validate(turn_input)
    messages = _turn_input.messages
    session_id = _turn_input.session_id
    turn_index = _turn_input.turn_index

    # messages should already contain either the user message
    # or the tool result message, so no need to do anything

    extra_tools = [bash, subagent] if _turn_input.gated else [bash_ungated]
    agent = DurableAgent(
        tools=extra_tools,
        session_id=session_id,
        turn_span=_turn_input.turn_span,
    )

    async def mediate(approval_event: Any, hook_id: str) -> None:
        # bridge a durable ApprovalHook back into the ai-library approval hook so
        # the gated tool proceeds in this same agent run.
        decision = await approval_event
        if decision is not None:
            ai.resolve_hook(
                hook_id,
                {
                    "granted": decision.response.granted,
                    "reason": decision.response.reason,
                },
            )

    # collect spans that happen inside the workflow body, and send them
    # once in a separate step.
    collector = (
        ai.experimental_telemetry.Collector()
        if _turn_input.turn_span is not None
        else None
    )
    try:
        model = ai.get_model(MODEL_ID)
        async with (
            ai.experimental_telemetry.use_sink(collector),
            ai.experimental_telemetry.use_span(_turn_input.turn_span),
            agent.run(model, messages) as run,
            asyncio.TaskGroup() as tg,
        ):
            async for event in run:
                if (
                    isinstance(event, ai.events.HookEvent)
                    and event.hook.status == "pending"
                    and event.hook.hook_type == ai.agents.TOOL_APPROVAL_HOOK_TYPE
                    and (tool_call_id := event.hook.tool_call_id) is not None
                ):
                    # HookEvents ride the runtime queue, not runner.events(),
                    # so the loop never wrote this; write it here so the UI
                    # gets the approval request part.
                    await write_event(session_id, event.model_dump(mode="json"))
                    tg.create_task(
                        mediate(
                            proto.ApprovalHook.wait(
                                token=proto.approval_hook_token(
                                    session_id, tool_call_id
                                )
                            ),
                            event.hook.hook_id,
                        )
                    )
                elif isinstance(event, ai.events.RunBlocked):
                    # the run is blocked on approvals; tell the client we're
                    # waiting on a human.
                    await write_event(
                        session_id,
                        stream.tool_approval_requested(turn_index=turn_index),
                    )

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
        output = proto.TurnOutput(kind="suspend", messages=messages)

    # deliver the body's collected spans. only complete records ship: a span
    # still open here would dangle in the shipping process's adapter.
    if collector is not None:
        finished = [s.model_dump(mode="json") for s in collector.finished]
        if finished:
            await ship_spans(finished)

    # notify session that the turn is complete (and export its span).
    span_attrs: dict[str, Any] | None = None
    if _turn_input.turn_span is not None:
        span_attrs = {"session.id": session_id, "turn_index": turn_index}
    await resume_turn_hook(
        _turn_input.turn_hook_token,
        output.model_dump(mode="json"),
        _turn_input.turn_span.model_dump(mode="json")
        if _turn_input.turn_span
        else None,
        span_attrs,
    )
