import asyncio
import contextvars
import dataclasses
import traceback
from collections.abc import (
    AsyncGenerator,
    Sequence,
)
from typing import Any, ClassVar

import ai
import pydantic
import vercel.workflow

from agent import ai_util, proto, stream, util, workflow

MODEL_ID = "gateway:openai/gpt-5.6-luna"
IMAGE_MODEL_ID = "gateway:google/gemini-3.1-flash-image"
SYSTEM_PROMPT = (
    "You are Seal, a coding assistant. Use bash, web_fetch, and subagent to "
    "inspect the environment, gather information, and delegate focused work. "
    "Use generate_image to create images."
)
SUBAGENT_SYSTEM_PROMPT = (
    "You are a focused Seal subagent. Use bash, web_fetch, and generate_image "
    "when useful, then answer the delegated task directly."
)
IMAGE_SYSTEM_PROMPT = (
    "You are an image generator. Generate an image for the user's prompt."
)


class EagerToolHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    payload: ai.messages.ToolCallPart


@workflow.step
async def llm_step(
    context: ai.Context,
    writer: vercel.workflow.WorkflowWritable[proto.StreamEvent] | None,
    tool_token: str | None = None,
    turn_span: ai.experimental_telemetry.Span | None = None,
) -> ai.messages.Message:
    metadata = vercel.workflow.get_step_metadata()

    # On a retry, emit a message requesting a reload. The will trigger
    # the client to drop everything from the last step.
    if writer is not None and metadata.attempt > 1:
        await writer.write(stream.reload_requested())

    # parent this step's spans under the turn's span
    async with (
        ai.experimental_telemetry.use_span(turn_span),
        ai.stream(context=context) as model_stream,
    ):
        async for e in model_stream:
            if e.replay:
                continue

            if writer is not None:
                await writer.write(e)
            if tool_token and isinstance(e, ai.types.events.ToolEnd):
                await EagerToolHook(payload=e.tool_call).resume(tool_token)

    return model_stream.message


@workflow.step
async def write_event(
    # writes one stream event (agent or lifecycle) to the durable stream.
    # the handle passed in arrives here as a live writer.
    writer: vercel.workflow.WorkflowWritable[proto.StreamEvent],
    event: proto.StreamEvent,
) -> None:
    await writer.write(event)


# closes a durable event stream once the owning session is terminal.
@workflow.step
async def close_stream(
    writer: vercel.workflow.WorkflowWritable[proto.StreamEvent],
) -> None:
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
@workflow.step
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


@ai.tool
@workflow.step
async def generate_image(prompt: str) -> ai.messages.ContentOutput:
    """Generate an image from a text prompt. Describe the desired image in
    detail, including subject, style, and composition."""

    # the ai library has no direct image-generation API yet, so this
    # runs a model that emits images inline with its response
    # (FileParts on the message).
    model = ai.get_model(IMAGE_MODEL_ID)
    messages = [ai.system_message(IMAGE_SYSTEM_PROMPT), ai.user_message(prompt)]
    async with ai.stream(model, messages) as model_stream:
        async for _ in model_stream:
            pass
    message = model_stream.message

    if not message.images:
        return ai.content_output(
            message.text or "The image model returned no image.",
        )
    # keep any caption text the model emitted alongside its images
    return ai.content_output(
        *(
            part
            for part in message.parts
            if isinstance(part, ai.messages.TextPart | ai.messages.FilePart)
        )
    )


@workflow.step(max_retries=0)
async def spawn_subagent_turn(
    turn_input: proto.TurnInput,
    parent_span: ai.experimental_telemetry.Span | None = None,
) -> str:
    # a subagent is just one ungated turn writing to its own stream. its span
    if ai.experimental_telemetry.is_enabled():
        # create and nest the span for the subagent turn
        turn_span = ai.experimental_telemetry.create_span(
            "turn", parent=parent_span
        ).stamp_start()
        turn_span.set_attrs({"openinference.span.kind": "AGENT"})
        turn_input = turn_input.model_copy(update={"turn_span": turn_span})
    started = await vercel.workflow.start(run_turn, turn_input)
    return started.run_id


# the agent whose turn is running, set by run_turn so a tool can reach the
# turn's session/stream/span without smuggling args. tasks copy the
# contextvars at creation, so every tool task under the run sees it.
current_agent: contextvars.ContextVar["DurableAgent"] = contextvars.ContextVar(
    "current_agent"
)


@ai.tool(to_model_input=ai.agents.MessageAggregator.to_model_input)
async def subagent(prompt: str, name: str | None = None) -> ai.agents.MessageBundle:
    """Delegate a focused task to a child agent and return its answer."""
    agent = current_agent.get()
    session_id = agent.session_id
    tool_call_id = ai_util.current_tool_call_id.get()
    assert tool_call_id

    name = name or "subagent"
    child_session_id = f"{session_id}:child:{tool_call_id}"
    hook = proto.TurnHook.wait(token=proto.turn_hook_token(child_session_id))

    child_run_id = await spawn_subagent_turn(
        proto.TurnInput(
            session_id=child_session_id,
            messages=[
                ai.system_message(SUBAGENT_SYSTEM_PROMPT),
                ai.user_message(prompt),
            ],
            gated=False,
        ),
        # the child turn's root span nests under this turn's root span.
        agent.turn_span,
    )
    assert agent.writer is not None
    await write_event(
        agent.writer,
        stream.subagent_called(
            tool_call_id=tool_call_id,
            child_session_id=child_session_id,
            child_run_id=child_run_id,
            name=name,
        ),
    )
    resolution = await hook
    hook.dispose()

    output = resolution.output
    await write_event(
        agent.writer,
        stream.subagent_completed(
            tool_call_id=tool_call_id, is_error=output.kind == "error"
        ),
    )
    return ai.agents.MessageBundle(
        messages=tuple(m for m in output.messages if m.role in ("assistant", "tool"))
    )


# Tools that we can run eagerly, before the llm call generating them
# has completed. These should be non-effectful (because they might get
# cancelled) and non-streaming (because that would take some extra
# thought).
EAGER_TOOLS = {"generate_image", "web_fetch"}


class DurableAgent(ai.Agent):
    # bash is gated/ungated per mode, so it is supplied via tools=, not here.
    TOOLS: ClassVar[list[ai.AgentTool]] = [web_fetch, generate_image]

    def __init__(
        self,
        *,
        tools: Sequence[ai.AgentTool | ai.Tool] | None = None,
        session_id: str | None = None,
        writer: vercel.workflow.WorkflowWritable[proto.StreamEvent] | None = None,
        turn_span: ai.experimental_telemetry.Span | None = None,
    ) -> None:
        super().__init__(tools=tools)
        self.session_id = session_id
        self.writer = writer
        self.turn_span = turn_span

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        session_id = self.session_id

        tool_token = f"seal-early-tool:{session_id}"
        eager_tool_hook = EagerToolHook.wait(token=tool_token)

        while context.keep_running():
            async with ai_util.SpeculativeToolRunner(
                tool_stream=(
                    context.resolve(ev.payload)
                    async for ev in eager_tool_hook
                    if ev.payload.tool_name in EAGER_TOOLS
                ),
            ) as runner:
                assistant_message = await llm_step(
                    context,
                    self.writer,
                    tool_token,
                    self.turn_span,
                )
                context.add(assistant_message)
                # llm_step streamed this turn out-of-band (straight to the durable
                # stream), so yield the final StreamEnd here for run-blocked
                # tracking, which counts the turn's tool calls from it.
                yield ai.events.StreamEnd(message=assistant_message)

                tool_calls = context.resolve(assistant_message.tool_calls)
                runner.discard_except(tool_calls)
                for tool_call in tool_calls:
                    runner.schedule(tool_call)

                async for event in runner.events():
                    # write tool-running events from the producer side so they land
                    # in loop order (results before the next turn's answer); run_turn
                    # only writes HookEvents, which ride the runtime queue instead.
                    if self.writer is not None:
                        await write_event(self.writer, event)
                    yield event

                context.add(runner.get_tool_message())

        eager_tool_hook.dispose()


@workflow.step
async def ship_spans(spans: list[ai.experimental_telemetry.Span]) -> None:
    # re-deliver spans collected in the workflow body to the real adapters.
    await ai.experimental_telemetry.push_all(spans)


@workflow.step
async def resume_turn_hook(token: str, output: proto.TurnOutput) -> None:
    # resume() is a side effect, so it must run in a step. the driver may not
    # have parked on the hook yet, so retry while it is missing.
    hook = proto.TurnHook(output=output)
    async for last_attempt in util.hook_retries():
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if last_attempt:
                raise


# runs one agent turn, parking on a durable hook per gated tool call
@workflow.workflow
# Draw message/part ids from the workflow's deterministic RNG so they're
# stable across replay. ``vercel.workflow.random`` is a factory resolved on
# entry (only valid inside the workflow).
@ai.messages.use_random(vercel.workflow.random)
@ai.experimental_telemetry.use_time(vercel.workflow.time_ns)
async def run_turn(
    turn_input: proto.TurnInput,
    writer: vercel.workflow.WorkflowWritable[proto.StreamEvent] | None = None,
) -> None:
    messages = turn_input.messages
    session_id = turn_input.session_id
    turn_index = turn_input.turn_index

    # messages should already contain either the user message
    # or the tool result message, so no need to do anything

    # main turns write to the session stream (handle passed in by the driver);
    # a subagent turn owns its run's stream and must close it when done.
    owns_stream = writer is None
    if writer is None:
        writer = vercel.workflow.get_writable(type=proto.StreamEvent)

    extra_tools = [bash, subagent] if turn_input.gated else [bash_ungated]
    agent = DurableAgent(
        tools=extra_tools,
        session_id=session_id,
        writer=writer,
        turn_span=turn_input.turn_span,
    )
    # tool tasks are created under this run's context and inherit this.
    current_agent.set(agent)

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
        ai.experimental_telemetry.DictSink()
        if turn_input.turn_span is not None
        else None
    )
    try:
        model = ai.get_model(MODEL_ID)
        async with (
            ai.experimental_telemetry.use_sink(collector),
            ai.experimental_telemetry.use_span(turn_input.turn_span),
            agent.run(model, messages) as run,
            ai.util.TaskGroup() as tg,
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
                    await write_event(writer, event)
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
                        writer,
                        stream.tool_approval_requested(),
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
        # a copy: the turn span is appended below
        finished = list(collector.finished_spans)
        if turn_input.turn_span is not None:
            # complete the turn span here (pure data ops on workflow time) so
            # it ships with the rest instead of riding the resume step.
            turn_span = turn_input.turn_span.stamp_end(
                error=ai.experimental_telemetry.SpanError(
                    type="TurnError", message=output.error
                )
                if output.kind == "error" and output.error
                else None
            )
            turn_span.set_attrs({"session.id": session_id, "turn_index": turn_index})
            finished.append(turn_span)
        if finished:
            await ship_spans(finished)

    if owns_stream:
        # a subagent turn ends its own stream so readers tailing it terminate.
        await close_stream(writer)

    # notify session that the turn is complete.
    await resume_turn_hook(proto.turn_hook_token(session_id), output)
