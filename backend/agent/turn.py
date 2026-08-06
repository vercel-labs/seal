import asyncio
import contextvars
import dataclasses
import traceback
from collections.abc import AsyncGenerator, Sequence
from typing import Any, ClassVar

import ai
import pydantic
import vercel.workflow

from agent import proto, stream, util, workflow

MODEL_ID = "gateway:anthropic/claude-sonnet-4.6"
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
    model_id: str,
    messages_data: list[dict[str, object]],
    tools_data: list[dict[str, object]],
    session_id: str | None,
    tool_token: str | None = None,
    turn_span_data: dict[str, object] | None = None,
) -> dict[str, object]:
    model = ai.get_model(model_id)
    messages = [
        ai.messages.Message.model_validate(message) for message in messages_data
    ]
    tools = [ai.Tool.model_validate(tool) for tool in tools_data]

    writer = await stream.get_writable(session_id) if session_id else None
    metadata = vercel.workflow.get_step_metadata()

    # On a retry, emit a message requesting a reload. The will trigger
    # the client to drop everything from the last step.
    if writer is not None and metadata.attempt > 1:
        await writer.write(stream.reload_requested())

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
            if e.replay:
                continue

            if writer is not None:
                await writer.write(e)
            if tool_token and isinstance(e, ai.types.events.ToolEnd):
                await EagerToolHook(payload=e.tool_call).resume(tool_token)

    return model_stream.message.model_dump(mode="json")


@workflow.step
async def write_event(
    # writes one stream event (agent or lifecycle) to the durable stream
    session_id: str,
    event_data: dict[str, object],
) -> None:
    writer = await stream.get_writable(session_id)
    await writer.write(event_data)


# closes a durable event stream once the owning session is terminal.
@workflow.step
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


@workflow.step
async def image_step(prompt: str) -> dict[str, object]:
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
        ).model_dump(mode="json")
    # keep any caption text the model emitted alongside its images
    return ai.content_output(
        *(
            part
            for part in message.parts
            if isinstance(part, ai.messages.TextPart | ai.messages.FilePart)
        )
    ).model_dump(mode="json")


@ai.tool
async def generate_image(prompt: str) -> ai.messages.ContentOutput:
    # TODO: annoyingly we have to have a model_validate in a tool outside the step
    return ai.messages.ContentOutput.model_validate(await image_step(prompt))


@workflow.step(max_retries=0)
async def spawn_subagent_turn(
    turn_input: dict[str, object],
    parent_span_data: dict[str, object] | None = None,
) -> dict[str, object]:
    # a subagent is just one ungated turn writing to its own stream. its span
    payload = dict(turn_input)
    if ai.experimental_telemetry.is_enabled():
        # create and nest the span for the subagent turn
        parent = (
            ai.experimental_telemetry.Span.model_validate(parent_span_data)
            if parent_span_data
            else None
        )
        turn_span = ai.experimental_telemetry.create_span(
            "turn", parent=parent
        ).stamp_start()
        turn_span.set_attrs({"openinference.span.kind": "AGENT"})
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


# Tools that we can run eagerly, before the llm call generating them
# has completed. These should be non-effectful (because they might get
# cancelled) and non-streaming (because that would take some extra
# thought).
EAGER_TOOLS = {"generate_image", "web_fetch"}


class DurableAgent(ai.Agent):
    # bash is gated/ungated per mode, so it is supplied via tools=, not here.
    TOOLS: ClassVar[list[ai.AgentTool]] = [web_fetch, generate_image]

    tg: asyncio.TaskGroup

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

        tool_token = f"seal-early-tool:{session_id}"
        live_tool_calls = {}

        def launch_tool(tool_call: ai.messages.ToolCallPart) -> None:
            # Launch a tool in a task under the right context, track
            # it in the live call table.
            token = tool_call_context.set(
                proto.ToolCallContext(
                    session_id=session_id or "",
                    tool_call_id=tool_call.tool_call_id,
                    turn_span=self.turn_span,
                )
            )
            live_tool_calls[tool_call.tool_call_id] = self.tg.create_task(
                context.resolve(tool_call)()
            )
            tool_call_context.reset(token)

        eager_tool_hook = EagerToolHook.wait(token=tool_token)

        async def watcher() -> None:
            # Wait on our eager tool hook. For EAGER_TOOLS, trigger
            # them now, from the watcher thread.
            #
            # Once llm_step returns, the tool runner will schedule a
            # ToolRunner task that waits on them.
            async for ev in eager_tool_hook:
                tool_call = ev.payload
                if tool_call.tool_name in EAGER_TOOLS:
                    launch_tool(tool_call)

        watcher_task = self.tg.create_task(watcher())

        while context.keep_running():
            live_tool_calls.clear()

            result = await llm_step(
                model_id,
                [message.model_dump(mode="json") for message in context.messages],
                [tool.model_dump(mode="json") for tool in context.tools],
                session_id,
                tool_token,
                turn_span_data,
            )

            assistant_message = ai.messages.Message.model_validate(result)
            context.add(assistant_message)
            # llm_step streamed this turn out-of-band (straight to the durable
            # stream), so yield the final StreamEnd here for run-blocked
            # tracking, which counts the turn's tool calls from it.
            yield ai.events.StreamEnd(message=assistant_message)

            async with ai.ToolRunner() as runner:
                # Cancel eager tool calls that are not legit -- that
                # is, ones that are from a retried llm call. They
                # won't actually get stopped if they are steps, unless
                # the cancellation happens before the step was
                # launched, but it will stop us from waiting on them.
                legit_call_ids = {
                    tc.tool_call_id for tc in assistant_message.tool_calls
                }
                for id, task in list(live_tool_calls.items()):
                    if id not in legit_call_ids:
                        task.cancel()
                        del live_tool_calls[id]

                for tool_call in assistant_message.tool_calls:
                    # Launch the tool if it isn't running already
                    if tool_call.tool_call_id not in live_tool_calls:
                        launch_tool(tool_call)

                    # Wait on it
                    async def _wait(tc: ai.messages.ToolCallPart = tool_call) -> Any:
                        return await live_tool_calls[tc.tool_call_id]

                    runner.schedule(_wait)

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

        watcher_task.cancel()
        eager_tool_hook.dispose()


@workflow.step
async def ship_spans(spans_data: list[dict[str, Any]]) -> None:
    # re-deliver spans collected in the workflow body to the real adapters.
    await ai.experimental_telemetry.push_all(spans_data)


@workflow.step
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


# runs one agent turn, parking on a durable hook per gated tool call
@workflow.workflow
# Draw message/part ids from the workflow's deterministic RNG so they're
# stable across replay. ``vercel.workflow.random`` is a factory resolved on
# entry (only valid inside the workflow).
@ai.messages.use_random(vercel.workflow.random)
@ai.experimental_telemetry.use_time(vercel.workflow.time_ns)
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
        ai.experimental_telemetry.DictSink()
        if _turn_input.turn_span is not None
        else None
    )
    try:
        model = ai.get_model(MODEL_ID)
        async with (
            ai.experimental_telemetry.use_sink(collector),
            ai.experimental_telemetry.use_span(_turn_input.turn_span),
            agent.run(model, messages) as run,
            ai.util.TaskGroup() as tg,
        ):
            agent.tg = tg

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
        finished = [s.model_dump(mode="json") for s in collector.finished_spans]
        if _turn_input.turn_span is not None:
            # complete the turn span here (pure data ops on workflow time) so
            # it ships with the rest instead of riding the resume step.
            turn_span = _turn_input.turn_span.stamp_end(
                error=ai.experimental_telemetry.SpanError(
                    type="TurnError", message=output.error
                )
                if output.kind == "error" and output.error
                else None
            )
            turn_span.set_attrs({"session.id": session_id, "turn_index": turn_index})
            finished.append(turn_span.model_dump(mode="json"))
        if finished:
            await ship_spans(finished)

    # notify session that the turn is complete.
    await resume_turn_hook(_turn_input.turn_hook_token, output.model_dump(mode="json"))
