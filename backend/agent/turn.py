from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import datetime
import os
import signal
import traceback
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from typing import Any, cast

import temporalio.activity
import temporalio.common
import temporalio.contrib.workflow_streams
import temporalio.workflow

with temporalio.workflow.unsafe.imports_passed_through():
    import ai
    import pydantic

    from agent import TASK_QUEUE, ai_util, proto, stream

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

NO_RETRIES = temporalio.common.RetryPolicy(maximum_attempts=1)
ACTIVITY_TIMEOUT = datetime.timedelta(minutes=5)


class ModelActivityInput(pydantic.BaseModel):
    model: ai.Model
    messages: list[ai.messages.Message]
    tools: list[ai.Tool]


class PartialMessagesInput(pydantic.BaseModel):
    start_index: int
    messages: list[ai.messages.Message]


@temporalio.activity.defn
async def stream_offset_activity() -> int:
    return await (
        temporalio.contrib.workflow_streams.WorkflowStreamClient.from_within_activity()
    ).get_offset()


@temporalio.activity.defn
async def partial_messages_activity(
    activity_input: PartialMessagesInput,
) -> list[ai.messages.Message]:
    stream_client = (
        temporalio.contrib.workflow_streams.WorkflowStreamClient.from_within_activity()
    )
    end_offset = await stream_client.get_offset()
    events: list[ai.events.Event] = []
    if end_offset > activity_input.start_index:
        topic = stream_client.topic(
            stream.EVENTS_TOPIC, type=cast(Any, proto.StreamEvent)
        )
        async for item in topic.subscribe(
            from_offset=activity_input.start_index,
            poll_cooldown=datetime.timedelta(milliseconds=20),
        ):
            event = item.data
            if isinstance(event, ai.events.Event):
                events.append(event)
            if item.offset + 1 >= end_offset:
                break
    return ai_util.recover_partial_messages(activity_input.messages, events)


@temporalio.activity.defn
async def llm_activity(activity_input: ModelActivityInput) -> ai.messages.Message:
    stream_client = (
        temporalio.contrib.workflow_streams.WorkflowStreamClient.from_within_activity(
            batch_interval=datetime.timedelta(milliseconds=50),
            max_batch_size=64,
        )
    )
    events = stream_client.topic(stream.EVENTS_TOPIC, type=cast(Any, proto.StreamEvent))
    message: ai.messages.Message | None = None
    async with stream_client:
        if temporalio.activity.info().attempt > 1:
            events.publish(stream.reload_requested(), force_flush=True)

        async with ai.stream(
            activity_input.model,
            activity_input.messages,
            tools=activity_input.tools,
        ) as model_stream:
            async for event in model_stream:
                if not event.replay:
                    events.publish(event)
                if isinstance(event, ai.events.StreamEnd):
                    message = event.message

            if message is None:
                message = model_stream.message

    return message


@temporalio.activity.defn
async def bash_activity(command: str, timeout: int | None = None) -> str:
    process = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except (asyncio.CancelledError, TimeoutError) as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.communicate(), timeout=0.5)
        except TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            await process.communicate()
        if isinstance(error, asyncio.CancelledError):
            raise
        return f"Command timed out after {timeout}s."

    output = stdout.decode() if stdout else ""
    if process.returncode != 0:
        return f"[exit code {process.returncode}]\n{output}"
    return output


@temporalio.activity.defn
async def web_fetch_activity(
    url: str,
    method: str = "GET",
    headers: str = "",
    body: str = "",
) -> str:
    import httpx2

    parsed_headers: dict[str, str] = {}
    for line in headers.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            parsed_headers[key.strip()] = value.strip()

    async with httpx2.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.request(
            method,
            url,
            headers=parsed_headers or None,
            content=body or None,
        )

    return "\n".join(
        [
            f"HTTP {response.status_code}",
            *(f"{key}: {value}" for key, value in response.headers.items()),
            "",
            response.text[:50_000],
        ]
    )


@temporalio.activity.defn
async def generate_image_activity(prompt: str) -> ai.messages.ContentOutput:
    model = ai.get_model(IMAGE_MODEL_ID)
    messages = [ai.system_message(IMAGE_SYSTEM_PROMPT), ai.user_message(prompt)]
    async with ai.stream(model, messages) as model_stream:
        async for _ in model_stream:
            pass
    message = model_stream.message

    if not message.images:
        output = ai.content_output(message.text or "The image model returned no image.")
    else:
        output = ai.content_output(
            *(
                part
                for part in message.parts
                if isinstance(part, ai.messages.TextPart | ai.messages.FilePart)
            )
        )
    return output


async def _execute_activity[*Args, Result](
    function: Callable[[*Args], Awaitable[Result]],
    *args: *Args,
    retry_policy: temporalio.common.RetryPolicy | None = None,
) -> Result:
    return await temporalio.workflow.execute_activity(
        function,
        args=args,
        start_to_close_timeout=ACTIVITY_TIMEOUT,
        retry_policy=retry_policy,
    )


async def write_event(
    events: temporalio.contrib.workflow_streams.WorkflowTopicHandle[proto.StreamEvent],
    event: proto.StreamEvent,
) -> None:
    events.publish(event)
    await asyncio.sleep(0)


@ai.tool(require_approval=True)
async def bash(command: str, timeout: int | None = None) -> str:
    return await _execute_activity(
        bash_activity, command, timeout, retry_policy=NO_RETRIES
    )


bash_ungated = dataclasses.replace(
    bash, tool=bash.tool.model_copy(update={"require_approval": False})
)


@ai.tool
async def web_fetch(
    url: str,
    method: str = "GET",
    headers: str = "",
    body: str = "",
) -> str:
    return await _execute_activity(web_fetch_activity, url, method, headers, body)


@ai.tool
async def generate_image(prompt: str) -> ai.messages.ContentOutput:
    """Generate an image from a detailed text prompt."""
    return await _execute_activity(generate_image_activity, prompt)


@dataclasses.dataclass
class TurnContext:
    session_id: str
    events: temporalio.contrib.workflow_streams.WorkflowTopicHandle[proto.StreamEvent]


current_turn_context: contextvars.ContextVar[TurnContext] = contextvars.ContextVar(
    "current_turn_context"
)


@ai.tool(to_model_input=ai.agents.MessageAggregator.to_model_input)
async def subagent(prompt: str, name: str | None = None) -> ai.agents.MessageBundle:
    """Delegate a focused task to a child agent and return its answer."""
    context = current_turn_context.get()
    tool_call_id = ai_util.current_tool_call_id.get()
    if tool_call_id is None:
        raise RuntimeError("subagent called outside a tool run")

    name = name or "subagent"
    child_session_id = f"{context.session_id}:child:{tool_call_id}"
    child_workflow_id = f"{temporalio.workflow.info().workflow_id}:child:{tool_call_id}"
    await write_event(
        context.events,
        stream.subagent_called(
            tool_call_id=tool_call_id,
            child_session_id=child_session_id,
            child_run_id=child_workflow_id,
            name=name,
        ),
    )
    try:
        output = await temporalio.workflow.execute_child_workflow(
            TurnWorkflow.run,
            proto.TurnInput(
                session_id=child_session_id,
                messages=[
                    ai.system_message(SUBAGENT_SYSTEM_PROMPT),
                    ai.user_message(prompt),
                ],
                gated=False,
            ),
            id=child_workflow_id,
            task_queue=TASK_QUEUE,
        )
    finally:
        await write_event(
            context.events,
            stream.subagent_completed(
                tool_call_id=tool_call_id,
                is_error="output" not in locals() or output.kind == "error",
            ),
        )

    return ai.agents.MessageBundle(
        messages=tuple(
            message
            for message in output.messages
            if message.role in ("assistant", "tool")
        )
    )


@contextlib.asynccontextmanager
async def temporal_streamer(
    *, context: ai.Context
) -> AsyncGenerator[AsyncIterator[ai.events.AgentEvent]]:
    assistant_message = await _execute_activity(
        llm_activity,
        ModelActivityInput(
            model=context.model,
            messages=context.messages,
            tools=context.tools,
        ),
    )
    async with ai.Stream.replay_message(assistant_message) as replay:
        yield replay


async def run_turn(
    turn_input: proto.TurnInput,
    *,
    events: temporalio.contrib.workflow_streams.WorkflowTopicHandle[proto.StreamEvent],
    hook_registry: ai.HookRegistry,
) -> proto.TurnOutput:
    start_index = await _execute_activity(stream_offset_activity)
    current_turn_context.set(
        TurnContext(session_id=turn_input.session_id, events=events)
    )
    tools = [web_fetch, generate_image]
    tools += [bash, subagent] if turn_input.gated else [bash_ungated]
    agent = ai_util.DurableAgent(tools=tools, streamer=temporal_streamer)
    messages = turn_input.messages

    try:
        async with agent.run(
            ai.get_model(MODEL_ID), messages, hook_registry=hook_registry
        ) as run:
            async for event in run:
                if not isinstance(event, ai.events.ModelEvent):
                    await write_event(events, event)
            messages = run.messages
    except asyncio.CancelledError:
        messages = await _execute_activity(
            partial_messages_activity,
            PartialMessagesInput(start_index=start_index, messages=messages),
        )
        return proto.TurnOutput(
            kind="interrupted",
            messages=messages,
        )
    except Exception as error:
        print(f"[seal] error in turn:\n{traceback.format_exc()}", flush=True)
        return proto.TurnOutput(
            kind="error",
            messages=messages,
            error=f"{type(error).__name__}: {error}",
        )
    return proto.TurnOutput(kind="suspend", messages=messages)


@temporalio.workflow.defn
class TurnWorkflow:
    def __init__(self) -> None:
        self.workflow_stream = temporalio.contrib.workflow_streams.WorkflowStream()
        self.events = self.workflow_stream.topic(
            stream.EVENTS_TOPIC, type=cast(Any, proto.StreamEvent)
        )

    @temporalio.workflow.run
    @ai.messages.use_random(temporalio.workflow.random)
    async def run(self, turn_input: proto.TurnInput) -> proto.TurnOutput:
        await write_event(self.events, stream.turn_started(turn_index=0))
        output = await run_turn(
            turn_input,
            events=self.events,
            hook_registry=ai.HookRegistry(),
        )
        await write_event(
            self.events,
            stream.session_completed(is_error=output.kind == "error"),
        )
        return output


ACTIVITIES: list[Callable[..., Any]] = [
    stream_offset_activity,
    partial_messages_activity,
    llm_activity,
    bash_activity,
    web_fetch_activity,
    generate_image_activity,
]
