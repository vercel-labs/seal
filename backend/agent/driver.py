import asyncio

import ai
import vercel.workflow

import agent.proto as proto
import agent.stream as stream
import agent.turn as turn
import agent.workflow_util as workflow_util
from agent import workflow


@workflow.step(max_retries=0)
async def spawn_turn_workflow(
    turn_input: proto.TurnInput,
    writer: vercel.workflow.WorkflowWritable[proto.StreamEvent],
) -> str:
    # TODO: making retry for this safe requires cooperation on the workflow side
    # ts docs suggest using a hook and checking uniqueness!
    # fires child workflow for an agent turn
    if ai.experimental_telemetry.is_enabled():
        # mint the span for the turn and pass it in. this way
        # whatever is going on inside will be able to nest under it.
        turn_span = ai.experimental_telemetry.create_span("turn").stamp_start()
        turn_span.set_attrs({"openinference.span.kind": "AGENT"})
        turn_input = turn_input.model_copy(update={"turn_span": turn_span})
    started = await workflow_util.start(turn.run_turn, turn_input, writer)
    return started.run_id


@workflow.step
async def save_session(
    state: proto.SessionState,
    writer: vercel.workflow.WorkflowWritable[proto.SessionState],
) -> None:
    # appends the current session state as the latest snapshot
    await writer.write(state)


@workflow.step
async def acknowledge_subagent_launch(
    request: proto.LaunchSubagentHook, session_id: str, task_id: str
) -> None:
    await workflow_util.respond_to_hook(
        request,
        proto.launch_subagent_hook_token(session_id),
        task_id,
        type=str,
    )


def _last_text(messages: list[ai.messages.Message]) -> str:
    for message in reversed(messages):
        if message.role == "assistant" and message.text:
            return message.text
    return ""


class SessionWorkflow(workflow_util.WorkflowClass, registry=workflow):
    @workflow_util.init
    def __init__(self, session_input: proto.SessionInput) -> None:
        self.session_id = session_input.session_id
        self.lock = asyncio.Lock()
        self.turn_finished = asyncio.Event()
        self.turn_finished.set()
        self.state: proto.SessionState
        self.writer: vercel.workflow.WorkflowWritable[proto.StreamEvent]
        self.state_writer: vercel.workflow.WorkflowWritable[proto.SessionState]
        self.active_turn_index: int | None = None
        self.next_turn_index = 0
        self.pending_messages: list[tuple[str, bool]] = []
        self.pending_background_updates: list[str] = []

    async def _start_turn(self, *, background: bool) -> None:
        turn_index = self.next_turn_index
        self.next_turn_index += 1
        self.active_turn_index = turn_index
        self.turn_finished.clear()
        await turn.write_event(
            self.writer,
            stream.turn_started(turn_index=turn_index, background=background),
        )
        await spawn_turn_workflow(
            proto.TurnInput(
                session_id=self.session_id,
                messages=self.state.messages,
                turn_index=turn_index,
            ),
            self.writer,
        )

    async def _start_pending_turn(self) -> bool:
        running_tasks = any(
            task.status == "running" for task in self.state.background_tasks.values()
        )
        has_user_message = any(
            not background for _, background in self.pending_messages
        )
        if (
            self.active_turn_index is not None
            or not self.pending_messages
            or (running_tasks and not has_user_message)
        ):
            return False

        queued_messages = self.pending_messages.copy()
        self.pending_messages.clear()
        user_message = ai.user_message("\n\n".join(text for text, _ in queued_messages))
        self.state.messages.append(user_message)
        background = all(background for _, background in queued_messages)
        if background:
            self.state.hidden_ui_message_ids.add(user_message.id)
        else:
            self.state.active_ui_run_message_index = len(self.state.messages)
        await save_session(self.state, self.state_writer)
        await self._start_turn(background=background)
        return True

    @workflow_util.hook(
        proto.LaunchSubagentHook,
        token=lambda session: proto.launch_subagent_hook_token(session.session_id),
    )
    @ai.messages.use_random(vercel.workflow.random)
    async def launch_subagent(self, request: proto.LaunchSubagentHook) -> None:
        async with self.lock:
            if request.task_id not in self.state.background_tasks:
                child_session_id = f"{self.session_id}:child:{request.task_id}"
                child_run_id = await turn.spawn_subagent_turn(
                    proto.TurnInput(
                        session_id=child_session_id,
                        messages=[
                            ai.system_message(turn.SUBAGENT_SYSTEM_PROMPT),
                            ai.user_message(request.prompt),
                        ],
                        gated=False,
                        parent_session_id=self.session_id,
                        background_task_id=request.task_id,
                    ),
                    request.parent_span,
                )
                self.state.background_tasks[request.task_id] = (
                    proto.BackgroundTaskState(
                        task_id=request.task_id,
                        child_session_id=child_session_id,
                        child_run_id=child_run_id,
                        name=request.name,
                    )
                )
                await save_session(self.state, self.state_writer)
                await turn.write_event(
                    self.writer,
                    stream.subagent_called(
                        tool_call_id=request.task_id,
                        child_session_id=child_session_id,
                        child_run_id=child_run_id,
                        name=request.name,
                    ),
                )

            await acknowledge_subagent_launch(request, self.session_id, request.task_id)

    @workflow_util.hook(
        proto.SubagentFinishedHook,
        token=lambda session: proto.subagent_finished_hook_token(session.session_id),
    )
    @ai.messages.use_random(vercel.workflow.random)
    async def subagent_finished(self, result: proto.SubagentFinishedHook) -> None:
        # The parent tool result is written before its turn completes. Holding
        # completion until then prevents a fast child result from being
        # overwritten by the tool's earlier "running" acknowledgement.
        await self.turn_finished.wait()
        async with self.lock:
            task = self.state.background_tasks.get(result.task_id)
            if task is None or task.status != "running":
                return

            is_error = result.output.kind == "error"
            child_messages = [
                message
                for message in result.output.messages
                if message.role in ("assistant", "tool")
            ]
            task.status = "failed" if is_error else "completed"
            task.messages = child_messages
            task.error = result.output.error if is_error else None
            await save_session(self.state, self.state_writer)
            await turn.write_event(
                self.writer,
                stream.subagent_completed(
                    tool_call_id=result.task_id,
                    is_error=is_error,
                    messages=[
                        message.model_dump(mode="json") for message in child_messages
                    ],
                    error=task.error,
                ),
            )

            if is_error:
                self.pending_background_updates.append(
                    f'Background subagent "{task.name}" failed:\n\n'
                    f"{task.error or 'Unknown subagent error'}"
                )
            else:
                self.pending_background_updates.append(
                    f'Background subagent "{task.name}" finished:\n\n'
                    f"{_last_text(result.output.messages)}"
                )
            if not any(
                task.status == "running"
                for task in self.state.background_tasks.values()
            ):
                self.pending_messages.extend(
                    (update, True) for update in self.pending_background_updates
                )
                self.pending_background_updates.clear()

            await self._start_pending_turn()

    # Draw message/part ids from the workflow's deterministic RNG so they're
    # stable across replay.
    @ai.messages.use_random(vercel.workflow.random)
    @ai.experimental_telemetry.use_time(vercel.workflow.time_ns)
    async def run(self, session_input: proto.SessionInput) -> None:
        # prepare the session
        session_id = self.session_id
        # the session's event stream is this run's workflow stream; the handle is
        # writable inside steps and rides TurnInput into the turn workflows.
        self.writer = vercel.workflow.get_writable(type=proto.StreamEvent)
        # session snapshots go on a second, namespaced stream on the same run.
        self.state_writer = vercel.workflow.get_writable(
            type=proto.SessionState, namespace=stream.SESSION_NAMESPACE
        )
        # Stable hooks carry all child-turn results and user messages. The turn
        # hook also lets stream readers discover this workflow's run id.
        turn_hook = proto.TurnHook.wait(token=proto.turn_hook_token(session_id))
        session_hook = proto.SessionHook.wait(
            token=proto.session_hook_token(session_id)
        )

        initial_messages = [
            ai.system_message(turn.SYSTEM_PROMPT),
            ai.user_message(session_input.prompt),
        ]
        self.state = proto.SessionState(
            session_id=session_id,
            messages=initial_messages,
            active_ui_run_message_index=len(initial_messages),
        )
        await save_session(self.state, self.state_writer)
        await turn.write_event(self.writer, stream.session_started())

        async with self.lock:
            await self._start_turn(background=False)

        while True:
            async with self.lock:
                await self._start_pending_turn()
                background_work = any(
                    task.status == "running"
                    for task in self.state.background_tasks.values()
                )
                turn_running = self.active_turn_index is not None

            if not turn_running and not background_work:
                message = await session_hook
                self.pending_messages.append((message.payload.prompt, False))
                continue

            result = await turn_hook
            async with self.lock:
                if self.active_turn_index is None:
                    continue
                completed_turn_index = self.active_turn_index
                self.active_turn_index = None
                self.turn_finished.set()
                self.state.messages = result.output.messages

                if result.output.kind == "interrupted":
                    self.state.active_ui_run_message_index = None
                    await save_session(self.state, self.state_writer)
                    await turn.write_event(self.writer, stream.session_interrupted())
                    continue

                running_tasks = sum(
                    task.status == "running"
                    for task in self.state.background_tasks.values()
                )
                if running_tasks == 0:
                    self.state.active_ui_run_message_index = None
                await save_session(self.state, self.state_writer)
                await turn.write_event(
                    self.writer,
                    stream.session_waiting(
                        turn_index=completed_turn_index,
                        active_background_tasks=running_tasks,
                    ),
                )


run_session = SessionWorkflow.registered_workflow()
