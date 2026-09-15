from __future__ import annotations

import asyncio
from typing import Any, cast

import temporalio.contrib.workflow_streams
import temporalio.workflow

with temporalio.workflow.unsafe.imports_passed_through():
    import ai

    from agent import proto, stream, turn


@temporalio.workflow.defn
class SessionWorkflow:
    def __init__(self) -> None:
        self.workflow_stream = temporalio.contrib.workflow_streams.WorkflowStream()
        self.events = self.workflow_stream.topic(
            stream.EVENTS_TOPIC, type=cast(Any, proto.StreamEvent)
        )
        self.hook_registry = ai.HookRegistry()
        self.eager_tool_events = ai.util.AsyncIterableQueue[ai.events.ToolEnd]()
        self.pending_messages: list[proto.NewUserMessage] = []
        self.run_task: asyncio.Task[proto.TurnOutput] | None = None
        self.state: proto.SessionState | None = None

    @temporalio.workflow.query
    def get_state(self) -> proto.SessionState | None:
        return self.state

    @temporalio.workflow.signal
    def new_message(self, message: proto.NewUserMessage) -> None:
        self.pending_messages.append(message)

    @temporalio.workflow.signal
    def approvals(self, signal: proto.ApprovalSignal) -> None:
        for response in signal.responses:
            ai.resolve_hook(
                response.hook_id,
                {"granted": response.granted, "reason": response.reason},
                registry=self.hook_registry,
            )

    @temporalio.workflow.signal
    def interrupt(self) -> None:
        if self.run_task is not None:
            self.run_task.cancel()

    @temporalio.workflow.signal(name=turn.EAGER_TOOL_SIGNAL)
    def eager_tool(self, signal: proto.EagerToolSignal) -> None:
        self.eager_tool_events.put_nowait(
            ai.events.ToolEnd(
                tool_call_id=signal.tool_call.tool_call_id,
                tool_call=signal.tool_call,
            )
        )

    @temporalio.workflow.run
    @ai.messages.use_random(temporalio.workflow.random)
    async def run(self, session_input: proto.SessionInput) -> None:
        self.state = proto.SessionState(
            session_id=session_input.session_id,
            messages=[
                ai.system_message(turn.SYSTEM_PROMPT),
                ai.user_message(session_input.prompt),
            ],
        )
        await turn.write_event(self.events, stream.session_started())

        turn_index = 0
        while True:
            self.eager_tool_events = ai.util.AsyncIterableQueue()
            await turn.write_event(
                self.events, stream.turn_started(turn_index=turn_index)
            )
            self.run_task = asyncio.create_task(
                turn.run_turn(
                    proto.TurnInput(
                        session_id=session_input.session_id,
                        messages=self.state.messages,
                        turn_index=turn_index,
                    ),
                    events=self.events,
                    eager_tool_events=self.eager_tool_events,
                    hook_registry=self.hook_registry,
                )
            )
            output = await self.run_task
            self.run_task = None
            self.state.messages = output.messages

            if output.kind == "interrupted":
                await turn.write_event(self.events, stream.session_interrupted())
            else:
                await turn.write_event(
                    self.events, stream.session_waiting(turn_index=turn_index)
                )

            await temporalio.workflow.wait_condition(
                lambda: bool(self.pending_messages)
            )
            next_message = self.pending_messages.pop(0)
            self.state.messages.append(ai.user_message(next_message.prompt))
            turn_index += 1
