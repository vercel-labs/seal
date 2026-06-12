import asyncio
import random
from typing import Any

import ai
import vercel.workflow

import agent.proto as proto
import agent.session as session
import agent.stream as stream
import agent.turn as turn
from agent import workflow


@workflow.step
async def write_event(
    # writes one stream event (here, lifecycle) to the durable stream
    session_id: str,
    event_data: dict[str, object],
) -> None:
    writer = await stream.get_writable(session_id)
    await writer.write(event_data)


write_event.max_retries = 0


@workflow.step
async def spawn_turn_workflow(turn_input: dict[str, object]) -> dict[str, object]:
    # fires child workflow for an agent turn
    started = await vercel.workflow.start(turn.run_turn, turn_input)
    return {"run_id": started.run_id}


spawn_turn_workflow.max_retries = 0


@workflow.step
async def spawn_task_session_workflow(
    session_input: dict[str, object],
) -> dict[str, object]:
    # fires child workflow for subagent session
    started = await vercel.workflow.start(run_session, session_input)
    return {"run_id": started.run_id}


spawn_task_session_workflow.max_retries = 0


@workflow.step
async def load_session(session_id: str) -> dict[str, Any] | None:
    # restores the latest persisted session snapshot, if any
    state = await session.read_session(session_id)
    return state.model_dump(mode="json") if state is not None else None


load_session.max_retries = 0


@workflow.step
async def save_session(state_data: dict[str, Any]) -> None:
    # appends the current session state as the latest snapshot
    await session.write_session(proto.SessionState.model_validate(state_data))


save_session.max_retries = 0


@workflow.step
async def resume_session_hook(token: str, payload_data: dict[str, Any]) -> None:
    # resume() is a side effect, so it must run in a step. the session may not
    # have parked on the hook yet, so retry while it is missing.
    payload = proto.RESUME_PAYLOAD_ADAPTER.validate_python(payload_data)
    hook = proto.SessionHook(payload=payload)
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except RuntimeError as error:
            message = str(error).lower()
            if attempt == 39 or "not found" not in message:
                raise
            await asyncio.sleep(0.05)


resume_session_hook.max_retries = 0


def _last_text(messages: list[ai.messages.Message]) -> str:
    for message in reversed(messages):
        if message.role == "assistant" and message.text:
            return message.text
    return ""


@workflow.workflow
# HACK: workflow sets up `random` as a custom seeded thing...
# We ought to make it have something explicit instead
@ai.messages.use_random_async(lambda: random)  # type: ignore
async def run_session(session_input: dict[str, Any]) -> dict[str, Any]:
    # prepare the session
    _session_input = proto.SessionInput.model_validate(session_input)
    session_id = _session_input.session_id

    restored = await load_session(session_id)
    if restored is not None:
        # resume a persisted session with the new user message appended.
        state = proto.SessionState.model_validate(restored)
        state.messages.append(ai.user_message(_session_input.prompt))
    else:
        system = (
            turn.SUBAGENT_SYSTEM_PROMPT
            if _session_input.mode == "task"
            else turn.SYSTEM_PROMPT
        )
        state = proto.SessionState(
            session_id=session_id,
            mode=_session_input.mode,
            messages=[
                ai.system_message(system),
                ai.user_message(_session_input.prompt),
            ],
        )
    await save_session(state.model_dump(mode="json"))
    await write_event(session_id, stream.session_started(mode=state.mode))

    turn_index = 0
    while True:
        # run turn workflow and suspend on a hook until it completes
        await write_event(session_id, stream.turn_started(turn_index=turn_index))
        turn_hook_token = f"seal-turn:{session_id}:{turn_index}"
        turn_hook = proto.TurnHook.wait(token=turn_hook_token)
        turn_input = proto.TurnInput(
            session_id=session_id,
            messages=state.messages,
            mode=state.mode,
            turn_hook_token=turn_hook_token,
            tool_approvals=state.tool_approvals,
        )
        await spawn_turn_workflow(turn_input.model_dump(mode="json"))
        turn_resolution = await turn_hook
        turn_hook.dispose()
        assert turn_resolution is not None
        turn_result = turn_resolution.output

        # process turn results
        state.messages = turn_result.messages
        state.tool_approvals = []  # clear because turn has consumed them
        await save_session(state.model_dump(mode="json"))
        await write_event(
            session_id,
            stream.turn_completed(turn_index=turn_index, kind=turn_result.kind),
        )

        match turn_result.kind:
            case "done":
                # we're currently in a subagent session; return output and quit.
                # we're returning all assistant and tool messages in the bundle
                output = proto.SessionOutput(
                    tool_call_id=_session_input.tool_call_id,
                    session_id=session_id,
                    output=ai.agents.MessageBundle(
                        messages=tuple(
                            msg
                            for msg in state.messages
                            if msg.role in ("assistant", "tool")
                        )
                    ),
                )

                # notify the parent session. output carries the tool_call_id
                # so parent knows which subagent this is
                if _session_input.session_hook_token is not None:
                    await resume_session_hook(
                        _session_input.session_hook_token,
                        proto.SubagentResult(output=output).model_dump(mode="json"),
                    )
                await write_event(session_id, stream.session_completed())
                await turn.close_stream(session_id)
                return output.model_dump(mode="json")

            case "suspend":
                # we are currently in the main session. wait for the next user message.
                await write_event(
                    session_id, stream.session_waiting(turn_index=turn_index)
                )
                hook = proto.SessionHook.wait(
                    token=f"seal-session:{session_id}:{turn_index}"
                )
                resolution = await hook
                hook.dispose()
                message = resolution.payload if resolution is not None else None

                if not isinstance(message, proto.NewUserMessage) or message.close:
                    await write_event(session_id, stream.session_completed())
                    await turn.close_stream(session_id)
                    return proto.SessionOutput(
                        session_id=session_id,
                        output=_last_text(state.messages),
                    ).model_dump(mode="json")

                state.messages.append(ai.user_message(message.prompt or ""))

            case "pending_requests":
                # we are currently in the main session. the turn has requested
                # out-of-session work (subagents or human-in-the-loop).
                # dispatch work, suspend on the hook and wait.
                if state.pending is None:
                    state.pending = proto.PendingState(
                        turn_index=turn_index,
                        subagents=[
                            request
                            for request in turn_result.pending_requests
                            if isinstance(request, proto.SubagentRequest)
                        ],
                        tool_approval_requests=[
                            request
                            for request in turn_result.pending_requests
                            if isinstance(request, proto.ToolApprovalRequest)
                        ],
                    )
                    await save_session(state.model_dump(mode="json"))
                pending = state.pending
                token = f"seal-session:{session_id}:{turn_index}"

                # process all requests
                if not pending.dispatched:
                    for request in pending.subagents:
                        child_input = proto.SessionInput(
                            session_id=f"{session_id}:child:{request.tool_call_id}",
                            prompt=request.prompt,
                            mode="task",
                            session_hook_token=token,
                            tool_call_id=request.tool_call_id,
                        )
                        await spawn_task_session_workflow(
                            child_input.model_dump(mode="json")
                        )
                        await write_event(
                            session_id,
                            stream.subagent_called(
                                tool_call_id=request.tool_call_id,
                                child_session_id=child_input.session_id,
                                name=request.name,
                            ),
                        )
                    if pending.tool_approval_requests:
                        await write_event(
                            session_id,
                            stream.tool_approval_requested(
                                turn_index=turn_index,
                                requests=pending.tool_approval_requests,
                            ),
                        )
                    pending.dispatched = True
                    await save_session(state.model_dump(mode="json"))

                # suspend on the same hook repeatedly until we collect results
                # from all out-of-loop work.
                hook = proto.SessionHook.wait(token=token)
                while pending.subagent_outputs.keys() != {
                    request.tool_call_id for request in pending.subagents
                } or (
                    pending.tool_approval_requests and pending.tool_approvals is None
                ):
                    resolution = await hook
                    payload = resolution.payload if resolution is not None else None

                    match payload:
                        case proto.SubagentResult(output=output):
                            pending.subagent_outputs[output.tool_call_id] = output
                            await write_event(
                                session_id,
                                stream.subagent_completed(
                                    tool_call_id=output.tool_call_id,
                                    is_error=output.is_error,
                                ),
                            )
                        case proto.ToolApprovals(tool_approvals=tool_approvals):
                            pending.tool_approvals = tool_approvals
                            await write_event(
                                session_id,
                                stream.tool_approval_resolved(
                                    turn_index=turn_index,
                                    tool_approvals=tool_approvals,
                                ),
                            )
                        case _:
                            # a close (or no payload) tears the session down even
                            # with work outstanding; the child runs are orphaned.
                            hook.dispose()
                            await write_event(session_id, stream.session_completed())
                            await turn.close_stream(session_id)
                            return proto.SessionOutput(
                                session_id=session_id,
                                output=_last_text(state.messages),
                            ).model_dump(mode="json")

                    await save_session(state.model_dump(mode="json"))
                hook.dispose()

                # the turn always leaves a trailing tool message; extend it with
                # the subagent results.
                if pending.subagents:
                    tool_message = state.messages[-1]
                    assert tool_message.role == "tool"
                    for request in pending.subagents:
                        completed = pending.subagent_outputs[request.tool_call_id]
                        # the result is the child's MessageBundle
                        tool_message.parts.append(
                            ai.tool_result_part(
                                request.tool_call_id,
                                tool_name=request.name,
                                result=completed.output,
                                is_error=completed.is_error,
                            )
                        )

                # every tool call must now have a result.
                tool_calls = {
                    part.tool_call_id
                    for message in state.messages
                    for part in message.tool_calls
                }
                tool_results = {
                    part.tool_call_id
                    for message in state.messages
                    for part in message.tool_results
                }
                assert tool_calls == tool_results, (
                    f"incomplete tool history: unsatisfied={tool_calls - tool_results}"
                )

                # store tool approvals for the next turn
                if pending.tool_approvals is not None:
                    state.tool_approvals = pending.tool_approvals

                # collection complete; clear so the next turn replays clean.
                state.pending = None

            case "error":
                await write_event(session_id, stream.session_completed(is_error=True))
                await turn.close_stream(session_id)
                return proto.SessionOutput(
                    session_id=session_id,
                    output=turn_result.error or _last_text(state.messages),
                    is_error=True,
                ).model_dump(mode="json")

        # persist post-turn mutations (resume prompt / subagent results) so the
        # next turn resumes from the latest state after a crash.
        await save_session(state.model_dump(mode="json"))
        turn_index += 1
