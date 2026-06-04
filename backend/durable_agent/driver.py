import asyncio
from typing import Any

import ai
import vercel.workflow

import durable_agent.proto as proto
import durable_agent.session as session
import durable_agent.stream as stream
import durable_agent.turn as turn
from durable_agent import workflow


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
async def resume_subagent_hook(token: str, output_data: dict[str, Any]) -> None:
    # resume() is a side effect, so it must run in a step. the parent may not
    # have parked on the hook yet, so retry while it is missing.
    hook = proto.SubagentHook(output=proto.SessionOutput.model_validate(output_data))
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except RuntimeError as error:
            message = str(error).lower()
            if attempt == 39 or "not found" not in message:
                raise
            await asyncio.sleep(0.05)


resume_subagent_hook.max_retries = 0


def _last_text(messages: list[ai.messages.Message]) -> str:
    for message in reversed(messages):
        if message.role == "assistant" and message.text:
            return message.text
    return ""


@workflow.workflow
async def run_session(session_input: dict[str, Any]) -> dict[str, Any]:
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
        # park on the turn hook, then fire-and-forget the turn workflow which
        # resumes the hook with its output when the agent loop exits.
        await write_event(session_id, stream.turn_started(turn_index=turn_index))
        turn_hook_token = f"seal-turn:{session_id}:{turn_index}"
        turn_hook = proto.TurnHook.wait(token=turn_hook_token)
        turn_input = proto.TurnInput(
            session_id=session_id,
            messages=state.messages,
            mode=state.mode,
            turn_hook_token=turn_hook_token,
            approvals=state.approvals,
        )
        await spawn_turn_workflow(turn_input.model_dump(mode="json"))
        turn_resolution = await turn_hook
        turn_hook.dispose()
        assert turn_resolution is not None
        turn_result = turn_resolution.output

        state.messages = turn_result.messages
        # decisions are consumed by the turn we just ran; clear so they are not
        # replayed again.
        state.approvals = []
        await save_session(state.model_dump(mode="json"))
        await write_event(
            session_id,
            stream.turn_completed(turn_index=turn_index, kind=turn_result.kind),
        )

        match turn_result.kind:
            case "done":
                # this is a subagent session; return output and quit
                output = proto.SessionOutput(
                    session_id=session_id,
                    output=_last_text(state.messages),
                )

                # notify parent workflow session (resume() in a step)
                if _session_input.subagent_hook_token is not None:
                    await resume_subagent_hook(
                        _session_input.subagent_hook_token,
                        output.model_dump(mode="json"),
                    )
                await write_event(session_id, stream.session_completed())
                await turn.close_stream(session_id)
                return output.model_dump(mode="json")

            case "suspend":
                # this is the main session. wait for the next user message
                await write_event(
                    session_id, stream.session_waiting(turn_index=turn_index)
                )
                token = f"seal-session:{session_id}:{turn_index}"
                hook = proto.SessionResumeHook.wait(token=token)
                resolution = await hook
                hook.dispose()

                if resolution is None or resolution.close:
                    await write_event(session_id, stream.session_completed())
                    await turn.close_stream(session_id)
                    return proto.SessionOutput(
                        session_id=session_id,
                        output=_last_text(state.messages),
                    ).model_dump(mode="json")

                state.messages.append(ai.user_message(resolution.prompt or ""))

            case "pending_requests":
                # this is the main session. the agent requested subagents
                # and/or gated tool calls awaiting a human decision. both must
                # be non-blocking for each other, so they share one task group:
                # subagents do work and await their own hooks, while approvals
                # park once on the same resume channel a user message uses
                # (matches ash, where approvals and messages share the channel).
                subagent_requests = [
                    request
                    for request in turn_result.pending_requests
                    if isinstance(request, proto.SubagentRequest)
                ]
                approval_requests = [
                    request
                    for request in turn_result.pending_requests
                    if isinstance(request, proto.ToolApprovalRequest)
                ]

                outputs: dict[str, proto.SessionOutput] = {}
                # one shared, single-shot resume hook for all approvals.
                resume_hook = (
                    proto.SessionResumeHook.wait(
                        token=f"seal-session:{session_id}:{turn_index}"
                    )
                    if approval_requests
                    else None
                )
                resume: proto.SessionResumeHook | None = None

                async def dispatch_subagent(
                    index: int,
                    request: proto.SubagentRequest,
                    turn_index: int = turn_index,
                    outputs: dict[str, proto.SessionOutput] = outputs,
                ) -> None:
                    token = f"seal-driver:{session_id}:{turn_index}:{index}"
                    hook = proto.SubagentHook.wait(token=token)

                    child_input = proto.SessionInput(
                        session_id=f"{session_id}:child:{request.tool_call_id}",
                        prompt=request.prompt,
                        mode="task",
                        subagent_hook_token=token,
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

                    resolution = await hook
                    hook.dispose()
                    assert resolution is not None
                    outputs[request.tool_call_id] = resolution.output
                    await write_event(
                        session_id,
                        stream.subagent_completed(
                            tool_call_id=request.tool_call_id,
                            is_error=resolution.output.is_error,
                        ),
                    )

                async def await_approvals(
                    hook: vercel.workflow.HookEvent[proto.SessionResumeHook],
                    requests: list[proto.ToolApprovalRequest] = approval_requests,
                    turn_index: int = turn_index,
                ) -> None:
                    # surface the gated calls, then park on the shared hook.
                    nonlocal resume
                    await write_event(
                        session_id,
                        stream.approval_requested(
                            turn_index=turn_index, requests=requests
                        ),
                    )
                    resume = await hook

                async with asyncio.TaskGroup() as tg:
                    for index, request in enumerate(subagent_requests):
                        tg.create_task(dispatch_subagent(index, request))
                    if resume_hook is not None:
                        tg.create_task(await_approvals(resume_hook))

                if resume_hook is not None:
                    resume_hook.dispose()

                # the turn always leaves a trailing tool message; extend it with
                # the subagent results (gated calls keep their is_hook_pending
                # placeholder result, replaced when the next turn replays them).
                if subagent_requests:
                    tool_message = state.messages[-1]
                    assert tool_message.role == "tool"
                    tool_message.parts.extend(
                        ai.tool_result_part(
                            request.tool_call_id,
                            tool_name=request.name,
                            result=outputs[request.tool_call_id].output,
                            is_error=outputs[request.tool_call_id].is_error,
                        )
                        for request in subagent_requests
                    )

                # every tool call must now be balanced by a result.
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

                # apply the human decision (turn-level, after the group joins).
                if approval_requests:
                    if resume is None or resume.close:
                        await write_event(session_id, stream.session_completed())
                        await turn.close_stream(session_id)
                        return proto.SessionOutput(
                            session_id=session_id,
                            output=_last_text(state.messages),
                        ).model_dump(mode="json")

                    state.approvals = resume.approvals
                    await write_event(
                        session_id,
                        stream.approval_resolved(
                            turn_index=turn_index, decisions=resume.approvals
                        ),
                    )

        # persist post-turn mutations (resume prompt / subagent results) so the
        # next turn resumes from the latest state after a crash.
        await save_session(state.model_dump(mode="json"))
        turn_index += 1
