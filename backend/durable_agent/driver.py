import asyncio
from typing import Any

import ai
import vercel.workflow

import durable_agent.proto as proto
import durable_agent.turn as turn

workflow = vercel.workflow.Workflows()


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


def _last_text(messages: list[ai.messages.Message]) -> str:
    for message in reversed(messages):
        if message.role == "assistant" and message.text:
            return message.text
    return ""


@workflow.workflow
async def run_session(session_input: dict[str, Any]) -> dict[str, Any]:
    _session_input = proto.SessionInput.model_validate(session_input)
    session_id = _session_input.session_id

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

    turn_index = 0
    while True:
        # park on the turn hook, then fire-and-forget the turn workflow which
        # resumes the hook with its output when the agent loop exits.
        turn_hook_token = f"seal-turn:{session_id}:{turn_index}"
        turn_hook = proto.TurnHook.wait(token=turn_hook_token)
        turn_input = proto.TurnInput(
            session_id=session_id,
            messages=state.messages,
            mode=state.mode,
            turn_hook_token=turn_hook_token,
        )
        await spawn_turn_workflow(turn_input.model_dump(mode="json"))
        turn_resolution = await turn_hook
        turn_hook.dispose()
        assert turn_resolution is not None
        turn_result = turn_resolution.output

        state.messages = turn_result.messages

        match turn_result.kind:
            case "done":
                # this is a subagent session; return output and quit
                output = proto.SessionOutput(
                    session_id=session_id,
                    output=_last_text(state.messages),
                )

                # notify parent workflow session
                if _session_input.subagent_hook_token is not None:
                    await proto.SubagentHook(output=output).resume(
                        _session_input.subagent_hook_token
                    )
                await turn.close_stream(session_id)
                return output.model_dump(mode="json")

            case "suspend":
                # this is the main session. wait for the next user message
                token = f"seal-session:{session_id}:{turn_index}"
                hook = proto.SessionResumeHook.wait(token=token)
                resolution = await hook
                hook.dispose()

                if resolution is None or resolution.close:
                    await turn.close_stream(session_id)
                    return proto.SessionOutput(
                        session_id=session_id,
                        output=_last_text(state.messages),
                    ).model_dump(mode="json")

                state.messages.append(ai.user_message(resolution.prompt or ""))

            case "subagents":
                # this is the main session, agent has requested subagents
                outputs: dict[str, proto.SessionOutput] = {}

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

                    resolution = await hook
                    hook.dispose()
                    assert resolution is not None
                    outputs[request.tool_call_id] = resolution.output

                async with asyncio.TaskGroup() as tg:
                    for index, request in enumerate(turn_result.subagent_requests):
                        tg.create_task(dispatch_subagent(index, request))

                # the turn always leaves a trailing tool message; extend it with
                # the subagent results and verify that tool calls are all satisfied.
                tool_message = state.messages[-1]
                assert tool_message.role == "tool"
                tool_message.parts.extend(
                    ai.tool_result_part(
                        request.tool_call_id,
                        tool_name=request.name,
                        result=outputs[request.tool_call_id].output,
                        is_error=outputs[request.tool_call_id].is_error,
                    )
                    for request in turn_result.subagent_requests
                )

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

        turn_index += 1


# - implement stream storage backend (jsonl, pg)
# - implement session state storage (jsonl, pg)
# - emit control events for observability
