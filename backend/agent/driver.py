from typing import Any

import ai
import vercel.workflow

import agent.proto as proto
import agent.session as session
import agent.stream as stream
import agent.turn as turn
from agent import workflow


@workflow.step(max_retries=0)
async def write_event(
    # writes one stream event (here, lifecycle) to the durable stream
    session_id: str,
    event_data: dict[str, object],
) -> None:
    writer = await stream.get_writable(session_id)
    await writer.write(event_data)


@workflow.step(max_retries=0)
async def load_session(session_id: str) -> dict[str, Any] | None:
    # restores the latest persisted session snapshot, if any
    state = await session.read_session(session_id)
    return state.model_dump(mode="json") if state is not None else None


@workflow.step(max_retries=0)
async def save_session(state_data: dict[str, Any]) -> None:
    # appends the current session state as the latest snapshot
    await session.write_session(proto.SessionState.model_validate(state_data))


def _last_text(messages: list[ai.messages.Message]) -> str:
    for message in reversed(messages):
        if message.role == "assistant" and message.text:
            return message.text
    return ""


@workflow.workflow
# Draw message/part ids from the workflow's deterministic RNG so they're
# stable across replay.
@ai.messages.use_random(vercel.workflow.random)
@ai.experimental_telemetry.use_clock(vercel.workflow.time_ns)
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
        state = proto.SessionState(
            session_id=session_id,
            messages=[
                ai.system_message(turn.SYSTEM_PROMPT),
                ai.user_message(_session_input.prompt),
            ],
        )
    await save_session(state.model_dump(mode="json"))
    await write_event(session_id, stream.session_started())

    turn_index = 0
    while True:
        # run turn workflow and suspend on a hook until it completes
        await write_event(session_id, stream.turn_started(turn_index=turn_index))
        turn_hook_token = f"seal-turn:{session_id}:{turn_index}"
        turn_hook = proto.TurnHook.wait(token=turn_hook_token)
        turn_input = proto.TurnInput(
            session_id=session_id,
            messages=state.messages,
            turn_hook_token=turn_hook_token,
        )
        await turn.spawn_turn(turn_input.model_dump(mode="json"))
        turn_resolution = await turn_hook
        turn_hook.dispose()
        assert turn_resolution is not None
        turn_result = turn_resolution.output

        # process turn results
        state.messages = turn_result.messages
        await save_session(state.model_dump(mode="json"))
        await write_event(
            session_id,
            stream.turn_completed(turn_index=turn_index, kind=turn_result.kind),
        )

        match turn_result.kind:
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
