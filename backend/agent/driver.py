import ai
import vercel.workflow

import agent.proto as proto
import agent.stream as stream
import agent.turn as turn
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
    started = await vercel.workflow.start(turn.run_turn, turn_input, writer)
    return started.run_id


@workflow.step
async def save_session(
    state: proto.SessionState,
    writer: vercel.workflow.WorkflowWritable[proto.SessionState],
) -> None:
    # appends the current session state as the latest snapshot
    await writer.write(state)


@workflow.workflow
# Draw message/part ids from the workflow's deterministic RNG so they're
# stable across replay.
@ai.messages.use_random(vercel.workflow.random)
@ai.experimental_telemetry.use_time(vercel.workflow.time_ns)
async def run_session(session_input: proto.SessionInput) -> None:
    # prepare the session
    session_id = session_input.session_id

    # the session's event stream is this run's workflow stream; the handle is
    # writable inside steps and rides TurnInput into the turn workflows.
    writer = vercel.workflow.get_writable(type=proto.StreamEvent)
    # session snapshots go on a second, namespaced stream on the same run.
    state_writer = vercel.workflow.get_writable(
        type=proto.SessionState, namespace=stream.SESSION_NAMESPACE
    )
    # Stable hooks carry all child-turn results and user messages. The turn
    # hook also lets stream readers discover this workflow's run id.
    turn_hook = proto.TurnHook.wait(token=proto.turn_hook_token(session_id))
    session_hook = proto.SessionHook.wait(token=proto.session_hook_token(session_id))

    state = proto.SessionState(
        session_id=session_id,
        messages=[
            ai.system_message(turn.SYSTEM_PROMPT),
            ai.user_message(session_input.prompt),
        ],
    )
    await save_session(state, state_writer)
    await turn.write_event(writer, stream.session_started())

    turn_index = 0
    while True:
        # run turn workflow and suspend on a hook until it completes
        await turn.write_event(writer, stream.turn_started(turn_index=turn_index))
        turn_input = proto.TurnInput(
            session_id=session_id,
            messages=state.messages,
            turn_index=turn_index,
        )
        await spawn_turn_workflow(turn_input, writer)
        turn_result = (await turn_hook).output

        # process turn results
        state.messages = turn_result.messages
        await save_session(state, state_writer)

        # A failed turn should not destroy the session. Park for another user
        # message just like a successful turn.
        await turn.write_event(writer, stream.session_waiting(turn_index=turn_index))
        resolution = await session_hook
        state.messages.append(ai.user_message(resolution.payload.prompt))

        # persist post-turn mutations (resume prompt / subagent results) so the
        # next turn resumes from the latest state after a crash.
        await save_session(state, state_writer)
        turn_index += 1
