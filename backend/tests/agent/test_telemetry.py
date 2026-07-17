"""The seal otel adapter: stock SDK adapter + seal attribute enrichment,
spans as serializable records (mint in one step, continue in another,
export the completed record at turn completion), and the collect-then-ship
pattern for workflow-body spans.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import ai.experimental_telemetry
import ai.models.core.params as params_
import ai.types.messages as messages_
import opentelemetry.sdk.trace as sdk_trace
import opentelemetry.sdk.trace.export as sdk_export
import opentelemetry.sdk.trace.export.in_memory_span_exporter as in_memory
import pytest
from conftest import MockProvider, text_msg, tool_call_msg
from harness import (
    InProcessWorld,
    resume_approval,
    resume_session,
    start_session,
    wait_for_lifecycle,
    wait_run,
)

from agent import proto, session, telemetry


def test_install_is_noop_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert telemetry.install("seal-test") is None
    assert not ai.experimental_telemetry.enabled()


def test_install_registers_the_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:6006")
    adapter = telemetry.install("seal-test")
    assert isinstance(adapter, telemetry.SealOtelAdapter)
    try:
        # the spawn steps gate turn-span minting on this
        assert ai.experimental_telemetry.enabled()
        ai.experimental_telemetry.unregister(adapter)  # raises if not registered
        assert not ai.experimental_telemetry.enabled()
    finally:
        # stop the batch worker now: left alive, the provider's atexit hook
        # would stall interpreter exit flushing to the dead endpoint.
        adapter.shutdown()


@pytest.fixture
def exporter() -> Iterator[in_memory.InMemorySpanExporter]:
    exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    adapter = telemetry.SealOtelAdapter(tracer_provider=provider)
    ai.experimental_telemetry.register(adapter)
    yield exporter
    ai.experimental_telemetry.unregister(adapter)


async def test_spans_nest_via_ambient_context(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    async with (
        ai.experimental_telemetry.span("span_outer", source="test"),
        ai.experimental_telemetry.span("span_inner"),
    ):
        pass

    inner, outer = exporter.get_finished_spans()  # inner ends first
    assert inner.context is not None and outer.context is not None
    assert inner.parent is not None
    assert inner.parent.span_id == outer.context.span_id
    assert inner.context.trace_id == outer.context.trace_id
    assert outer.attributes is not None
    assert outer.attributes["source"] == "test"


async def test_body_spans_collect_then_ship(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    # the run_turn pattern: the workflow body re-runs on every delivery, so
    # its spans are diverted into a collector instead of live adapters...
    collector = ai.experimental_telemetry.Collector()
    with ai.experimental_telemetry.use_sink(collector):
        async with ai.experimental_telemetry.span("body_work"):
            pass
    assert exporter.get_finished_spans() == ()

    # ...and a step re-pushes them to the real adapters, exactly once.
    await ai.experimental_telemetry.push_all(
        [s.model_dump(mode="json") for s in collector.finished]
    )
    (span,) = exporter.get_finished_spans()
    assert span.name == "body_work"


async def test_turn_span_continues_across_steps_and_exports(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    # "spawn step": mint the turn span as journaled data; nothing exports.
    turn_span = ai.experimental_telemetry.create_span("turn").stamp_start()
    payload = turn_span.model_dump(mode="json")
    assert exporter.get_finished_spans() == ()

    # "llm step": children nest under the restored span.
    restored = ai.experimental_telemetry.Span.model_validate(payload)
    with ai.experimental_telemetry.use_span(restored):
        async with ai.experimental_telemetry.span("span_a"):
            pass

    # "resume step": the completed record exports with attributes and outcome.
    done = ai.experimental_telemetry.Span.model_validate(payload)
    done.stamp_end(
        error=ai.experimental_telemetry.SpanError(type="TurnError", message="boom")
    )
    done.set({"session.id": "s1"})
    await done.push()

    child, turn = exporter.get_finished_spans()  # child ended first
    assert child.context is not None and turn.context is not None
    # the child parented on ids derived from the journaled span before the
    # turn existed anywhere in otel; the record exports under those same ids.
    assert child.parent is not None
    assert child.parent.span_id == turn.context.span_id
    assert child.context.trace_id == turn.context.trace_id
    assert turn.parent is None  # a session turn roots its own trace
    assert (turn.start_time, turn.end_time) == (turn_span.started_at, done.ended_at)
    assert turn.attributes is not None
    assert turn.attributes["session.id"] == "s1"
    assert turn.attributes["openinference.span.kind"] == "AGENT"
    assert turn.status.status_code.name == "ERROR"
    assert "boom" in (turn.status.description or "")


async def test_subagent_turn_nests_under_parent_turn(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    # the spawn_subagent_turn pattern: the child turn's span is minted under
    # the calling turn's journaled span.
    parent_span = ai.experimental_telemetry.create_span("turn").stamp_start()
    parent_data = parent_span.model_dump(mode="json")

    parent = ai.experimental_telemetry.Span.model_validate(parent_data)
    child_span = ai.experimental_telemetry.create_span(
        "turn", parent=parent
    ).stamp_start()
    assert child_span.trace_id == parent_span.trace_id
    assert child_span.parent_id == parent_span.id

    await child_span.stamp_end().push()
    await parent.stamp_end().push()

    child, root = exporter.get_finished_spans()
    assert child.context is not None and root.context is not None
    assert child.parent is not None
    assert child.parent.span_id == root.context.span_id
    assert child.context.trace_id == root.context.trace_id


async def test_real_error_is_recorded(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    with pytest.raises(ValueError, match="boom"):
        async with ai.experimental_telemetry.span("span_a"):
            raise ValueError("boom")
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code.name == "ERROR"
    assert "boom" in (span.status.description or "")


async def _chat_attributes(
    exporter: in_memory.InMemorySpanExporter,
    data: ai.experimental_telemetry.SpanData,
    first_token_after_ns: int | None = None,
) -> dict[str, object]:
    async with ai.experimental_telemetry.span(data) as span_:
        if first_token_after_ns is not None:
            assert span_.started_at is not None
            span_.events.append(
                ai.experimental_telemetry.SpanEvent(
                    name=ai.experimental_telemetry.FIRST_TOKEN,
                    time_ns=span_.started_at + first_token_after_ns,
                    attributes={},
                )
            )
    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    return dict(span.attributes)


async def test_chat_messages_are_semconv_shaped(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    data = ai.experimental_telemetry.AiStreamSpanData(
        model="anthropic/claude",
        messages=[
            messages_.Message(
                role="system", parts=[messages_.TextPart(text="be brief")]
            ),
            messages_.Message(role="user", parts=[messages_.TextPart(text="hi")]),
            messages_.Message(
                role="assistant",
                parts=[
                    messages_.ToolCallPart(
                        tool_call_id="call_1",
                        tool_name="bash",
                        tool_args='{"command": "ls"}',
                    )
                ],
            ),
            messages_.Message(
                role="tool",
                parts=[
                    messages_.ToolResultPart(
                        tool_call_id="call_1", tool_name="bash", result="ok"
                    )
                ],
            ),
        ],
        message=messages_.Message(
            role="assistant", parts=[messages_.TextPart(text="done")]
        ),
    )
    attrs = await _chat_attributes(exporter, data)

    # system messages leave input.messages for system_instructions
    assert json.loads(str(attrs["gen_ai.system_instructions"])) == [
        {"type": "text", "content": "be brief"}
    ]
    inputs = json.loads(str(attrs["gen_ai.input.messages"]))
    assert [m["role"] for m in inputs] == ["user", "assistant", "tool"]
    assert inputs[0]["parts"] == [{"type": "text", "content": "hi"}]
    assert inputs[1]["parts"] == [
        {
            "type": "tool_call",
            "id": "call_1",
            "name": "bash",
            "arguments": {"command": "ls"},
        }
    ]
    assert inputs[2]["parts"] == [
        {"type": "tool_call_response", "id": "call_1", "response": "ok"}
    ]
    (output,) = json.loads(str(attrs["gen_ai.output.messages"]))
    assert output["role"] == "assistant"
    assert output["parts"] == [{"type": "text", "content": "done"}]
    assert output["finish_reason"] == "stop"


async def test_output_with_tool_calls_finishes_as_tool_call(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    data = ai.experimental_telemetry.AiStreamSpanData(
        model="anthropic/claude",
        messages=[],
        message=messages_.Message(
            role="assistant",
            parts=[
                messages_.ToolCallPart(
                    tool_call_id="call_1", tool_name="bash", tool_args="{}"
                )
            ],
        ),
    )
    attrs = await _chat_attributes(exporter, data)
    (output,) = json.loads(str(attrs["gen_ai.output.messages"]))
    assert output["finish_reason"] == "tool_call"


async def test_file_parts_map_to_uri_or_blob(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    data = ai.experimental_telemetry.AiGenerateSpanData(
        model="anthropic/claude",
        messages=[
            messages_.Message(
                role="user",
                parts=[
                    messages_.FilePart(
                        data="https://example.com/cat.png", media_type="image/png"
                    ),
                    messages_.FilePart(data=b"\x89PNG", media_type="image/png"),
                ],
            )
        ],
    )
    attrs = await _chat_attributes(exporter, data)
    (message,) = json.loads(str(attrs["gen_ai.input.messages"]))
    uri, blob = message["parts"]
    assert uri == {
        "type": "uri",
        "modality": "image",
        "mime_type": "image/png",
        "uri": "https://example.com/cat.png",
    }
    assert blob["type"] == "blob"
    assert blob["content"] == "iVBORw=="  # base-64 of the raw bytes


async def test_seal_spans_get_openinference_kinds(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    async with ai.experimental_telemetry.span("turn"):
        pass
    async with ai.experimental_telemetry.span("generate_title"):
        pass
    async with ai.experimental_telemetry.span("unclassified"):
        pass
    turn, title, unclassified = exporter.get_finished_spans()
    assert turn.attributes is not None and title.attributes is not None
    assert turn.attributes["openinference.span.kind"] == "AGENT"
    assert title.attributes["openinference.span.kind"] == "CHAIN"
    assert unclassified.attributes is not None
    assert "openinference.span.kind" not in unclassified.attributes


async def test_chat_span_gets_provider_params_and_ttft(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    data = ai.experimental_telemetry.AiStreamSpanData(
        model="anthropic/claude",
        messages=[],
        params=params_.InferenceRequestParams(
            sampling={
                params_.TemperatureSamplerParams: params_.TemperatureSamplerParams(
                    temperature=0.2
                )
            }
        ),
    )
    attrs = await _chat_attributes(exporter, data, first_token_after_ns=500)
    assert attrs["gen_ai.provider.name"] == "anthropic"
    assert attrs["gen_ai.request.temperature"] == 0.2
    assert attrs["ai.time_to_first_token_ms"] == pytest.approx(0.0005)


# --- telemetry enabled end-to-end on the real engine -------------------------
#
# the driver tests run with telemetry off (no turn span minted); these
# re-run the critical flows with the adapter installed, covering the
# mint -> sandbox-validate -> collect -> ship -> export path.


@pytest.fixture
def telemetry_on() -> Iterator[in_memory.InMemorySpanExporter]:
    # registering the adapter is all it takes: the spawn steps gate on
    # ``ai.experimental_telemetry.enabled()``.
    exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    adapter = telemetry.SealOtelAdapter(tracer_provider=provider)
    ai.experimental_telemetry.register(adapter)
    yield exporter
    ai.experimental_telemetry.unregister(adapter)


async def test_turn_with_telemetry_suspends_then_closes(
    telemetry_on: in_memory.InMemorySpanExporter,
    world: InProcessWorld,
    scripted_model: MockProvider,
) -> None:
    scripted_model.responses = [[text_msg("hello there")]]

    run = await start_session("s1", "hi")
    await wait_for_lifecycle("s1", proto.SESSION_WAITING)
    await resume_session("seal-session:s1:0", proto.NewUserMessage(close=True))
    output = proto.SessionOutput.model_validate(await wait_run(run))
    assert not output.is_error

    spans = {s.name: s for s in telemetry_on.get_finished_spans()}
    # the turn root exported at completion; the model call and the agent run
    # hang under it in one trace.
    assert "turn" in spans, f"exported: {list(spans)}"
    turn = spans["turn"]
    chat = spans["chat mock-model"]
    agent_run = spans["invoke_agent DurableAgent"]
    assert turn.context is not None
    assert chat.context is not None and agent_run.context is not None
    assert chat.context.trace_id == turn.context.trace_id
    assert agent_run.context.trace_id == turn.context.trace_id
    assert agent_run.parent is not None
    assert agent_run.parent.span_id == turn.context.span_id
    assert turn.attributes is not None
    assert turn.attributes["session.id"] == "s1"


async def test_gated_tool_approval_with_telemetry(
    telemetry_on: in_memory.InMemorySpanExporter,
    world: InProcessWorld,
    scripted_model: MockProvider,
) -> None:
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-1",
                name="bash",
                args='{"command": "echo approved-run"}',
                text="running it",
            )
        ],
        [text_msg("done")],
    ]

    await start_session("s1", "run it")
    # the approval request must still reach the stream with telemetry on.
    await wait_for_lifecycle("s1", proto.TOOL_APPROVAL_REQUESTED)

    await resume_approval(
        "s1", proto.ToolApprovalResponse(tool_call_id="tc-1", granted=True)
    )
    await wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    [tool_message] = [m for m in state.messages if m.role == "tool"]
    [result] = tool_message.tool_results
    assert result.result == "approved-run\n"

    spans = {s.name: s for s in telemetry_on.get_finished_spans()}
    assert "turn" in spans, f"exported: {list(spans)}"
    tool = spans["execute_tool bash"]
    turn = spans["turn"]
    assert tool.context is not None and turn.context is not None
    assert tool.context.trace_id == turn.context.trace_id
