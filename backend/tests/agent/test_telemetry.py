"""Seal's stock OTel setup and durable collect-then-ship integration."""

from __future__ import annotations

import unittest.mock
from collections.abc import Iterator

import ai.experimental_telemetry
import ai.experimental_telemetry.otel as otel_adapter
import opentelemetry.exporter.otlp.proto.http.trace_exporter as otlp
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


def test_install_is_noop_without_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    assert telemetry.install("seal-test") is None
    assert not ai.experimental_telemetry.enabled()


def test_install_configures_langfuse_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com/")
    exporter_factory = unittest.mock.Mock(return_value=in_memory.InMemorySpanExporter())
    monkeypatch.setattr(otlp, "OTLPSpanExporter", exporter_factory)

    adapter = telemetry.install("seal-test")
    assert isinstance(adapter, otel_adapter.OtelAdapter)
    try:
        exporter_factory.assert_called_once_with(
            endpoint="https://us.cloud.langfuse.com/api/public/otel/v1/traces",
            headers={
                "Authorization": "Basic cGstdGVzdDpzay10ZXN0",
                "x-langfuse-ingestion-version": "4",
            },
        )
        # the spawn steps gate turn-span minting on this
        assert ai.experimental_telemetry.enabled()
        ai.experimental_telemetry.unregister(adapter)  # raises if not registered
        assert not ai.experimental_telemetry.enabled()
    finally:
        adapter.shutdown()


def test_install_captures_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:6006")

    adapter = telemetry.install("seal-test")
    assert adapter is not None
    try:
        span = ai.experimental_telemetry.create_span(
            ai.experimental_telemetry.AiStreamSpanData(
                model="anthropic/claude",
                messages=[ai.user_message("secret")],
            )
        )
        attrs = adapter.span_attributes(span)
        assert "gen_ai.input.messages" in attrs
        assert "langfuse.observation.input" in attrs
        assert attrs["langfuse.observation.type"] == "generation"
    finally:
        ai.experimental_telemetry.unregister(adapter)
        adapter.shutdown()


def test_adapter_maps_observation_types() -> None:
    provider = sdk_trace.TracerProvider()
    adapter = telemetry._LangfuseAdapter(
        tracer_provider=provider, capture_content=False
    )
    cases = [
        (
            ai.experimental_telemetry.RunSpanData(
                agent="Seal", model="anthropic/claude", messages=[]
            ),
            "agent",
        ),
        (
            ai.experimental_telemetry.ToolExecutionSpanData(
                tool_name="bash", tool_call_id="tc-1"
            ),
            "tool",
        ),
        (ai.experimental_telemetry.LoopTurnSpanData(), "chain"),
        (
            ai.experimental_telemetry.CustomSpanData(
                attributes={"openinference.span.kind": "AGENT"}
            ),
            "agent",
        ),
    ]
    try:
        for index, (data, expected) in enumerate(cases):
            span = ai.experimental_telemetry.Span(
                name="test",
                data=data,
                id=f"span-{index}",
                trace_id="trace-1",
            )
            assert adapter.span_attributes(span)["langfuse.observation.type"] == (
                expected
            )
    finally:
        adapter.shutdown()


def test_adapter_maps_turn_trace_attributes() -> None:
    adapter = telemetry._LangfuseAdapter(
        tracer_provider=sdk_trace.TracerProvider(), capture_content=False
    )
    span = ai.experimental_telemetry.Span(
        name="turn",
        data=ai.experimental_telemetry.CustomSpanData(
            attributes={
                "openinference.span.kind": "AGENT",
                "session.id": "session-1:child:tool-call-1",
            }
        ),
        id="span-1",
        trace_id="trace-1",
    )
    try:
        attributes = adapter.span_attributes(span)
        assert attributes["langfuse.session.id"] == "session-1"
        assert attributes["langfuse.trace.name"] == "agent-turn"
    finally:
        adapter.shutdown()


async def test_export_context_uses_conversation_session_for_subagents() -> None:
    exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    adapter = telemetry._LangfuseAdapter(
        tracer_provider=provider, capture_content=False
    )
    ai.experimental_telemetry.register(adapter)
    try:
        span = (
            ai.experimental_telemetry.create_span("child-work")
            .stamp_start()
            .stamp_end()
        )
        with telemetry.use_trace("session-1:child:tool-call-1"):
            await ai.experimental_telemetry.push_all([span])
        (exported,) = exporter.get_finished_spans()
        assert exported.attributes is not None
        assert exported.attributes["langfuse.session.id"] == "session-1"
    finally:
        ai.experimental_telemetry.unregister(adapter)


async def test_export_context_accepts_none() -> None:
    exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    adapter = telemetry._LangfuseAdapter(
        tracer_provider=provider, capture_content=False
    )
    ai.experimental_telemetry.register(adapter)
    try:
        span = ai.experimental_telemetry.create_span("work").stamp_start().stamp_end()
        with telemetry.use_trace(None):
            await ai.experimental_telemetry.push_all([span])
        (exported,) = exporter.get_finished_spans()
        assert exported.attributes is not None
        assert "langfuse.session.id" not in exported.attributes
        assert "langfuse.trace.name" not in exported.attributes
    finally:
        ai.experimental_telemetry.unregister(adapter)


@pytest.fixture
def exporter() -> Iterator[in_memory.InMemorySpanExporter]:
    exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    adapter = otel_adapter.OtelAdapter(tracer_provider=provider, capture_content=False)
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
    turn_span = ai.experimental_telemetry.create_span(
        "turn", {"openinference.span.kind": "AGENT"}
    ).stamp_start()
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
    parent_span = ai.experimental_telemetry.create_span(
        "turn", {"openinference.span.kind": "AGENT"}
    ).stamp_start()
    parent_data = parent_span.model_dump(mode="json")

    parent = ai.experimental_telemetry.Span.model_validate(parent_data)
    child_span = ai.experimental_telemetry.create_span(
        "turn",
        {"openinference.span.kind": "AGENT"},
        parent=parent,
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
    adapter = telemetry._LangfuseAdapter(
        tracer_provider=provider, capture_content=False
    )
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
    assert turn.attributes["langfuse.session.id"] == "s1"
    assert turn.attributes["langfuse.trace.name"] == "agent-turn"
    assert turn.attributes["langfuse.observation.type"] == "agent"
    assert turn.attributes["openinference.span.kind"] == "AGENT"
    assert chat.attributes is not None
    assert chat.attributes["langfuse.session.id"] == "s1"
    assert chat.attributes["langfuse.trace.name"] == "agent-turn"
    assert chat.attributes["langfuse.observation.type"] == "generation"
    assert agent_run.attributes is not None
    assert agent_run.attributes["langfuse.session.id"] == "s1"
    assert agent_run.attributes["langfuse.observation.type"] == "agent"
    assert "input.value" not in turn.attributes
    assert "output.value" not in turn.attributes
    assert "langfuse.observation.input" not in turn.attributes
    assert "langfuse.observation.output" not in turn.attributes


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
