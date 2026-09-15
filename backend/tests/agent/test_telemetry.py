"""Seal's stock OTel setup and durable collect-then-ship integration."""

from __future__ import annotations

from collections.abc import Iterator

import ai.experimental_telemetry
import ai.experimental_telemetry.otel as otel_adapter
import opentelemetry.sdk.trace as sdk_trace
import opentelemetry.sdk.trace.export as sdk_export
import opentelemetry.sdk.trace.export.in_memory_span_exporter as in_memory
import pytest

from agent import telemetry


def test_install_is_noop_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    assert telemetry.install("seal-test") is None
    assert not ai.experimental_telemetry.is_enabled()


def test_install_registers_the_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:6006")
    adapter = telemetry.install("seal-test")
    assert type(adapter) is otel_adapter.OtelAdapter
    try:
        # the spawn steps gate turn-span minting on this
        assert ai.experimental_telemetry.is_enabled()
        ai.experimental_telemetry.unregister(adapter)  # raises if not registered
        assert not ai.experimental_telemetry.is_enabled()
    finally:
        # stop the batch worker now: left alive, the provider's atexit hook
        # would stall interpreter exit flushing to the dead endpoint.
        adapter.shutdown()


def test_install_accepts_trace_specific_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "http://127.0.0.1:6006/v1/traces",
    )
    adapter = telemetry.install("seal-test")
    assert type(adapter) is otel_adapter.OtelAdapter
    assert adapter is not None
    ai.experimental_telemetry.unregister(adapter)
    adapter.shutdown()


@pytest.mark.parametrize(
    ("capture", "expected"),
    [(None, False), ("true", True)],
)
def test_install_uses_sdk_content_capture_default(
    monkeypatch: pytest.MonkeyPatch,
    capture: str | None,
    expected: bool,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:6006")
    if capture is None:
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
        )
    else:
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", capture
        )

    adapter = telemetry.install("seal-test")
    assert adapter is not None
    try:
        span = ai.experimental_telemetry.create_span(
            ai.experimental_telemetry.AiStreamSpanData(
                model="anthropic/claude",
                messages=[ai.user_message("secret")],
            )
        )
        attrs = adapter.span_attrs(span)
        assert ("gen_ai.input.messages" in attrs) is expected
    finally:
        ai.experimental_telemetry.unregister(adapter)
        adapter.shutdown()


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
        ai.experimental_telemetry.span("span_outer") as sp,
        ai.experimental_telemetry.span("span_inner"),
    ):
        sp.set_attrs(source="test")

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
    collector = ai.experimental_telemetry.DictSink()
    async with (
        ai.experimental_telemetry.use_sink(collector),
        ai.experimental_telemetry.span("body_work"),
    ):
        pass
    assert exporter.get_finished_spans() == ()

    # ...and a step re-pushes them to the real adapters, exactly once.
    await ai.experimental_telemetry.push_all(
        [s.model_dump(mode="json") for s in collector.finished_spans]
    )
    (span,) = exporter.get_finished_spans()
    assert span.name == "body_work"


async def test_turn_span_continues_across_steps_and_exports(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    # "spawn step": mint the turn span as journaled data; nothing exports.
    turn_span = ai.experimental_telemetry.create_span("turn").stamp_start()
    turn_span.set_attrs({"openinference.span.kind": "AGENT"})
    payload = turn_span.model_dump(mode="json")
    assert exporter.get_finished_spans() == ()

    # "llm step": children nest under the restored span.
    restored = ai.experimental_telemetry.Span.model_validate(payload)
    async with (
        ai.experimental_telemetry.use_span(restored),
        ai.experimental_telemetry.span("span_a"),
    ):
        pass

    # "workflow body": complete the record with outcome and attributes (pure
    # data ops)...
    done = ai.experimental_telemetry.Span.model_validate(payload)
    done.stamp_end(
        error=ai.experimental_telemetry.SpanError(type="TurnError", message="boom")
    )
    done.set_attrs({"session.id": "s1"})
    # ..."ship step": the completed record exports alongside collected spans.
    await ai.experimental_telemetry.push_all([done.model_dump(mode="json")])

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
    parent_span.set_attrs({"openinference.span.kind": "AGENT"})
    parent_data = parent_span.model_dump(mode="json")

    parent = ai.experimental_telemetry.Span.model_validate(parent_data)
    child_span = ai.experimental_telemetry.create_span(
        "turn", parent=parent
    ).stamp_start()
    child_span.set_attrs({"openinference.span.kind": "AGENT"})
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
