"""The live otel adapter: host-side spans export as they happen, workflow-body
spans are dropped, attributes are enriched over the SDK's stock mapping, and
turn spans export retroactively under their minted (journaled) context.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

import ai.models.core.params as params_
import ai.telemetry
import ai.types.messages as messages_
import opentelemetry.sdk.trace as sdk_trace
import opentelemetry.sdk.trace.export as sdk_export
import opentelemetry.sdk.trace.export.in_memory_span_exporter as in_memory
import pytest
import vercel._internal.workflow.runtime as workflow_runtime

from agent import proto, telemetry


def test_install_is_noop_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert telemetry.install("seal-test") is None


def test_install_registers_the_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:6006")
    adapter = telemetry.install("seal-test")
    assert isinstance(adapter, telemetry.LiveOtelAdapter)
    try:
        assert telemetry._installed is adapter
        ai.telemetry.unregister(adapter)  # raises if it was not registered
    finally:
        # stop the batch worker now: left alive, the provider's atexit hook
        # would stall interpreter exit flushing to the dead endpoint.
        adapter.shutdown()
        telemetry._installed = None


@pytest.fixture
def exporter() -> Iterator[in_memory.InMemorySpanExporter]:
    exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    adapter = telemetry.LiveOtelAdapter(provider)
    ai.telemetry.register(adapter)
    yield exporter
    ai.telemetry.unregister(adapter)


async def test_spans_nest_via_ambient_context(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    async with (
        ai.telemetry.span("span_outer", source="test"),
        ai.telemetry.span("span_inner"),
    ):
        pass

    inner, outer = exporter.get_finished_spans()  # inner ends first
    assert inner.context is not None and outer.context is not None
    assert inner.parent is not None
    assert inner.parent.span_id == outer.context.span_id
    assert inner.context.trace_id == outer.context.trace_id
    assert outer.attributes is not None
    assert outer.attributes["source"] == "test"


async def test_workflow_body_spans_are_dropped(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    # the runtime sets this context while a workflow body runs; its spans
    # replay on every delivery, so the adapter must skip them wholesale.
    body_ctx = workflow_runtime.WorkflowOrchestratorContext._ctx  # type: ignore[misc]
    token = body_ctx.set(cast(Any, object()))
    try:
        async with ai.telemetry.span("turn"):
            pass
    finally:
        body_ctx.reset(token)
    assert exporter.get_finished_spans() == ()

    # the same span host-side exports fine.
    async with ai.telemetry.span("turn"):
        pass
    assert len(exporter.get_finished_spans()) == 1


async def test_parent_carries_a_minted_context(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    context = proto.TraceContext(
        trace_id="0af7651916cd43dd8448eb211c80319c",
        span_id="b7ad6b7169203331",
    )
    with telemetry.parent(context):
        async with ai.telemetry.span("span_a"):
            pass
    (span,) = exporter.get_finished_spans()
    assert span.context is not None and span.parent is not None
    assert format(span.context.trace_id, "032x") == context.trace_id
    assert format(span.parent.span_id, "016x") == context.span_id

    # without a context the block is a no-op and the span roots itself.
    exporter.clear()
    with telemetry.parent(None):
        async with ai.telemetry.span("span_b"):
            pass
    (span,) = exporter.get_finished_spans()
    assert span.parent is None


async def test_real_error_is_recorded(
    exporter: in_memory.InMemorySpanExporter,
) -> None:
    with pytest.raises(ValueError, match="boom"):
        async with ai.telemetry.span("span_a"):
            raise ValueError("boom")
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code.name == "ERROR"
    assert "boom" in (span.status.description or "")


def test_export_span_reuses_the_minted_context() -> None:
    exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(exporter))
    adapter = telemetry.LiveOtelAdapter(provider)

    context = proto.TraceContext(
        trace_id="0af7651916cd43dd8448eb211c80319c",
        span_id="b7ad6b7169203331",
        parent_span_id="00f067aa0ba902b7",
        started_at_ns=1_000,
    )
    adapter.export_span(
        "turn", context, end_ns=2_500, attributes={"session.id": "s1"}, error="boom"
    )
    (span,) = exporter.get_finished_spans()
    assert span.context is not None and span.parent is not None
    assert format(span.context.trace_id, "032x") == context.trace_id
    assert format(span.context.span_id, "016x") == context.span_id
    assert format(span.parent.span_id, "016x") == context.parent_span_id
    assert (span.start_time, span.end_time) == (1_000, 2_500)
    assert span.attributes is not None
    assert span.attributes["session.id"] == "s1"
    assert span.attributes["openinference.span.kind"] == "AGENT"
    assert span.status.status_code.name == "ERROR"

    # without a parent the span roots its own trace; without an error it
    # stays unset. a context minted without a start time is unexportable.
    exporter.clear()
    root = proto.TraceContext(
        trace_id="0af7651916cd43dd8448eb211c80319d",
        span_id="b7ad6b7169203332",
        started_at_ns=1_000,
    )
    adapter.export_span("turn", root, end_ns=2_000)
    (span,) = exporter.get_finished_spans()
    assert span.parent is None
    assert span.status.status_code.name == "UNSET"
    adapter.export_span(
        "turn", proto.TraceContext(trace_id="00" * 16, span_id="00" * 8), end_ns=2_000
    )
    assert len(exporter.get_finished_spans()) == 1


def test_export_span_is_noop_when_uninstalled() -> None:
    assert telemetry._installed is None
    telemetry.export_span(
        "turn",
        proto.TraceContext(trace_id="00" * 16, span_id="00" * 8, started_at_ns=1),
        end_ns=2,
    )


async def _chat_attributes(
    exporter: in_memory.InMemorySpanExporter,
    data: ai.telemetry.SpanData,
    first_token_after_ns: int | None = None,
) -> dict[str, object]:
    async with ai.telemetry.span(data) as span_:
        if first_token_after_ns is not None:
            span_.span_events.append(
                ai.telemetry.SpanEvent(
                    name=ai.telemetry.FIRST_TOKEN,
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
    data = ai.telemetry.AiStreamSpanData(
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
    data = ai.telemetry.AiStreamSpanData(
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
    data = ai.telemetry.AiGenerateSpanData(
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
    async with ai.telemetry.span("turn"):
        pass
    async with ai.telemetry.span("generate_title"):
        pass
    async with ai.telemetry.span("unclassified"):
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
    data = ai.telemetry.AiStreamSpanData(
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
