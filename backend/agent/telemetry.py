"""Install Langfuse tracing for Seal processes through OpenTelemetry."""

import base64
import contextlib
import contextvars
import os
from collections.abc import Iterator, Mapping
from typing import Any

import ai.experimental_telemetry
import ai.experimental_telemetry.otel as otel_adapter

_export_attributes: contextvars.ContextVar[Mapping[str, str] | None] = (
    contextvars.ContextVar("langfuse_export_attributes", default=None)
)


class _LangfuseAdapter(otel_adapter.OtelAdapter):
    def span_attributes(
        self, span: ai.experimental_telemetry.Span, /
    ) -> dict[str, Any]:
        # translate vanilla otel spans to langfuse format
        attributes = super().span_attributes(span)

        match span.data:
            case (
                ai.experimental_telemetry.AiStreamSpanData()
                | ai.experimental_telemetry.AiGenerateSpanData()
            ):
                observation_type = "generation"
            case ai.experimental_telemetry.RunSpanData():
                observation_type = "agent"
            case ai.experimental_telemetry.ToolExecutionSpanData():
                observation_type = "tool"
            case ai.experimental_telemetry.LoopTurnSpanData():
                observation_type = "chain"
            case ai.experimental_telemetry.CustomSpanData() as data if (
                data.attributes.get("openinference.span.kind") == "AGENT"
            ):
                observation_type = "agent"
            case _:
                observation_type = "span"

        attributes["langfuse.observation.type"] = observation_type
        if export_attributes := _export_attributes.get():
            attributes.update(export_attributes)
        if session_id := attributes.get("session.id"):
            attributes["langfuse.session.id"] = str(session_id).partition(":child:")[
                0
            ]
        if span.name == "turn":
            attributes["langfuse.trace.name"] = "agent-turn"

        if self._capture_content:
            if input_value := attributes.get("gen_ai.input.messages"):
                attributes["langfuse.observation.input"] = input_value
            if output_value := attributes.get("gen_ai.output.messages"):
                attributes["langfuse.observation.output"] = output_value
            if input_value := attributes.get("gen_ai.tool.call.arguments"):
                attributes["langfuse.observation.input"] = input_value
            if output_value := attributes.get("gen_ai.tool.call.result"):
                attributes["langfuse.observation.output"] = output_value
        else:
            attributes.pop("langfuse.observation.input", None)
            attributes.pop("langfuse.observation.output", None)

        return attributes


def install(service: str) -> otel_adapter.OtelAdapter | None:
    """Install Langfuse tracing when its three settings are configured."""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    base_url = os.environ.get("LANGFUSE_BASE_URL")
    if not public_key or not secret_key or not base_url:
        return None

    import opentelemetry.exporter.otlp.proto.http.trace_exporter as otlp
    import opentelemetry.sdk.resources as resources
    import opentelemetry.sdk.trace as sdk_trace
    import opentelemetry.sdk.trace.export as sdk_export

    provider = sdk_trace.TracerProvider(
        resource=resources.Resource.create({"service.name": service})
    )
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    exporter = otlp.OTLPSpanExporter(
        endpoint=f"{base_url.rstrip('/')}/api/public/otel/v1/traces",
        headers={
            "Authorization": f"Basic {auth}",
            "x-langfuse-ingestion-version": "4",
        },
    )
    provider.add_span_processor(sdk_export.BatchSpanProcessor(exporter))
    adapter = _LangfuseAdapter(tracer_provider=provider, capture_content=True)
    ai.experimental_telemetry.register(adapter)
    return adapter


@contextlib.contextmanager
def use_trace(session_id: str | None) -> Iterator[None]:
    """Attach the shared Langfuse fields to spans exported in this context."""
    if session_id is None:
        yield
        return
    conversation_id = session_id.partition(":child:")[0]
    token = _export_attributes.set(
        {
            "langfuse.session.id": conversation_id,
            "langfuse.trace.name": "agent-turn",
        }
    )
    try:
        yield
    finally:
        _export_attributes.reset(token)
