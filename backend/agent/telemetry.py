"""Install Langfuse tracing for Seal processes through OpenTelemetry."""

import base64
import os
from typing import Any

import ai.experimental_telemetry
import ai.experimental_telemetry.otel as otel_adapter


def trace_attrs(session_id: str) -> dict[str, str]:
    """Langfuse trace fields for a turn's root span; children inherit them."""
    return {
        # subagent sessions report under their conversation's session
        "langfuse.session.id": session_id.partition(":child:")[0],
        "langfuse.trace.name": "agent-turn",
    }


class _LangfuseAdapter(otel_adapter.OtelAdapter):
    def span_attrs(self, span: ai.experimental_telemetry.Span, /) -> dict[str, Any]:
        # translate vanilla otel spans to langfuse format
        attributes = super().span_attrs(span)

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
                data.attrs.get("openinference.span.kind") == "AGENT"
            ):
                observation_type = "agent"
            case _:
                observation_type = "span"

        attributes["langfuse.observation.type"] = observation_type

        if self._is_capturing_content:
            if input_value := attributes.get("gen_ai.input.messages"):
                attributes["langfuse.observation.input"] = input_value
            if output_value := attributes.get("gen_ai.output.messages"):
                attributes["langfuse.observation.output"] = output_value
            if input_value := attributes.get("gen_ai.tool.call.arguments"):
                attributes["langfuse.observation.input"] = input_value
            if output_value := attributes.get("gen_ai.tool.call.result"):
                attributes["langfuse.observation.output"] = output_value

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
