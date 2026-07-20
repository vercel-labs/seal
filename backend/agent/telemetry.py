"""Install the AI SDK's OpenTelemetry adapter for Seal processes.

Telemetry is opt-in through the standard OTLP endpoint environment variables.
The AI SDK owns GenAI attribute mapping and content capture; message content is
off by default and follows ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``.
"""

import os

import ai.experimental_telemetry.otel as otel_adapter
import opentelemetry.exporter.otlp.proto.http.trace_exporter as otlp
import opentelemetry.sdk.resources as resources
import opentelemetry.sdk.trace as sdk_trace
import opentelemetry.sdk.trace.export as sdk_export


def install(service: str) -> otel_adapter.OtelAdapter | None:
    """Install OTLP tracing for ``service`` when an endpoint is configured."""
    if not (
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    ):
        return None

    provider = sdk_trace.TracerProvider(
        resource=resources.Resource.create({"service.name": service})
    )
    provider.add_span_processor(sdk_export.BatchSpanProcessor(otlp.OTLPSpanExporter()))
    return otel_adapter.install(tracer_provider=provider)
