"""OTel tracing bootstrap — one call per process entrypoint.

``install`` is a no-op unless ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, so
telemetry is opt-in and costs nothing otherwise. With it set (e.g.
``http://localhost:6006`` for a local Phoenix), spans are exported over
OTLP under ``gen_ai.*`` semantic conventions.

The adapter is the SDK's stock otel adapter (``ai.experimental_telemetry
.otel.OtelAdapter``) with seal-specific attribute enrichment on top:

- the span kinds for seal's custom spans (OpenInference, so Phoenix
  classifies them), the provider name, sampling params, time-to-first-
  token as a plain number, and ``gen_ai.{input,output}.messages``
  rewritten into the OTel gen_ai semconv message shape — the SDK's
  ``_messages_json`` dumps its internal ``Message`` model verbatim
  (parts keyed by ``kind``/``text``), which viewers validating against
  the semconv schema (Phoenix) reject wholesale, leaving chat renderings
  empty. The message rewrite should be lifted upstream.

Everything durable is handled by the SDK now: spans are serializable
records, the workflow body collects its spans into a sink and ships them
through a step (``turn.run_turn``), turn spans mint as data and export
as complete records at turn completion (``turn.resume_turn_hook``), and
the stock adapter derives otel ids from span ids so the pieces line up
across processes.

Must run in each process that opens spans (server and worker).
"""

import dataclasses
import json
import os
from typing import Any

import ai.experimental_telemetry
import ai.experimental_telemetry.otel as otel_adapter
import opentelemetry.exporter.otlp.proto.http.trace_exporter as otlp
import opentelemetry.sdk.resources as resources
import opentelemetry.sdk.trace as sdk_trace
import opentelemetry.sdk.trace.export as sdk_export

def install(service: str) -> "SealOtelAdapter | None":
    """Register the otel adapter exporting to OTLP as ``service``.

    Returns the registered adapter (for ``shutdown``/``unregister``), or
    ``None`` when telemetry is disabled.  Steps that only need to know
    whether telemetry is on use ``ai.experimental_telemetry.enabled()``.
    """
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None

    provider = sdk_trace.TracerProvider(
        resource=resources.Resource.create({"service.name": service})
    )
    # OTLPSpanExporter reads OTEL_EXPORTER_OTLP_ENDPOINT itself and appends
    # the standard /v1/traces path.
    provider.add_span_processor(sdk_export.BatchSpanProcessor(otlp.OTLPSpanExporter()))
    adapter = SealOtelAdapter(tracer_provider=provider)
    ai.experimental_telemetry.register(adapter)
    return adapter


def _semconv_part(part: dict[str, Any]) -> dict[str, Any]:
    """Map one dumped ``ai`` message part onto its gen_ai semconv shape."""

    def args(raw: str) -> Any:
        try:
            return json.loads(raw)
        except ValueError:
            return raw

    match part["kind"]:
        case "text":
            return {"type": "text", "content": part["text"]}
        case "reasoning":
            return {"type": "reasoning", "content": part["text"]}
        case "tool_call":
            return {
                "type": "tool_call",
                "id": part["tool_call_id"],
                "name": part["tool_name"],
                "arguments": args(part["tool_args"]),
            }
        case "tool_result":
            return {
                "type": "tool_call_response",
                "id": part["tool_call_id"],
                "response": part["result"],
            }
        case "builtin_tool_call":
            return {
                "type": "server_tool_call",
                "id": part["tool_call_id"],
                "name": part["tool_name"],
                "server_tool_call": {
                    "type": part["tool_name"],
                    "arguments": args(part["tool_args"]),
                },
            }
        case "builtin_tool_return":
            return {
                "type": "server_tool_call_response",
                "id": part["tool_call_id"],
                "server_tool_call_response": {
                    "type": part["tool_name"],
                    "response": part["result"],
                },
            }
        case "file":
            media_type = part["media_type"]
            modality = media_type.split("/")[0]
            if modality not in ("image", "video", "audio"):
                modality = "image"
            # data is a str after the JSON dump: a URL or base-64 content.
            if part["data"].startswith(("http://", "https://")):
                return {
                    "type": "uri",
                    "modality": modality,
                    "mime_type": media_type,
                    "uri": part["data"],
                }
            return {
                "type": "blob",
                "modality": modality,
                "mime_type": media_type,
                "content": part["data"],
            }
        case kind:
            return {"type": kind}


def _semconv_messages(data: Any) -> dict[str, Any]:
    """gen_ai message attributes in semconv shape, overriding the SDK's.

    System messages are excluded from ``input.messages`` and carried as
    ``gen_ai.system_instructions`` (a flat parts list), per semconv.
    """
    if not isinstance(
        data,
        ai.experimental_telemetry.AiStreamSpanData
        | ai.experimental_telemetry.AiGenerateSpanData,
    ):
        return {}
    system_parts: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for message in data.messages:
        dumped = message.model_dump(mode="json", fallback=str)
        parts = [_semconv_part(p) for p in dumped["parts"]]
        if dumped["role"] == "system":
            system_parts += parts
        else:
            inputs.append({"role": dumped["role"], "parts": parts})
    attrs = {"gen_ai.input.messages": json.dumps(inputs)}
    if system_parts:
        attrs["gen_ai.system_instructions"] = json.dumps(system_parts)
    if data.message is not None:
        dumped = data.message.model_dump(mode="json", fallback=str)
        parts = [_semconv_part(p) for p in dumped["parts"]]
        finish = "tool_call" if any(p["type"] == "tool_call" for p in parts) else "stop"
        attrs["gen_ai.output.messages"] = json.dumps(
            [{"role": dumped["role"], "parts": parts, "finish_reason": finish}]
        )
    return attrs


# seal's own spans, classified for LLM-aware viewers (OpenInference kinds).
# Phoenix leaves spans without a kind as UNKNOWN and renders them bare.
_SPAN_KINDS = {
    "turn": "AGENT",
    "generate_title": "CHAIN",
}


def _extra_attributes(span_: ai.experimental_telemetry.Span) -> dict[str, Any]:
    """Observability attributes the SDK's stock mapping doesn't emit.

    Semconv-shaped messages, span kinds for seal's custom spans, the
    provider name, sampling params, and time-to-first-token as a plain
    number so it can be filtered and charted.
    """
    attrs = _semconv_messages(span_.data)
    if isinstance(span_.data, ai.experimental_telemetry.CustomSpanData):
        if kind := _SPAN_KINDS.get(span_.name):
            attrs["openinference.span.kind"] = kind
        return attrs

    if (model := getattr(span_.data, "model", None)) and "/" in model:
        attrs["gen_ai.provider.name"] = model.partition("/")[0]

    params = getattr(span_.data, "params", None)
    if params is not None and isinstance(
        sampling := getattr(params, "sampling", None), dict
    ):
        for sampler in sampling.values():
            for field in dataclasses.fields(sampler):
                value = getattr(sampler, field.name)
                if isinstance(value, int | float):
                    attrs[f"gen_ai.request.{field.name}"] = value

    if span_.started_at is not None:
        for event in span_.events:
            if event.name == ai.experimental_telemetry.FIRST_TOKEN:
                attrs["ai.time_to_first_token_ms"] = (
                    event.time_ns - span_.started_at
                ) / 1e6
                break
    return attrs


class SealOtelAdapter(otel_adapter.OtelAdapter):
    """The stock otel adapter with seal's attribute enrichment."""

    def span_attributes(
        self, span_: ai.experimental_telemetry.Span, /
    ) -> dict[str, Any]:
        return super().span_attributes(span_) | _extra_attributes(span_)
