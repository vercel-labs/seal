"""OTel tracing bootstrap — one call per process entrypoint.

``install`` is a no-op unless ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, so
telemetry is opt-in and costs nothing otherwise. With it set (e.g.
``http://localhost:6006`` for a local Phoenix), spans are exported over
OTLP under ``gen_ai.*`` semantic conventions.

The adapter here follows the SDK's stock otel lifecycle
(``ai.telemetry.otel``): each ai span maps onto a live otel span, opened
at span start on the ambient otel context and ended at span end. Two
seal-specific behaviors on top:

- spans opened inside a durable workflow body are dropped. The body is
  journal-replayed code: every delivery re-runs it (duplicate spans) and
  suspension unwinds it mid-flight (phantom errors), so its spans are
  noise, never live work. Real work happens host-side — in workflow
  steps and the server — and only those spans export. Trace continuity
  across the body is carried as plain journaled data instead
  (``proto.TraceContext``, minted by the spawn steps); steps re-enter
  the trace via ``parent``, and the turn span exports retroactively
  through ``export_span`` at turn completion, under its minted ids and
  with its true duration.
- attributes are enriched over the stock mapping: the span kinds for
  seal's custom spans (OpenInference, so Phoenix classifies them), the
  provider name, sampling params, time-to-first-token as a plain
  number, and ``gen_ai.{input,output}.messages`` rewritten into the
  OTel gen_ai semconv message shape — the SDK's ``_messages_json``
  dumps its internal ``Message`` model verbatim (parts keyed by
  ``kind``/``text``), which viewers validating against the semconv
  schema (Phoenix) reject wholesale, leaving chat renderings empty.
  The message rewrite belongs in ``ai.telemetry.otel`` and should be
  lifted upstream.

Must run in each process that opens spans (server and worker).
"""

import contextlib
import contextvars
import dataclasses
import json
import os
from collections.abc import AsyncGenerator, Iterator
from typing import Any

import ai.telemetry
import ai.telemetry.otel as otel_adapter
import opentelemetry.context as otel_context
import opentelemetry.exporter.otlp.proto.http.trace_exporter as otlp
import opentelemetry.sdk.resources as resources
import opentelemetry.sdk.trace as sdk_trace
import opentelemetry.sdk.trace.export as sdk_export
import opentelemetry.sdk.trace.id_generator as id_generator
import opentelemetry.trace as otel_trace
import vercel._internal.workflow.runtime as workflow_runtime

from agent import proto


def _in_workflow_body() -> bool:
    # set by the runtime while a workflow body executes (its journal
    # replay), never in step handlers or the server. private API — the
    # runtime has no public probe for this yet.
    body_ctx = workflow_runtime.WorkflowOrchestratorContext._ctx  # type: ignore[misc]
    return body_ctx.get(None) is not None


def _plain(value: Any) -> str | bool | int | float:
    return value if isinstance(value, str | bool | int | float) else repr(value)


_installed: "LiveOtelAdapter | None" = None


def install(service: str) -> "LiveOtelAdapter | None":
    """Register the live otel adapter exporting to OTLP as ``service``.

    Returns the registered adapter (for ``shutdown``/``unregister``), or
    ``None`` when telemetry is disabled.
    """
    global _installed
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None

    provider = sdk_trace.TracerProvider(
        resource=resources.Resource.create({"service.name": service})
    )
    # OTLPSpanExporter reads OTEL_EXPORTER_OTLP_ENDPOINT itself and appends
    # the standard /v1/traces path.
    provider.add_span_processor(sdk_export.BatchSpanProcessor(otlp.OTLPSpanExporter()))
    adapter = LiveOtelAdapter(provider)
    ai.telemetry.register(adapter)
    _installed = adapter
    return adapter


def _span_context(context: proto.TraceContext, span_id: str) -> otel_trace.SpanContext:
    # rebuilds an otel SpanContext from a journaled ``proto.TraceContext``,
    # under the given span id (its own, or its parent's).
    return otel_trace.SpanContext(
        trace_id=int(context.trace_id, 16),
        span_id=int(span_id, 16),
        is_remote=True,
        trace_flags=otel_trace.TraceFlags(context.trace_flags),
    )


@contextlib.contextmanager
def parent(context: proto.TraceContext | None) -> Iterator[None]:
    """Set a journaled span context as the ambient otel parent for the block.

    Live spans opened inside continue that trace: this is how a trace
    crosses the (span-free) workflow body into host-side steps. No-op
    without a context.
    """
    if context is None:
        yield
        return
    token = otel_context.attach(
        otel_trace.set_span_in_context(
            otel_trace.NonRecordingSpan(_span_context(context, context.span_id))
        )
    )
    try:
        yield
    finally:
        otel_context.detach(token)


def export_span(
    name: str,
    context: proto.TraceContext,
    end_ns: int,
    attributes: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """Retroactively export a span under its minted (journaled) context.

    No-op when telemetry is off. This is how a span whose lifetime brackets
    a durable workflow exports: its identity is minted up front, children
    parent under it while it is still just data, and the span itself appears
    only here — at completion, with its true times and outcome.
    """
    if _installed is not None:
        _installed.export_span(name, context, end_ns, attributes, error=error)


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
        data, ai.telemetry.AiStreamSpanData | ai.telemetry.AiGenerateSpanData
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


def _extra_attributes(span_: ai.telemetry.Span) -> dict[str, Any]:
    """Observability attributes the SDK's ``_attributes`` doesn't emit.

    Semconv-shaped messages, span kinds for seal's custom spans, the
    provider name, sampling params, and time-to-first-token as a plain
    number so it can be filtered and charted.
    """
    attrs = _semconv_messages(span_.data)
    if isinstance(span_.data, ai.telemetry.CustomSpanData):
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

    for event in span_.span_events:
        if event.name == ai.telemetry.FIRST_TOKEN:
            attrs["ai.time_to_first_token_ms"] = (
                event.time_ns - span_.started_at
            ) / 1e6
            break
    return attrs


# ``Tracer.start_span`` offers no way to dictate a span's ids, but a
# retroactively exported span must come out under the exact ids its children
# already parented to; the provider's generator honors a preset while set.
_preset_ids: contextvars.ContextVar[tuple[int, int] | None] = contextvars.ContextVar(
    "preset_ids", default=None
)


class _PresetIdGenerator(id_generator.RandomIdGenerator):
    def generate_trace_id(self) -> int:
        preset = _preset_ids.get()
        return preset[0] if preset is not None else super().generate_trace_id()

    def generate_span_id(self) -> int:
        preset = _preset_ids.get()
        return preset[1] if preset is not None else super().generate_span_id()


class LiveOtelAdapter(ai.telemetry.Adapter):
    """Maps ai spans onto live otel spans; workflow-body spans are dropped."""

    def __init__(self, tracer_provider: sdk_trace.TracerProvider) -> None:
        self._provider = tracer_provider
        # export_span needs id presets to work, and the generator can only
        # be swapped provider-wide, before the tracer is created.
        tracer_provider.id_generator = _PresetIdGenerator()
        self._tracer = otel_trace.get_tracer("seal", tracer_provider=tracer_provider)

    def shutdown(self) -> None:
        """Flush and stop the exporter; spans emitted after this are dropped.

        The provider also registers an atexit shutdown, but calling this from
        an explicit shutdown path (server lifespan) flushes the batch exporter
        before the event loop is torn down.
        """
        self._provider.shutdown()

    def export_span(
        self,
        name: str,
        context: proto.TraceContext,
        end_ns: int,
        attributes: dict[str, Any] | None = None,
        *,
        error: str | None = None,
    ) -> None:
        """Export a finished span under a minted ``proto.TraceContext``."""
        if context.started_at_ns is None:
            return
        # an empty (never the ambient) otel context, so the span parents
        # only on what the minted context names.
        parent_ctx = otel_context.Context()
        if context.parent_span_id is not None:
            parent_ctx = otel_trace.set_span_in_context(
                otel_trace.NonRecordingSpan(
                    _span_context(context, context.parent_span_id)
                ),
                parent_ctx,
            )
        token = _preset_ids.set((int(context.trace_id, 16), int(context.span_id, 16)))
        try:
            otel_span = self._tracer.start_span(
                name, context=parent_ctx, start_time=context.started_at_ns
            )
        finally:
            _preset_ids.reset(token)
        attrs = dict(attributes or {})
        if kind := _SPAN_KINDS.get(name):
            attrs["openinference.span.kind"] = kind
        for key, value in attrs.items():
            otel_span.set_attribute(key, value)
        if error is not None:
            otel_span.set_status(otel_trace.StatusCode.ERROR, error)
        otel_span.end(end_time=end_ns)

    async def wrap_span(self, span_: ai.telemetry.Span, /) -> AsyncGenerator[None, Any]:
        if _in_workflow_body():
            # returning before the first yield opts out of this span.
            return
        # the SDK's gen_ai name/attribute mapping is reused verbatim;
        # only the attribute enrichment and body gating differ here.
        otel_span = self._tracer.start_span(
            otel_adapter._name(span_), start_time=span_.started_at
        )
        token = (
            otel_context.attach(otel_trace.set_span_in_context(otel_span))
            if span_.set_as_current
            else None
        )
        try:
            # span end resumes with None; a span error is thrown in here.
            while (ev := (yield)) is not None:
                otel_span.add_event(
                    ev.name,
                    {k: _plain(v) for k, v in ev.attributes.items()},
                    timestamp=ev.time_ns,
                )
        finally:
            if token is not None:
                otel_context.detach(token)
            attributes = otel_adapter._attributes(span_) | _extra_attributes(span_)
            for key, value in attributes.items():
                otel_span.set_attribute(key, value)
            if span_.error is not None:
                if isinstance(span_.error, Exception):
                    otel_span.record_exception(span_.error)
                otel_span.set_status(otel_trace.StatusCode.ERROR, str(span_.error))
            otel_span.end(end_time=span_.ended_at)
