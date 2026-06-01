from __future__ import annotations

import asyncio
import collections.abc
import typing

import ai
import pydantic

import durable_agent.storage

__all__ = ["get_readable", "get_writable"]

_DEFAULT_NAMESPACE = "default"
_DEFAULT_POLL_INTERVAL = 0.05
_HANDLE_TYPE = "seal.durable_agent.writable_stream"
_EVENT_ADAPTER: pydantic.TypeAdapter[ai.events.AgentEvent] = pydantic.TypeAdapter(
    ai.events.AgentEvent
)


class _WritableStream(dict[str, str]):
    def __init__(self, stream_id: str, namespace: str, *, writable: bool) -> None:
        super().__init__(
            {
                "type": _HANDLE_TYPE,
                "stream_id": stream_id,
                "namespace": namespace,
            }
        )
        self.stream_id = stream_id
        self.namespace = namespace
        self._writable = writable

    async def write(self, event: ai.events.AgentEvent | dict[str, typing.Any]) -> int:
        if not self._writable:
            raise RuntimeError(
                "cannot write to a workflow stream from workflow context; "
                "pass the handle to a step and reopen it with get_writable()"
            )

        validated = _EVENT_ADAPTER.validate_python(event)
        return await durable_agent.storage.append_event(
            self.stream_id,
            self.namespace,
            validated.model_dump(mode="json"),
        )

    async def close(self) -> None:
        if not self._writable:
            raise RuntimeError(
                "cannot close a workflow stream from workflow context; "
                "pass the handle to a step and reopen it with get_writable()"
            )

        await durable_agent.storage.close_stream(self.stream_id, self.namespace)


async def get_writable(
    stream_id: str | collections.abc.Mapping[str, typing.Any] | None = None,
    *,
    namespace: str = _DEFAULT_NAMESPACE,
) -> _WritableStream:
    """Return a serializable durable stream handle.

    Workflow code may obtain the handle and pass it to steps. Step code should
    call this function with that handle, then write or close the returned object.
    """
    await durable_agent.storage.ensure_ready()

    workflow_context = _workflow_context()
    parsed_stream_id = _stream_id_from_handle(stream_id)
    parsed_namespace = _namespace_from_handle(stream_id) or namespace
    if parsed_stream_id is None:
        parsed_stream_id = _run_id_from_workflow_context(workflow_context)
    if parsed_stream_id is None:
        raise RuntimeError(
            "get_writable() requires a stream_id outside workflow context"
        )

    return _WritableStream(
        parsed_stream_id,
        parsed_namespace,
        writable=workflow_context is None,
    )


async def get_readable(
    stream_id: str,
    *,
    namespace: str = _DEFAULT_NAMESPACE,
    start_index: int = 0,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
) -> collections.abc.AsyncIterator[ai.events.AgentEvent]:
    """Yield stream events from ``start_index`` until the stream is closed."""
    if poll_interval < 0:
        raise ValueError("poll_interval must be non-negative")

    await durable_agent.storage.ensure_ready()
    next_index = start_index
    if next_index < 0:
        event_count = await durable_agent.storage.count_events(stream_id, namespace)
        next_index = max(0, event_count + next_index)

    while True:
        found = False
        async for index, data in durable_agent.storage.list_events(
            stream_id,
            namespace,
            next_index,
        ):
            found = True
            next_index = index + 1
            yield _EVENT_ADAPTER.validate_python(data)

        if found:
            continue

        if await durable_agent.storage.is_closed(stream_id, namespace):
            async for index, data in durable_agent.storage.list_events(
                stream_id,
                namespace,
                next_index,
            ):
                found = True
                next_index = index + 1
                yield _EVENT_ADAPTER.validate_python(data)
            if not found:
                return
            continue

        await asyncio.sleep(poll_interval)


def _stream_id_from_handle(
    value: str | collections.abc.Mapping[str, typing.Any] | None,
) -> str | None:
    if isinstance(value, str):
        return value
    if value is None:
        return None

    raw = value.get("stream_id")
    if not isinstance(raw, str) or not raw:
        raise ValueError("stream handle is missing stream_id")
    handle_type = value.get("type")
    if handle_type is not None and handle_type != _HANDLE_TYPE:
        raise ValueError("stream handle has an invalid type")
    return raw


def _namespace_from_handle(
    value: str | collections.abc.Mapping[str, typing.Any] | None,
) -> str | None:
    if not isinstance(value, collections.abc.Mapping):
        return None

    raw = value.get("namespace")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError("stream handle has an invalid namespace")
    return raw


def _workflow_context() -> typing.Any | None:
    try:
        from vercel._internal.workflow import runtime
    except ImportError:
        return None

    try:
        context = runtime.WorkflowOrchestratorContext.current()
    except LookupError:
        return None

    return context


def _run_id_from_workflow_context(context: typing.Any | None) -> str | None:
    if context is None:
        return None

    for event in getattr(context, "events", ()):
        server_props = getattr(event, "server_props", None)
        run_id = getattr(server_props, "run_id", None)
        if isinstance(run_id, str) and run_id:
            return run_id

    return None
