from __future__ import annotations

import asyncio
import collections.abc

import pydantic
from typing import Any, Literal

from durable_agent import proto, storage

__all__ = ["get_readable", "get_writable"]


WRITABLE_STREAM_HANDLE_TYPE = "seal.durable_agent.writable_stream"
DEFAULT_STREAM_NAMESPACE = "default"
DEFAULT_STREAM_POLL_INTERVAL = 0.05


class WritableStreamHandle(pydantic.BaseModel):
    type: Literal["seal.durable_agent.writable_stream"] = WRITABLE_STREAM_HANDLE_TYPE
    stream_id: str
    namespace: str = DEFAULT_STREAM_NAMESPACE

    async def write(self, event: proto.StreamEvent | dict[str, Any]) -> int:
        import durable_agent.storage

        validated = proto.STREAM_EVENT_ADAPTER.validate_python(event)
        return await durable_agent.storage.append_event(
            self.stream_id,
            self.namespace,
            validated.model_dump(mode="json"),
        )

    async def close(self) -> None:
        import durable_agent.storage

        await durable_agent.storage.close_stream(self.stream_id, self.namespace)


async def get_writable(
    stream_id: str,
    *,
    namespace: str = DEFAULT_STREAM_NAMESPACE,
) -> WritableStreamHandle:
    """Return the writable handle for ``stream_id``."""
    await storage.ensure_ready()
    return WritableStreamHandle(
        stream_id=stream_id,
        namespace=namespace,
    )


async def get_readable(
    stream_id: str,
    *,
    namespace: str = DEFAULT_STREAM_NAMESPACE,
    start_index: int = 0,
    poll_interval: float = DEFAULT_STREAM_POLL_INTERVAL,
) -> collections.abc.AsyncIterator[proto.StreamEvent]:
    """Yield stream events from ``start_index`` until the stream is closed."""
    if poll_interval < 0:
        raise ValueError("poll_interval must be non-negative")

    await storage.ensure_ready()
    next_index = start_index
    if next_index < 0:
        event_count = await storage.count_events(stream_id, namespace)
        next_index = max(0, event_count + next_index)

    while True:
        found = False
        async for index, data in storage.list_events(
            stream_id,
            namespace,
            next_index,
        ):
            found = True
            next_index = index + 1
            yield proto.STREAM_EVENT_ADAPTER.validate_python(data)

        if found:
            continue

        if await storage.is_closed(stream_id, namespace):
            async for index, data in storage.list_events(
                stream_id,
                namespace,
                next_index,
            ):
                found = True
                next_index = index + 1
                yield proto.STREAM_EVENT_ADAPTER.validate_python(data)
            if not found:
                return
            continue

        await asyncio.sleep(poll_interval)


# current design implies that get_writable gets called inside a step with session_id
# if there's encryption involved, writable handle will have to be passed all the way 
# down session -> step -> turn -> step, in order for the turn step (e.g. llm call) to 
# write to session's stream.
