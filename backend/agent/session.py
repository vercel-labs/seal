"""Durable session snapshots.

Each session appends one ``proto.SessionState`` snapshot per mutation to the
shared store under the ``session`` namespace; the tail is the current state.
The single-writer-per-session invariant (the driver workflow body) makes
tail reads safe.
"""

from __future__ import annotations

from agent import proto, storage

__all__ = ["read_session", "write_session"]

SESSION_NAMESPACE = "session"


async def write_session(state: proto.SessionState) -> int:
    """Append ``state`` as the latest snapshot; return its index."""
    await storage.ensure_ready()
    return await storage.store().append(
        state.session_id,
        SESSION_NAMESPACE,
        state.model_dump(mode="json"),
    )


async def read_session(session_id: str) -> proto.SessionState | None:
    """Return the latest snapshot for ``session_id``, or ``None`` if absent."""
    await storage.ensure_ready()
    backend = storage.store()

    tail_index, _ = await backend.info(session_id, SESSION_NAMESPACE)
    if tail_index < 0:
        return None

    records = await backend.read(session_id, SESSION_NAMESPACE, tail_index)
    if not records:
        return None

    _, data = records[-1]
    return proto.SessionState.model_validate(data)
