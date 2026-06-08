"""Neon PostgreSQL pool.

The FastAPI service runs on one long-lived ASGI loop, so a cached pool is safe
there. The durable-agent worker is different: a warm process handles each queue
message on a *fresh* event loop and the host closes that loop when the message
finishes. An asyncpg pool (and its connections) is bound to the loop that
created it, so a globally cached pool poisons the next invocation — surfacing as
"another operation is in progress" or "Event loop is closed".

We therefore key the cache by the running loop and rebuild when the loop
changes. We never touch the stale pool (its transports already died with its
loop, so ``close``/``terminate`` would just re-raise "Event loop is closed");
the dead pool is simply dropped for GC.
"""

from __future__ import annotations

import asyncio
import os

import asyncpg  # type: ignore[import-untyped]

_pool: asyncpg.Pool | None = None
_pool_loop: asyncio.AbstractEventLoop | None = None


async def get_pool() -> asyncpg.Pool:
    """Return the pool bound to the running loop, creating it on first call."""
    global _pool, _pool_loop
    loop = asyncio.get_running_loop()
    if _pool is not None and _pool_loop is loop:
        return _pool
    # New (or first) loop: abandon any stale pool and build one on this loop.
    # min_size=0 avoids opening connections that would outlive a short-lived
    # worker loop; connections are made lazily on first use and recycled fast.
    _pool = await asyncpg.create_pool(
        dsn=os.environ["DATABASE_URL"],
        min_size=0,
        max_inactive_connection_lifetime=60.0,
    )
    _pool_loop = loop
    return _pool


async def close_pool() -> None:
    """Gracefully close the pool (call from FastAPI shutdown)."""
    global _pool, _pool_loop
    if _pool is not None and _pool_loop is asyncio.get_running_loop():
        await _pool.close()
    _pool = None
    _pool_loop = None
