"""Neon PostgreSQL pool shared by FastAPI and Temporal activities.

The cache is keyed by event loop so tests and reloads cannot reuse asyncpg
connections created on a closed loop.
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
