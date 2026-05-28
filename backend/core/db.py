"""Neon PostgreSQL pool."""

from __future__ import annotations

import os

import asyncpg  # type: ignore[import-untyped]

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Return the shared pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=os.environ["DATABASE_URL"])
    return _pool


async def close_pool() -> None:
    """Gracefully close the pool (call from FastAPI shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
