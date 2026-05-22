"""Neon PostgreSQL pool and schema setup."""

from __future__ import annotations

import os

import asyncpg  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Schema (inlined so the backend has no runtime dependency on repo layout)
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq         INTEGER,
    turn_id     TEXT,
    role        TEXT NOT NULL,
    parts       JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE messages ADD COLUMN IF NOT EXISTS seq INTEGER;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS turn_id TEXT;

WITH ranked AS (
    SELECT
        id,
        row_number() OVER (
            PARTITION BY session_id
            ORDER BY created_at, id
        ) - 1 AS seq
    FROM messages
    WHERE seq IS NULL
)
UPDATE messages
SET seq = ranked.seq
FROM ranked
WHERE messages.id = ranked.id;

ALTER TABLE messages ALTER COLUMN seq SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_messages_session_seq
    ON messages(session_id, seq, created_at, id);

"""


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Return the shared pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=os.environ["DATABASE_URL"])
    return _pool


async def ensure_schema() -> None:
    """Run ``CREATE TABLE IF NOT EXISTS`` for every table."""
    pool = await get_pool()
    await pool.execute(_SCHEMA)


async def close_pool() -> None:
    """Gracefully close the pool (call from FastAPI shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
