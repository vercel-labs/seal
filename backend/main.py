"""FastAPI application entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import fastapi
import fastapi.middleware.cors

import db
import sessions
from routers import chat, session


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI) -> AsyncIterator[None]:
    """Create configured storage on startup, close DB pool on shutdown."""
    await sessions.ensure_schema()
    yield
    await db.close_pool()


app = fastapi.FastAPI(
    title="seal",
    description="Seal – personal AI assistant",
    lifespan=lifespan,
)

app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(session.router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
