"""Worker entrypoint for the durable agent.

Importing `agent` constructs the `Workflows()` registry, which registers the queue
handlers that actually drive `run_session` / `run_turn` to completion.

The preamble (env defaults, log settup) must run before the
workflow libraries are imported, so it lives at module top.
"""

from __future__ import annotations

import logging
import os

# the vercel runtime forces the root logger to INFO, and httpx logs every
# request at INFO; the worker's constant HTTP traffic makes that unreadable.
logging.getLogger("httpx").setLevel(logging.WARNING)
# every queue delivery is an HTTP POST to this worker, so uvicorn's access log
# prints a line per step. the dev runtime applies its uvicorn dictConfig *after*
# importing this module, resetting the logger's level — but dictConfig never
# removes attached filters, so a filter is what survives.
logging.getLogger("uvicorn.access").addFilter(lambda record: False)
# uvicorn's reloader passes watch_filter=None to watchfiles and applies its
# *.py filter only afterward, so every .workflow-data/.seal write logs an INFO
# "N changes detected" without causing a reload; drop those count lines.
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)


_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault(
    "WORKFLOW_LOCAL_DATA_DIR",
    os.path.join(_BACKEND_DIR, ".workflow-data"),
)
os.environ.setdefault(
    "SEAL_STREAMS_DIR",
    os.path.join(_BACKEND_DIR, ".seal"),
)

# Importing the driver pulls in turn/session/stream.
import agent.driver  # noqa: E402, F401
import agent.turn  # noqa: E402, F401
from agent import workflow as workflow  # noqa: E402, F401
