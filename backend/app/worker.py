"""Worker entrypoint for the durable agent.

`vercel dev` runs this as the `__wkf_*` queue consumer. Importing the workflow
modules constructs their `Workflows()` registries, which registers the queue
handlers that actually drive `run_session` / `run_turn` to completion.

The preamble (env defaults + the `ai` sandbox passthrough) must run before the
workflow libraries are imported, so it lives at module top.
"""

from __future__ import annotations

import logging
import os

# the vercel runtime forces the root logger to INFO, and httpx logs every
# request at INFO; the worker's constant HTTP traffic makes that unreadable.
logging.getLogger("httpx").setLevel(logging.WARNING)


_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault(
    "WORKFLOW_LOCAL_DATA_DIR",
    os.path.join(_BACKEND_DIR, ".workflow-data"),
)
os.environ.setdefault(
    "SEAL_STREAMS_DIR",
    os.path.join(_BACKEND_DIR, ".seal"),
)

import vercel._internal.workflow.py_sandbox  # noqa: E402

# Need to make `ai` a passthrough currently because of how it uses
# uuid, though honestly that does seem to be the bug causing
# trouble. `rich` also needs host access for terminal detection.
vercel._internal.workflow.py_sandbox._PASSTHROUGHS.update({"rich", "modelsdotdev"})

# Importing the driver pulls in turn/session/stream; constructing each module's
# `Workflows()` registers its queue handlers.
import agent.driver  # noqa: E402, F401
import agent.turn  # noqa: E402, F401
