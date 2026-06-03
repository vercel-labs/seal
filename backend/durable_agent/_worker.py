"""Worker entrypoint for the durable agent.

`vercel dev` runs this as the `__wkf_*` queue consumer. Importing the workflow
modules constructs their `Workflows()` registries, which registers the queue
handlers that actually drive `run_session` / `run_turn` to completion.

The preamble (env defaults + the `ai` sandbox passthrough) must run before the
workflow libraries are imported, so it lives at module top.
"""

from __future__ import annotations

import os

_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault(
    "WORKFLOW_LOCAL_DATA_DIR",
    os.path.join(_BACKEND_DIR, ".workflow-data"),
)
os.environ.setdefault(
    "SEAL_DURABLE_AGENT_STREAMS_DIR",
    os.path.join(_BACKEND_DIR, ".durable_agent_streams"),
)

import vercel._internal.workflow.py_sandbox  # noqa: E402

# The workflow sandbox rewrites `os` in a way that breaks dependents at import
# time; serve these from the host unchanged. `ai` needs `shutil`/`os`; the
# durable_agent modules import `pathlib` (via storage) at the top level.
vercel._internal.workflow.py_sandbox._PASSTHROUGHS.update({"ai", "pathlib"})

# Importing the driver pulls in turn/session/stream; constructing each module's
# `Workflows()` registers its queue handlers.
import durable_agent.driver  # noqa: E402, F401
import durable_agent.turn  # noqa: E402, F401
