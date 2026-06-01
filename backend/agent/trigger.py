"""Compatibility shim for the renamed durable agent server."""

from __future__ import annotations

from agent import server

app = server.app
_tail_events = server._tail_events
run = server.run
run_stream = server.run_stream
session = server.session
session_events = server.session_events
session_subagent = server.session_subagent
session_turn = server.session_turn
status = server.status
stream = server.stream


if __name__ == "__main__":
    server.main()
