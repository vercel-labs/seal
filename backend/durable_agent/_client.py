"""CLI to try the durable agent end to end.

Starts one session over HTTP, prints every stream event human-readably, and
follows any subagent's stream concurrently (indented). When the main session
parks (`session.waiting`) it closes the session so the run finishes cleanly.

    uv run python -m durable_agent._client
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

BASE_URL = "http://localhost:3000/api"
PROMPT = (
    "this is a test run. use one tool and create a subagent. "
    "have subagent use one tool as well"
)


def _render(event: dict[str, Any], prefix: str) -> None:
    """Print one stream event as a single human-readable line."""
    kind = event.get("kind")

    if kind == "lifecycle":
        data = event.get("data") or {}
        detail = ", ".join(f"{k}={v}" for k, v in data.items())
        print(f"{prefix}* {event.get('type')}" + (f" ({detail})" if detail else ""))
        return

    match kind:
        case "text_delta":
            print(event.get("chunk", ""), end="", flush=True)
        case "text_end":
            print()
        case "tool_end":
            call = event.get("tool_call") or {}
            print(f"{prefix}> tool {call.get('tool_name')} {call.get('tool_args')}")
        case "tool_call_result":
            for result in event.get("results") or []:
                name, value = result.get("tool_name"), result.get("result")
                print(f"{prefix}< result {name}: {value}")
        case "hook":
            hook = event.get("hook") or {}
            print(f"{prefix}~ hook {hook.get('tool_name')}")


async def _follow(
    client: httpx.AsyncClient,
    session_id: str,
    prefix: str,
    children: set[str],
    tasks: list[asyncio.Task[None]],
) -> None:
    """Print a session's events; spawn a follower for each subagent."""
    async with client.stream("GET", f"/session/{session_id}/events") as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            event = json.loads(line)
            _render(event, prefix)

            if event.get("kind") == "lifecycle":
                event_type = event.get("type")
                data = event.get("data") or {}

                if event_type == "subagent.called":
                    child = data["child_session_id"]
                    if child not in children:
                        children.add(child)
                        tasks.append(
                            asyncio.create_task(
                                _follow(client, child, prefix + "    ", children, tasks)
                            )
                        )

                # gated tool calls; auto-approve everything and log what we
                # let through so the policy is visible.
                elif event_type == "tool_approval.requested" and prefix == "":
                    requests = data.get("requests") or []
                    for request in requests:
                        print(
                            f"{prefix}! auto-approving "
                            f"{request.get('tool_name')} {request.get('args')}"
                        )
                    await client.post(
                        f"/session/{session_id}/approve",
                        params={"turn_index": data.get("turn_index", 0)},
                        json={
                            "tool_approvals": [
                                {"tool_call_id": request["tool_call_id"], "granted": True}
                                for request in requests
                            ]
                        },
                    )

                # the root session parks after the one-shot turn; close it so
                # the run finishes and this stream ends.
                elif event_type == "session.waiting" and prefix == "":
                    turn_index = data.get("turn_index", 0)
                    await client.post(
                        f"/session/{session_id}/close",
                        params={"turn_index": turn_index},
                    )


async def main() -> None:
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=timeout) as client:
        started = (await client.post("/session", json={"prompt": PROMPT})).json()
        session_id = started["session_id"]
        print(f"session {session_id}\nprompt: {PROMPT}\n")

        children: set[str] = set()
        tasks: list[asyncio.Task[None]] = []
        await _follow(client, session_id, "", children, tasks)
        if tasks:
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
