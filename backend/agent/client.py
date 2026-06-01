"""Client for the minimal durable agent workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import typing
import uuid

import httpx

DEFAULT_BASE_URL = "http://localhost:3000/api"
DEFAULT_PROMPT = "Use bash to run pwd, then tell me the directory."


async def run(
    *,
    base_url: str,
    prompt: str,
    session_id: str,
    close: bool,
) -> None:
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=timeout,
    ) as client:
        response = await client.post("/session", json={"session_id": session_id})
        response.raise_for_status()
        session = response.json()
        if not isinstance(session, dict):
            raise RuntimeError("server returned a non-object session response")

        print(
            f"[client] session={session['session_id']} stream={session['stream_key']}",
            flush=True,
        )
        print(f"[client] prompt={prompt!r}", flush=True)

        turn_sent = False
        turn_done = False
        close_sent = False
        event_name = "message"
        data_lines: list[str] = []

        async with client.stream(
            "GET",
            f"/session/{session_id}/events",
            params={"start_index": 0},
        ) as stream:
            stream.raise_for_status()
            async for line in stream.aiter_lines():
                if line:
                    if line.startswith("event: "):
                        event_name = line.removeprefix("event: ")
                    elif line.startswith("data: "):
                        data_lines.append(line.removeprefix("data: "))
                    continue

                if not data_lines:
                    event_name = "message"
                    continue

                payload = json.loads("\n".join(data_lines))
                data_lines = []
                if not isinstance(payload, dict):
                    event_name = "message"
                    continue

                record = typing.cast(dict[str, typing.Any], payload)
                data = record.get("data")
                data = data if isinstance(data, dict) else {}
                kind = record.get("kind")
                kind = kind if isinstance(kind, str) else event_name
                index = record.get("index")
                prefix = f"[{index}]" if isinstance(index, int) else "[-]"
                print(f"{prefix} {kind}", flush=True)

                if kind == "model.event":
                    event = data.get("event")
                    event_type = data.get("event_type")
                    event_label = event_type if isinstance(event_type, str) else None
                    if isinstance(event, dict):
                        event_label = event_label or typing.cast(
                            str | None,
                            event.get("kind"),
                        )
                        chunk = event.get("chunk")
                        if isinstance(chunk, str) and chunk:
                            print(f"    text: {chunk}", flush=True)
                        tool_call = event.get("tool_call")
                        if isinstance(tool_call, dict):
                            print(
                                "    tool-call: "
                                f"{tool_call.get('tool_name')} "
                                f"{tool_call.get('tool_args')}",
                                flush=True,
                            )
                    if event_label is not None:
                        print(f"    event: {event_label}", flush=True)

                elif kind == "agent.event":
                    event = data.get("event")
                    event_type = data.get("event_type")
                    if isinstance(event_type, str):
                        print(f"    event: {event_type}", flush=True)
                    if isinstance(event, dict):
                        results = event.get("results")
                        if isinstance(results, list):
                            for result in results:
                                if isinstance(result, dict):
                                    print(
                                        "    tool-result: "
                                        f"{result.get('tool_name')} "
                                        f"{result.get('result')}",
                                        flush=True,
                                    )

                elif kind == "message.committed":
                    message = data.get("message")
                    if isinstance(message, dict):
                        role = message.get("role")
                        parts = message.get("parts")
                        print(f"    message: {role}", flush=True)
                        if isinstance(parts, list):
                            for part in parts:
                                if not isinstance(part, dict):
                                    continue
                                if part.get("kind") == "tool_call":
                                    print(
                                        "    committed tool-call: "
                                        f"{part.get('tool_name')} "
                                        f"{part.get('tool_args')}",
                                        flush=True,
                                    )
                                elif part.get("kind") == "tool_result":
                                    print(
                                        "    committed tool-result: "
                                        f"{part.get('tool_name')} "
                                        f"{part.get('result')}",
                                        flush=True,
                                    )
                                elif part.get("kind") == "text":
                                    text = part.get("text")
                                    if isinstance(text, str) and text:
                                        print(f"    committed text: {text}", flush=True)

                elif kind == "turn.completed":
                    turn_done = True

                elif kind == "session.waiting":
                    token = data.get("continuation_token")
                    if not isinstance(token, str):
                        event_name = "message"
                        continue
                    if not turn_sent:
                        print(f"[client] send turn token={token}", flush=True)
                        turn = await client.post(
                            f"/session/{session_id}/turn",
                            json={"prompt": prompt, "continuation_token": token},
                        )
                        turn.raise_for_status()
                        turn_sent = True
                    elif close and turn_done and not close_sent:
                        print(f"[client] close session token={token}", flush=True)
                        turn = await client.post(
                            f"/session/{session_id}/turn",
                            json={"continuation_token": token, "close": True},
                        )
                        turn.raise_for_status()
                        close_sent = True

                elif kind == "session.completed":
                    print("[client] complete", flush=True)

                event_name = "message"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help="Prompt to send after the session is waiting.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SEAL_AGENT_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--session-id", default=f"demo-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--keep-open", action="store_true")
    args = parser.parse_args()

    try:
        asyncio.run(
            run(
                base_url=typing.cast(str, args.base_url),
                prompt=typing.cast(str, args.prompt),
                session_id=typing.cast(str, args.session_id),
                close=not typing.cast(bool, args.keep_open),
            )
        )
    except httpx.ConnectError as error:
        print(
            f"Could not connect to {args.base_url}. Start `vercel dev -L` first.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
