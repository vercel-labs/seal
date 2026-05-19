"""Agent definition and tool declarations."""

from __future__ import annotations

import asyncio

import ai
import httpx


@ai.tool(require_approval=True)
async def bash(command: str, timeout: int | None = None) -> str:
    """Execute a bash command.

    Use timeout (seconds) to limit long-running commands.
    """
    proc = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        return f"Command timed out after {timeout}s."

    output = stdout.decode() if stdout else ""
    if proc.returncode != 0:
        return f"[exit code {proc.returncode}]\n{output}"
    return output


@ai.tool(require_approval=True)
async def web_fetch(
    url: str, method: str = "GET", headers: str = "", body: str = ""
) -> str:
    """Fetch a URL and return the response.

    Args:
        url: The URL to fetch.
        method: HTTP method (GET, POST, PUT, DELETE, etc.).
        headers: Optional headers as newline-separated "Key: Value" pairs.
        body: Optional request body for POST/PUT.
    """
    parsed_headers: dict[str, str] = {}
    for line in headers.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            parsed_headers[k.strip()] = v.strip()

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.request(
            method,
            url,
            headers=parsed_headers or None,
            content=body or None,
        )

    parts = [
        f"HTTP {response.status_code}",
        *(f"{k}: {v}" for k, v in response.headers.items()),
        "",
        response.text[:50_000],
    ]
    return "\n".join(parts)


SYSTEM = """You are a helpful assistant with access to a bash shell and the internet."""

TOOLS: list[ai.AgentTool] = [bash, web_fetch]

_TITLE_PROMPT = (
    "Generate a concise 3-6 word title for a conversation that starts with "
    "the following message. Reply with ONLY the title, no quotes or punctuation."
)


def get_model() -> ai.Model:
    """Create the primary LLM instance."""
    return ai.get_model("anthropic/claude-opus-4.6")


def _get_fast_model() -> ai.Model:
    """Cheap / fast model for lightweight tasks like title generation."""
    return ai.get_model("anthropic/claude-sonnet-4.6")


async def generate_title(first_message: str) -> str:
    """Generate a short title for a session using a cheap LLM call."""
    model = _get_fast_model()
    messages = [
        ai.system_message(_TITLE_PROMPT),
        ai.user_message(first_message),
    ]
    async with ai.stream(model, messages) as stream:
        async for _ in stream:
            pass
        return stream.text.strip()


# Agent with human-in-the-loop tool approval via ``require_approval=True``
# on each tool.  The default agent loop handles approval suspension and
# resume natively, so no custom loop is needed.
seal = ai.agent(tools=TOOLS)
