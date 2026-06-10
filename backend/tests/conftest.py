"""Shared test setup.

Every test runs against an isolated temp store (jsonl streams, session
metadata, workflow data); ``DATABASE_URL`` is cleared so the jsonl/file
backends are always selected.

``mock_llm`` scripts model responses for tests that drive the agent loop:
it attaches a queue of complete messages to ``MOCK_MODEL``'s provider, and
each ``ai.stream`` call pops one response and replays it as a realistic
event stream (Start/Delta/End triples bookended by StreamStart/StreamEnd).
This is the only test double in the suite — everything else (storage,
streams, hooks, workflows, the UI adapter) is real.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator, Iterator, Sequence
from pathlib import Path
from typing import Any

import pydantic
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ai  # noqa: E402
import ai.models as models  # noqa: E402
import ai.types.events as events_  # noqa: E402
import ai.types.messages as messages_  # noqa: E402

from agent import storage  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SEAL_STREAMS_DIR", str(tmp_path / "streams"))
    monkeypatch.setenv("SEAL_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("WORKFLOW_LOCAL_DATA_DIR", str(tmp_path / "workflow-data"))
    storage._locks.clear()
    yield
    storage._locks.clear()


# --- scripted model -------------------------------------------------------------


class MockProvider(models.Provider):
    """Provider whose ``stream`` replays scripted responses (no network)."""

    def __init__(self) -> None:
        super().__init__(name="mock", base_url="http://mock.test", api_key_env=None)
        self.responses: list[list[messages_.Message]] = []
        self.call_count = 0

    async def list_models(self) -> list[str]:
        return []

    def stream(
        self,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: Any = None,
        protocol: Any = None,
    ) -> AsyncGenerator[events_.Event]:
        if not self.responses:
            raise RuntimeError("MockProvider: no more responses configured")
        self.call_count += 1
        return _emit_events(self.responses.pop(0))

    async def generate(
        self,
        model: models.Model,
        messages: list[messages_.Message],
        params: Any,
        *,
        protocol: Any = None,
    ) -> messages_.Message:
        raise NotImplementedError


MOCK_PROVIDER = MockProvider()
MOCK_MODEL = models.Model(id="mock-model", provider=MOCK_PROVIDER)


async def _emit_events(
    seq: list[messages_.Message],
) -> AsyncGenerator[events_.Event]:
    """Replay complete messages as the event stream a real adapter would emit."""
    yield events_.StreamStart()
    for message in seq:
        for index, part in enumerate(message.parts):
            if isinstance(part, messages_.TextPart):
                block_id = f"text-{index}"
                yield events_.TextStart(block_id=block_id)
                if part.text:
                    yield events_.TextDelta(block_id=block_id, chunk=part.text)
                yield events_.TextEnd(block_id=block_id)
            elif isinstance(part, messages_.ToolCallPart):
                yield events_.ToolStart(
                    tool_call_id=part.tool_call_id, tool_name=part.tool_name
                )
                if part.tool_args:
                    yield events_.ToolDelta(
                        tool_call_id=part.tool_call_id, chunk=part.tool_args
                    )
                yield events_.ToolEnd(tool_call_id=part.tool_call_id, tool_call=part)
    yield events_.StreamEnd()


@pytest.fixture
def mock_llm() -> Iterator[MockProvider]:
    """Reset the scripted provider; tests append to ``responses``."""
    MOCK_PROVIDER.responses = []
    MOCK_PROVIDER.call_count = 0
    yield MOCK_PROVIDER
    MOCK_PROVIDER.responses = []


# --- message builders -----------------------------------------------------------


def text_msg(text: str) -> messages_.Message:
    return messages_.Message(role="assistant", parts=[messages_.TextPart(text=text)])


def tool_call_msg(
    *, tc_id: str, name: str, args: str = "{}", text: str | None = None
) -> messages_.Message:
    parts: list[messages_.Part] = []
    if text is not None:
        parts.append(messages_.TextPart(text=text))
    parts.append(
        messages_.ToolCallPart(tool_call_id=tc_id, tool_name=name, tool_args=args)
    )
    return messages_.Message(role="assistant", parts=parts)


# --- message-history invariants ---------------------------------------------------


def assert_message_invariants(messages: list[messages_.Message]) -> None:
    """The invariants every persisted history must hold.

    No duplicate message ids, every tool call answered exactly once, and the
    system prompt appears once, at the head.
    """
    ids = [message.id for message in messages]
    assert len(ids) == len(set(ids)), f"duplicate message ids: {ids}"

    tool_calls = [
        part.tool_call_id for message in messages for part in message.tool_calls
    ]
    tool_results = [
        part.tool_call_id for message in messages for part in message.tool_results
    ]
    assert len(tool_calls) == len(set(tool_calls)), f"duplicate calls: {tool_calls}"
    assert len(tool_results) == len(set(tool_results)), (
        f"duplicate results: {tool_results}"
    )
    assert set(tool_calls) == set(tool_results), (
        f"unanswered={set(tool_calls) - set(tool_results)} "
        f"orphaned={set(tool_results) - set(tool_calls)}"
    )

    system_indices = [
        index for index, message in enumerate(messages) if message.role == "system"
    ]
    assert system_indices in ([], [0]), f"system message misplaced: {system_indices}"
