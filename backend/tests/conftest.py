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
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import harness`

import ai  # noqa: E402
import ai.models as models  # noqa: E402
import ai.types.events as events_  # noqa: E402
import ai.types.messages as messages_  # noqa: E402
import harness  # noqa: E402
import vercel._internal.workflow.world as wf_world  # noqa: E402

import agent  # noqa: E402
from agent import storage, stream  # noqa: E402


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
    """Provider whose ``stream`` replays scripted responses (no network).

    ``responses`` is a FIFO of turns. ``keyed_responses`` maps a substring of
    the conversation's last user message to a response; it wins over the FIFO
    and is not consumed. Concurrent sessions (parallel subagents) hit the
    model in nondeterministic order, so their responses must be keyed by the
    child's prompt — both to keep regeneration stable and to keep each child's
    content distinct, which is what lets a test catch a transcript attached to
    the wrong tool call.
    """

    def __init__(self) -> None:
        super().__init__(name="mock", base_url="http://mock.test", api_key_env=None)
        self.responses: list[list[messages_.Message]] = []
        self.keyed_responses: dict[str, list[messages_.Message]] = {}
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
        self.call_count += 1
        last_user = next((m.text for m in reversed(messages) if m.role == "user"), "")
        for key, response in self.keyed_responses.items():
            if key in last_user:
                return _emit_events(response)
        if not self.responses:
            raise RuntimeError("MockProvider: no more responses configured")
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
    MOCK_PROVIDER.keyed_responses = {}
    MOCK_PROVIDER.call_count = 0
    yield MOCK_PROVIDER
    MOCK_PROVIDER.responses = []
    MOCK_PROVIDER.keyed_responses = {}


# --- in-process engine fixtures ---------------------------------------------------


@pytest.fixture
async def world() -> AsyncGenerator[harness.InProcessWorld]:
    bridged = harness.InProcessWorld(agent.workflow)
    wf_world.set_world(bridged)
    try:
        yield bridged
        # let in-flight deliveries finish before judging the engine's state —
        # a run can look wedged while its final events are still queued.
        await bridged.drain()
    finally:
        for task in list(bridged._tasks):
            task.cancel()
        wf_world.set_world(None)
    assert bridged.errors == [], f"workflow delivery errors: {bridged.errors}"
    bridged.check_settled()
    bridged.check_replay_determinism()


@pytest.fixture
def scripted_model(
    monkeypatch: pytest.MonkeyPatch, mock_llm: MockProvider
) -> MockProvider:
    model = models.Model(id="mock-model", provider=mock_llm)
    monkeypatch.setattr(ai, "get_model", lambda model_id=None, **kwargs: model)
    return mock_llm


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


async def assert_stream_invariants(
    session_id: str,
    messages: Sequence[messages_.Message],
) -> None:
    """The durable stream answered every tool call with one consistent result.

    Pending placeholders may repeat (each interrupted pass re-announces them)
    and a cached result may be re-emitted on replay, but the final results
    must cover exactly the history's tool calls and re-emissions must carry
    the same payload — a different payload means results got cross-wired.
    Emission order is deliberately unconstrained: concurrent tools finish in
    whatever order they finish.
    """
    finals: dict[str, list[Any]] = {}
    async for event in stream.replay(session_id):
        if isinstance(event, events_.ToolCallResult):
            for part in event.results:
                if part.is_hook_pending:
                    continue
                # a cached re-emission mints a fresh part id; content is what
                # must not change between emissions.
                finals.setdefault(part.tool_call_id, []).append(
                    part.model_dump(mode="json", exclude={"id"})
                )

    history = [part.tool_call_id for message in messages for part in message.tool_calls]
    # subagent finals may be absent: the driver attaches them to history off
    # stream, and the live UI renders child progress from the child's own
    # stream. (They appear when a replayed turn re-emits them as cached.)
    required = {
        part.tool_call_id
        for message in messages
        for part in message.tool_calls
        if part.tool_name != "subagent"
    }
    assert required <= set(finals), (
        f"stream is missing results for {required - set(finals)}"
    )
    assert set(finals) <= set(history), (
        f"stream carries results for unknown calls: {set(finals) - set(history)}"
    )
    for tool_call_id, payloads in finals.items():
        assert all(payload == payloads[0] for payload in payloads), (
            f"stream re-emitted {tool_call_id} with a different result — "
            "results were wired to the wrong calls"
        )
