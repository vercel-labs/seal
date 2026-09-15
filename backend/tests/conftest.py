"""Shared test setup for the Temporal port."""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator, Iterator, Sequence
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ai  # noqa: E402
import ai.models as models  # noqa: E402
import ai.types.events as events_  # noqa: E402
import ai.types.messages as messages_  # noqa: E402
import pydantic
import pytest
import temporalio.client
import temporalio.contrib.workflow_streams
import temporalio.testing
import temporalio.worker
import temporalio.workflow

from agent import driver, proto, stream, temporal, turn  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SEAL_SESSIONS_DIR", str(tmp_path / "sessions"))
    yield


@temporalio.workflow.defn(sandboxed=False)
class StreamFixtureWorkflow:
    def __init__(self) -> None:
        workflow_stream = temporalio.contrib.workflow_streams.WorkflowStream()
        self.events = workflow_stream.topic(
            stream.EVENTS_TOPIC, type=cast(Any, proto.StreamEvent)
        )

    @temporalio.workflow.run
    async def run(self, events: list[proto.StreamEvent]) -> None:
        for event in events:
            self.events.publish(event)
        await temporalio.workflow.wait_condition(lambda: False)


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

    # Provider is now a frozen pydantic model; opt this test double back into
    # mutability and keep the scripted state out of serialization/hashing.
    model_config = pydantic.ConfigDict(frozen=False)

    provider_class_id: str = "mock"
    name: str = "mock"
    default_base_url: str = "http://mock.test"

    responses: list[list[messages_.Message]] = pydantic.Field(
        default_factory=list, exclude=True
    )
    keyed_responses: dict[str, list[messages_.Message]] = pydantic.Field(
        default_factory=dict, exclude=True
    )
    call_count: int = pydantic.Field(default=0, exclude=True)
    calls: list[list[messages_.Message]] = pydantic.Field(
        default_factory=list, exclude=True
    )

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
        self.calls.append(messages)
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
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: Any = None,
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
            elif isinstance(part, messages_.FilePart):
                yield events_.FileEvent(
                    block_id=part.id,
                    data=part.data,
                    media_type=part.media_type,
                    filename=part.filename,
                )
    yield events_.StreamEnd()


@pytest.fixture
def mock_llm() -> Iterator[MockProvider]:
    """Reset the scripted provider; tests append to ``responses``."""
    MOCK_PROVIDER.responses = []
    MOCK_PROVIDER.keyed_responses = {}
    MOCK_PROVIDER.call_count = 0
    MOCK_PROVIDER.calls = []
    yield MOCK_PROVIDER
    MOCK_PROVIDER.responses = []
    MOCK_PROVIDER.keyed_responses = {}
    MOCK_PROVIDER.calls = []


@pytest.fixture
async def temporal_client() -> AsyncGenerator[temporalio.client.Client]:
    environment = await temporalio.testing.WorkflowEnvironment.start_time_skipping(
        data_converter=temporal.DATA_CONVERTER
    )
    temporal._client = environment.client
    async with temporalio.worker.Worker(
        environment.client,
        task_queue="seal-temporal",
        workflows=[driver.SessionWorkflow, turn.TurnWorkflow, StreamFixtureWorkflow],
        activities=turn.ACTIVITIES,
    ):
        yield environment.client
    temporal._client = None
    await environment.shutdown()


@pytest.fixture
def scripted_model(
    monkeypatch: pytest.MonkeyPatch, mock_llm: MockProvider
) -> MockProvider:
    provider_type = type(ai.get_model("gateway:openai/gpt-5.6-luna").provider)

    def stream(
        _provider: models.Provider,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: Any = None,
    ) -> AsyncGenerator[events_.Event]:
        return mock_llm.stream(
            model,
            messages,
            tools=tools,
            output_type=output_type,
            params=params,
        )

    monkeypatch.setattr(provider_type, "stream", stream)
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
