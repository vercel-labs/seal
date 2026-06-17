"""Scripted model support for e2e tests."""

from __future__ import annotations

import collections.abc
import json
import pathlib
from typing import Any

import ai
import ai.models as models
import ai.types.events as events_
import ai.types.messages as messages_
import pydantic


class ScriptedProvider(models.Provider):
    def __init__(self, path: pathlib.Path) -> None:
        super().__init__(name="scripted", base_url="http://scripted.test")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.responses = [
            [_message(item) for item in turn] for turn in data.get("responses", [])
        ]
        self.keyed_responses = {
            key: [_message(item) for item in turn]
            for key, turn in data.get("keyed_responses", {}).items()
        }

    async def list_models(self) -> list[str]:
        return []

    def stream(
        self,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: collections.abc.Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: Any = None,
        protocol: Any = None,
    ) -> collections.abc.AsyncGenerator[events_.Event]:
        last_user = next((m.text for m in reversed(messages) if m.role == "user"), "")
        for key, response in self.keyed_responses.items():
            if key in last_user:
                return _emit_events(response)
        if not self.responses:
            raise RuntimeError("ScriptedProvider: no more responses configured")
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


def install(path: str) -> None:
    model = models.Model(
        id="scripted-model", provider=ScriptedProvider(pathlib.Path(path))
    )

    def get_scripted_model(
        model_id: str | None = None,
        *,
        protocol: ai.ProviderProtocol[Any] | None = None,
    ) -> models.Model:
        _ = (model_id, protocol)
        return model

    ai.get_model = get_scripted_model  # ty: ignore[invalid-assignment]


def _message(data: dict[str, Any]) -> messages_.Message:
    parts: list[messages_.Part] = []
    text = data.get("text")
    if text is not None:
        parts.append(messages_.TextPart(text=str(text)))
    for tool in data.get("tools", []):
        parts.append(
            messages_.ToolCallPart(
                tool_call_id=str(tool["id"]),
                tool_name=str(tool["name"]),
                tool_args=json.dumps(tool.get("args", {})),
            )
        )
    return messages_.Message(role="assistant", parts=parts)


async def _emit_events(
    seq: list[messages_.Message],
) -> collections.abc.AsyncGenerator[events_.Event]:
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
                    tool_call_id=part.tool_call_id,
                    tool_name=part.tool_name,
                )
                if part.tool_args:
                    yield events_.ToolDelta(
                        tool_call_id=part.tool_call_id,
                        chunk=part.tool_args,
                    )
                yield events_.ToolEnd(tool_call_id=part.tool_call_id, tool_call=part)
    yield events_.StreamEnd()
