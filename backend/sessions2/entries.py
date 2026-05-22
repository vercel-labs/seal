from __future__ import annotations

from typing import Annotated, Any, Literal

import ai
import pydantic


class ModelSettings(pydantic.BaseModel):
    model_id: str | None = None
    params: dict[str, Any] = pydantic.Field(default_factory=dict)


class BaseEntry(pydantic.BaseModel):
    id: str
    parent_id: str | None
    timestamp: str


class MessageEntry(BaseEntry):
    # records messages
    kind: Literal["message"] = "message"
    message: ai.messages.Message


class ModelSettingsEntry(BaseEntry):
    # records change in settings e.g. model, thinking
    kind: Literal["model_settings"] = "model_settings"
    settings: ModelSettings = pydantic.Field(default_factory=ModelSettings)


class SessionInfoEntry(BaseEntry):
    # records change in settings e.g. model, thinking
    kind: Literal["session_info"] = "session_info"
    title: str | None = None


class LeafEntry(BaseEntry):
    # records the active branch pointer.
    kind: Literal["leaf"] = "leaf"
    target_id: str | None


class CustomEntry(BaseEntry):
    # stores app/extension state; not sent to the model.
    kind: Literal["custom"] = "custom"
    custom_type: str
    data: Any | None = None


Entry = Annotated[
    MessageEntry | ModelSettingsEntry | SessionInfoEntry | LeafEntry | CustomEntry,
    pydantic.Field(discriminator="kind"),
]
