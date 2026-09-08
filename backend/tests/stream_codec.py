import pydantic

from agent import proto

adapter: pydantic.TypeAdapter[proto.StreamEvent] = pydantic.TypeAdapter(
    proto.StreamEvent
)
