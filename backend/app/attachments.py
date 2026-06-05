"""Attachment helpers shared by chat routes and workflow steps."""

from __future__ import annotations

import base64

from ai import messages as ai_messages
from vercel.blob import AsyncBlobClient

# Prefix used by proxy URLs returned from the upload endpoint.
# Includes /api so the browser can fetch directly (Vercel routes /api/* to
# the backend and strips the prefix before forwarding).
FILES_PREFIX = "/api/files/"


async def inline_file_parts(
    messages: list[ai_messages.Message],
) -> list[ai_messages.Message]:
    """Replace proxy-URL file parts with inline base64 data URLs."""
    result: list[ai_messages.Message] = []
    for msg in messages:
        new_parts: list[ai_messages.Part] = []
        for part in msg.parts:
            pathname = (
                part.data[len(FILES_PREFIX) :]
                if (
                    isinstance(part, ai_messages.FilePart)
                    and isinstance(part.data, str)
                    and part.data.startswith(FILES_PREFIX)
                )
                else None
            )
            if isinstance(part, ai_messages.FilePart) and pathname is not None:
                async with AsyncBlobClient() as client:
                    blob = await client.get(pathname, access="private")

                b64 = base64.b64encode(blob.content).decode("ascii")
                media_type = blob.content_type or part.media_type
                data_url = f"data:{media_type};base64,{b64}"

                new_parts.append(
                    part.model_copy(update={"data": data_url, "media_type": media_type})
                )
            else:
                new_parts.append(part)

        result.append(msg.model_copy(update={"parts": new_parts}))
    return result
