"""The inbound boundary ``POST /chat`` branches on.

``server.post_chat`` decides between "resume a parked approval" and "start a
new turn" with::

    is_approval_resume = bool(approvals) and not (
        messages and messages[-1].role == "user")

after running the request history through ``ai_sdk.to_messages``. These tests
pin the SDK outputs that decision relies on, with the exact UIMessage shapes
the browser sends (``sendAutomaticallyWhen`` resubmits the full history after
an approval response).
"""

from __future__ import annotations

from typing import Any

import ai.agents.ui.ai_sdk as ai_sdk


def _ui(role: str, *parts: dict[str, Any], id: str = "m1") -> ai_sdk.UIMessage:
    return ai_sdk.UIMessage.model_validate(
        {"id": id, "role": role, "parts": list(parts)}
    )


def _answered_approval(granted: bool = True) -> dict[str, Any]:
    return {
        "type": "tool-bash",
        "toolCallId": "tc-1",
        "state": "approval-responded",
        "input": {"command": "ls"},
        "approval": {"id": "approve_tc-1", "approved": granted, "reason": "ok"},
    }


def test_approval_resubmission_resumes_the_parked_turn() -> None:
    # trailing message is the assistant turn holding the answered approval
    messages, approvals = ai_sdk.to_messages(
        [
            _ui("user", {"type": "text", "text": "run ls"}, id="u1"),
            _ui("assistant", _answered_approval(), id="a1"),
        ]
    )
    assert messages[-1].role != "user"
    assert [a.tool_call_id for a in approvals] == ["tc-1"]
    assert approvals[0].granted is True
    assert approvals[0].reason == "ok"


def test_trailing_user_message_starts_a_new_turn_even_with_past_approvals() -> None:
    messages, approvals = ai_sdk.to_messages(
        [
            _ui("user", {"type": "text", "text": "run ls"}, id="u1"),
            _ui("assistant", _answered_approval(), id="a1"),
            _ui("user", {"type": "text", "text": "now do more"}, id="u2"),
        ]
    )
    assert messages[-1].role == "user"
    assert messages[-1].text == "now do more"
