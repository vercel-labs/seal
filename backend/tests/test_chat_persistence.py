from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import ai
import pytest
from ai import messages as ai_messages
from ai.agents.hooks import TOOL_APPROVAL_HOOK_TYPE
from ai.agents.ui.ai_sdk import UIMessage, to_ui_messages

import sessions
from routers import chat as chat_router


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


@pytest.fixture
def repo(tmp_path: Path) -> sessions.JsonlSessionRepo:
    repo = sessions.JsonlSessionRepo(tmp_path / "sessions")
    sessions.set_repo(repo)
    try:
        yield repo
    finally:
        sessions.set_repo(None)


def test_jsonl_store_roundtrips_canonical_ai_messages(
    repo: sessions.JsonlSessionRepo,
) -> None:
    messages = [
        ai_messages.Message(
            id="user-1",
            role="user",
            parts=[ai_messages.TextPart(id="part-1", text="hello")],
        ),
        ai_messages.Message(
            id="assistant-1",
            turn_id="turn-1",
            role="assistant",
            parts=[ai_messages.TextPart(id="part-2", text="hi")],
        ),
    ]

    session = run(repo.create("session-1", title="Title"))
    run(session.save_messages(messages))
    loaded = run(session.load_messages())

    assert [m.model_dump(mode="json") for m in loaded] == [
        m.model_dump(mode="json") for m in messages
    ]
    metas = run(repo.list())
    assert metas[0].title == "Title"


def test_save_messages_is_idempotent_for_unchanged_history(
    repo: sessions.JsonlSessionRepo,
) -> None:
    messages = [ai.user_message("one"), ai.assistant_message("two")]
    session = run(repo.create("s1"))
    run(session.save_messages(messages))
    entries_before = len(session.entries())
    run(session.save_messages(messages))
    assert len(session.entries()) == entries_before


def test_save_messages_rewinds_leaf_on_diverging_user_message(
    repo: sessions.JsonlSessionRepo,
) -> None:
    original = [
        ai_messages.Message(
            id="u1", role="user", parts=[ai_messages.TextPart(text="one")]
        ),
        ai.assistant_message("two"),
    ]
    branch = [
        ai_messages.Message(
            id="u1", role="user", parts=[ai_messages.TextPart(text="alt")]
        ),
        ai.assistant_message("alt-reply"),
    ]

    session = run(repo.create("s1"))
    run(session.save_messages(original))
    run(session.save_messages(branch))

    loaded = run(session.load_messages())
    assert [m.text for m in loaded] == ["alt", "alt-reply"]
    # Original branch is still reachable from its leaf entry.
    leaf_entries = [
        e
        for e in session.entries()
        if isinstance(e, sessions.LeafEntry)
    ]
    assert len(leaf_entries) >= 1


def test_branch_fork_label_and_compact_use_active_branch(
    repo: sessions.JsonlSessionRepo,
) -> None:
    original = [ai.user_message("one"), ai.assistant_message("two")]
    branch = [ai.user_message("one"), ai.assistant_message("alternate")]

    session = run(repo.create("session-1"))
    run(session.save_messages(original))
    tree = session.build_tree()
    first_leaf = tree["leaf_id"]
    assert first_leaf is not None

    run(session.save_messages(branch))
    run(session.move_to(first_leaf))
    assert [m.text for m in run(session.load_messages())] == ["one", "two"]

    run(session.label(first_leaf, "checkpoint"))
    labeled_tree = session.build_tree()
    labeled = next(e for e in labeled_tree["entries"] if e["id"] == first_leaf)
    assert labeled["label"] == "checkpoint"

    forked = run(repo.fork("session-1", "session-2"))
    assert [m.text for m in run(forked.load_messages())] == ["one", "two"]

    path_messages = [
        e for e in forked.path() if isinstance(e, sessions.MessageEntry)
    ]
    first_kept = path_messages[-1].id  # keep the last message
    run(
        forked.compact(
            "summary text",
            first_kept_entry_id=first_kept,
            tokens_before=10,
        )
    )
    compacted = run(forked.load_messages())
    assert compacted[0].text.startswith("Previous conversation summary")
    assert compacted[-1].text == "two"


def test_fork_position_before_drops_target_user_message(
    repo: sessions.JsonlSessionRepo,
) -> None:
    messages = [
        ai_messages.Message(
            id="u1", role="user", parts=[ai_messages.TextPart(text="first")]
        ),
        ai.assistant_message("reply"),
        ai_messages.Message(
            id="u2", role="user", parts=[ai_messages.TextPart(text="second")]
        ),
        ai.assistant_message("reply2"),
    ]
    session = run(repo.create("src"))
    run(session.save_messages(messages))

    target_entry = next(
        e
        for e in session.entries()
        if isinstance(e, sessions.MessageEntry)
        and isinstance(e.message, ai_messages.Message)
        and e.message.id == "u2"
    )

    forked = run(
        repo.fork(
            "src", "fork", entry_id=target_entry.id, position="before"
        )
    )
    loaded = run(forked.load_messages())
    assert [m.text for m in loaded] == ["first", "reply"]


def test_move_to_with_summary_appends_branch_summary_entry(
    repo: sessions.JsonlSessionRepo,
) -> None:
    session = run(repo.create("s1"))
    run(session.save_messages([ai.user_message("hi"), ai.assistant_message("there")]))
    first_user_entry = next(
        e for e in session.entries() if isinstance(e, sessions.MessageEntry)
    )
    run(session.move_to(first_user_entry.id, summary="forking off here"))
    summaries = [
        e for e in session.entries() if isinstance(e, sessions.BranchSummaryEntry)
    ]
    assert len(summaries) == 1
    assert summaries[0].summary == "forking off here"
    assert summaries[0].from_id == first_user_entry.id


def test_session_persists_across_repo_reopen(
    repo: sessions.JsonlSessionRepo, tmp_path: Path
) -> None:
    session = run(repo.create("persist-1", title="Persisted"))
    run(session.save_messages([ai.user_message("hi"), ai.assistant_message("hey")]))

    fresh = sessions.JsonlSessionRepo(tmp_path / "sessions")
    reopened = run(fresh.open("persist-1"))
    assert reopened is not None
    assert reopened.meta.title == "Persisted"
    assert [m.text for m in run(reopened.load_messages())] == ["hi", "hey"]


def test_prepare_chat_request_accepts_ui_only_step_start_shape() -> None:
    stored_messages = [
        ai_messages.Message(
            id="user-1",
            role="user",
            parts=[ai_messages.TextPart(id="user-text-1", text="message 1")],
        ),
        ai_messages.Message(
            id="assistant-1",
            turn_id="turn-1",
            role="assistant",
            parts=[ai_messages.TextPart(id="assistant-text-1", text="reply 1")],
        ),
    ]
    request_messages = [
        UIMessage.model_validate(m.model_dump(mode="json", by_alias=True))
        for m in to_ui_messages(stored_messages)
    ]
    assistant_data = request_messages[1].model_dump(mode="json", by_alias=True)
    assistant_data["parts"] = [{"type": "step-start"}, *assistant_data["parts"]]
    request_messages[1] = UIMessage.model_validate(assistant_data)
    request_messages.append(
        UIMessage.model_validate(
            {
                "id": "user-2",
                "role": "user",
                "parts": [{"type": "text", "text": "message 2"}],
            }
        )
    )

    prepared = chat_router.prepare_chat_request(
        request_messages=request_messages,
        stored_messages=stored_messages,
    )

    assert prepared.has_work is True
    assert prepared.changed is True
    assert [(m.id, m.role) for m in prepared.messages] == [
        ("user-1", "user"),
        ("assistant-1", "assistant"),
        ("user-2", "user"),
    ]


def test_prepare_chat_request_accepts_approval_response() -> None:
    stored_messages = [
        ai_messages.Message(
            id="assistant-1",
            turn_id="turn-1",
            role="assistant",
            parts=[
                ai_messages.TextPart(id="text-1", text="Need approval"),
                ai_messages.ToolCallPart(
                    id="call-part-1",
                    tool_call_id="call-1",
                    tool_name="bash",
                    tool_args="{}",
                ),
            ],
        ),
        ai_messages.Message(
            id="tool-1",
            turn_id="turn-1",
            role="tool",
            parts=[
                ai_messages.ToolResultPart(
                    id="result-1",
                    tool_call_id="call-1",
                    tool_name="bash",
                    result="Pending",
                    is_error=True,
                    is_hook_pending=True,
                )
            ],
        ),
        ai_messages.Message(
            id="internal-1",
            turn_id="turn-1",
            role="internal",
            parts=[
                ai_messages.HookPart(
                    id="hook-part-1",
                    hook_id="approve_call-1",
                    hook_type=TOOL_APPROVAL_HOOK_TYPE,
                    status="pending",
                    metadata={"tool": "bash"},
                )
            ],
        ),
    ]
    request_data = [
        m.model_dump(mode="json", by_alias=True)
        for m in to_ui_messages(stored_messages)
    ]
    for part in request_data[0]["parts"]:
        if part.get("toolCallId") == "call-1":
            part["state"] = "approval-responded"
            part["approval"] = {"id": "approve_call-1", "approved": True}

    prepared = chat_router.prepare_chat_request(
        request_messages=[UIMessage.model_validate(request_data[0])],
        stored_messages=stored_messages,
    )

    assert prepared.has_work is True
    assert prepared.changed is False
    assert prepared.messages is stored_messages
    assert [(a.hook_id, a.granted) for a in prepared.approvals] == [
        ("approve_call-1", True)
    ]


def test_framework_renders_persisted_pending_approval() -> None:
    messages = [
        ai_messages.Message(
            id="assistant-1",
            turn_id="turn-1",
            role="assistant",
            parts=[
                ai_messages.TextPart(id="text-1", text="Need approval"),
                ai_messages.ToolCallPart(
                    id="call-part-1",
                    tool_call_id="call-1",
                    tool_name="bash",
                    tool_args="{}",
                ),
            ],
        ),
        ai_messages.Message(
            id="tool-1",
            turn_id="turn-1",
            role="tool",
            parts=[
                ai_messages.ToolResultPart(
                    id="result-1",
                    tool_call_id="call-1",
                    tool_name="bash",
                    result="Pending",
                    is_error=True,
                    is_hook_pending=True,
                )
            ],
        ),
        ai_messages.Message(
            id="internal-1",
            turn_id="turn-1",
            role="internal",
            parts=[
                ai_messages.HookPart(
                    id="hook-part-1",
                    hook_id="approve_call-1",
                    hook_type=TOOL_APPROVAL_HOOK_TYPE,
                    status="pending",
                    metadata={"tool": "bash"},
                )
            ],
        ),
    ]

    ui_messages = to_ui_messages(messages)
    tool_part = next(
        part.model_dump(mode="json", by_alias=True)
        for part in ui_messages[0].parts
        if getattr(part, "tool_call_id", None) == "call-1"
    )

    assert tool_part["state"] == "approval-requested"
    assert tool_part["approval"]["id"] == "approve_call-1"
