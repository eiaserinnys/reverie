"""Tests for message and passage serialization."""

from datetime import datetime, timezone

from reverie.parser.models import (
    AssistantEntry,
    AssistantMessage,
    Passage,
    TextContent,
    ToolUseContent,
    UserEntry,
    UserMessage,
)
from reverie.parser.serializer import serialize_messages, serialize_passage


def _make_user_text(text: str) -> UserEntry:
    return UserEntry(
        type="user",
        message=UserMessage(role="user", content=text),
        sessionId="test-123",
        timestamp="2026-02-05T10:00:00Z",
        uuid="u1",
        gitBranch="master",
    )


def _make_assistant_text(text: str) -> AssistantEntry:
    return AssistantEntry(
        type="assistant",
        message=AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text=text)],
        ),
        sessionId="test-123",
        timestamp="2026-02-05T10:00:01Z",
        uuid="a1",
    )


def _make_tool_use(name: str, input_data: dict) -> AssistantEntry:
    return AssistantEntry(
        type="assistant",
        message=AssistantMessage(
            role="assistant",
            content=[
                ToolUseContent(type="tool_use", id="toolu_001", name=name, input=input_data)
            ],
        ),
        sessionId="test-123",
        timestamp="2026-02-05T10:00:01Z",
        uuid="a1",
    )


class TestSerializeMessages:
    """Test serialize_messages function."""

    def test_user_text(self):
        messages = [_make_user_text("Hello")]
        text = serialize_messages(messages)
        assert "User: Hello" in text

    def test_assistant_text(self):
        messages = [_make_assistant_text("Response")]
        text = serialize_messages(messages)
        assert "Assistant: Response" in text

    def test_tool_use(self):
        messages = [_make_tool_use("Read", {"file_path": "/test.txt"})]
        text = serialize_messages(messages)
        assert "Tool: Read(" in text
        assert "/test.txt" in text

    def test_conversation(self):
        messages = [
            _make_user_text("Read the file"),
            _make_assistant_text("I'll read it"),
        ]
        text = serialize_messages(messages)
        assert "User: Read the file" in text
        assert "Assistant: I'll read it" in text


class TestSerializePassage:
    """Test serialize_passage function."""

    def test_with_metadata(self):
        passage = Passage(
            session_id="test-123",
            turn_index=0,
            timestamp=datetime(2026, 2, 5, 10, 0, 0, tzinfo=timezone.utc),
            git_branch="master",
            messages=[_make_user_text("Hello")],
            text="User: Hello",
            token_count=3,
        )
        result = serialize_passage(passage)
        assert "[Session: test-123]" in result
        assert "[Turn: 0]" in result
        assert "[Branch: master]" in result
        assert "Timestamp:" in result
        assert "User: Hello" in result

    def test_without_branch(self):
        passage = Passage(
            session_id="test-123",
            turn_index=1,
            timestamp=datetime(2026, 2, 5, 10, 0, 0, tzinfo=timezone.utc),
            git_branch=None,
            messages=[],
            text="test",
            token_count=1,
        )
        result = serialize_passage(passage)
        assert "[Branch:" not in result
