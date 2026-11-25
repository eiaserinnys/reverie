"""Tests for JSONL data models."""

from datetime import datetime, timezone

from reverie.parser.models import (
    AssistantEntry,
    AssistantMessage,
    Passage,
    QueueOperation,
    TextContent,
    ToolResultContent,
    ToolUseContent,
    UserEntry,
    UserMessage,
)


class TestQueueOperation:
    """Test QueueOperation model."""

    def test_parse_valid(self):
        data = {
            "type": "queue-operation",
            "operation": "dequeue",
            "timestamp": "2026-02-05T10:00:00Z",
            "sessionId": "test-123",
        }
        op = QueueOperation.model_validate(data)
        assert op.operation == "dequeue"
        assert op.sessionId == "test-123"

    def test_extra_fields_ignored(self):
        data = {
            "type": "queue-operation",
            "operation": "dequeue",
            "timestamp": "2026-02-05T10:00:00Z",
            "sessionId": "test-123",
            "extraField": "should be ignored",
        }
        op = QueueOperation.model_validate(data)
        assert op.operation == "dequeue"


class TestUserEntry:
    """Test UserEntry model."""

    def test_text_content(self):
        data = {
            "type": "user",
            "message": {"role": "user", "content": "Hello"},
            "sessionId": "test-123",
            "timestamp": "2026-02-05T10:00:00Z",
            "uuid": "u-001",
            "gitBranch": "master",
        }
        entry = UserEntry.model_validate(data)
        assert isinstance(entry.message.content, str)
        assert entry.message.content == "Hello"
        assert entry.gitBranch == "master"

    def test_tool_result_content(self):
        data = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_001",
                        "content": "file contents here",
                        "is_error": False,
                    }
                ],
            },
            "sessionId": "test-123",
            "timestamp": "2026-02-05T10:00:00Z",
            "uuid": "u-002",
        }
        entry = UserEntry.model_validate(data)
        assert isinstance(entry.message.content, list)
        assert len(entry.message.content) == 1
        assert entry.message.content[0].tool_use_id == "toolu_001"

    def test_optional_fields(self):
        data = {
            "type": "user",
            "message": {"role": "user", "content": "Hi"},
            "sessionId": "test-123",
            "timestamp": "2026-02-05T10:00:00Z",
            "uuid": "u-001",
        }
        entry = UserEntry.model_validate(data)
        assert entry.gitBranch is None
        assert entry.parentUuid is None


class TestAssistantEntry:
    """Test AssistantEntry model."""

    def test_text_content(self):
        data = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}],
            },
            "sessionId": "test-123",
            "timestamp": "2026-02-05T10:00:00Z",
            "uuid": "a-001",
        }
        entry = AssistantEntry.model_validate(data)
        assert len(entry.message.content) == 1
        assert isinstance(entry.message.content[0], TextContent)
        assert entry.message.content[0].text == "Hello!"

    def test_tool_use_content(self):
        data = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_001",
                        "name": "Read",
                        "input": {"file_path": "/test.txt"},
                    }
                ],
            },
            "sessionId": "test-123",
            "timestamp": "2026-02-05T10:00:00Z",
            "uuid": "a-001",
        }
        entry = AssistantEntry.model_validate(data)
        assert isinstance(entry.message.content[0], ToolUseContent)
        assert entry.message.content[0].name == "Read"

    def test_mixed_content(self):
        data = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_001",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    },
                ],
            },
            "sessionId": "test-123",
            "timestamp": "2026-02-05T10:00:00Z",
            "uuid": "a-001",
        }
        entry = AssistantEntry.model_validate(data)
        assert len(entry.message.content) == 2
        assert isinstance(entry.message.content[0], TextContent)
        assert isinstance(entry.message.content[1], ToolUseContent)


class TestPassage:
    """Test Passage model."""

    def test_basic(self):
        p = Passage(
            session_id="test-123",
            turn_index=0,
            timestamp=datetime(2026, 2, 5, tzinfo=timezone.utc),
            git_branch="master",
            messages=[],
            text="test text",
            token_count=2,
        )
        assert p.session_id == "test-123"
        assert p.turn_index == 0
        assert p.token_count == 2
