"""Tests for turn-based chunking."""

from pathlib import Path

from reverie.parser.jsonl_reader import filter_messages, read_jsonl
from reverie.parser.models import (
    AssistantEntry,
    AssistantMessage,
    Passage,
    TextContent,
    ToolUseContent,
    UserEntry,
    UserMessage,
)
from reverie.parser.turn_chunker import _filter_for_indexing, _is_user_text, chunk_by_turns

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_user_text(text: str, session_id: str = "test", uuid: str = "u1") -> UserEntry:
    return UserEntry(
        type="user",
        message=UserMessage(role="user", content=text),
        sessionId=session_id,
        timestamp="2026-02-05T10:00:00Z",
        uuid=uuid,
        gitBranch="master",
    )


def _make_assistant_text(text: str, session_id: str = "test", uuid: str = "a1") -> AssistantEntry:
    return AssistantEntry(
        type="assistant",
        message=AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text=text)],
        ),
        sessionId=session_id,
        timestamp="2026-02-05T10:00:01Z",
        uuid=uuid,
    )


def _make_tool_result(content: str, session_id: str = "test", uuid: str = "u2") -> UserEntry:
    return UserEntry(
        type="user",
        message=UserMessage(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_001",
                    "content": content,
                    "is_error": False,
                }
            ],
        ),
        sessionId=session_id,
        timestamp="2026-02-05T10:00:02Z",
        uuid=uuid,
    )


def _make_tool_use(name: str, session_id: str = "test", uuid: str = "a2") -> AssistantEntry:
    return AssistantEntry(
        type="assistant",
        message=AssistantMessage(
            role="assistant",
            content=[
                ToolUseContent(
                    type="tool_use",
                    id="toolu_001",
                    name=name,
                    input={"command": "ls"},
                )
            ],
        ),
        sessionId=session_id,
        timestamp="2026-02-05T10:00:01Z",
        uuid=uuid,
    )


class TestIsUserText:
    """Test _is_user_text function."""

    def test_user_text_is_boundary(self):
        entry = _make_user_text("Hello")
        assert _is_user_text(entry) is True

    def test_tool_result_is_not_boundary(self):
        entry = _make_tool_result("some result")
        assert _is_user_text(entry) is False

    def test_assistant_is_not_boundary(self):
        entry = _make_assistant_text("response")
        assert _is_user_text(entry) is False


class TestFilterForIndexing:
    """Test _filter_for_indexing function."""

    def test_keeps_user_text(self):
        """User text messages are preserved."""
        messages = [_make_user_text("Hello")]
        filtered = _filter_for_indexing(messages)
        assert len(filtered) == 1
        assert filtered[0].message.content == "Hello"

    def test_keeps_assistant_text(self):
        """Assistant text blocks are preserved."""
        messages = [_make_assistant_text("Response")]
        filtered = _filter_for_indexing(messages)
        assert len(filtered) == 1
        assert filtered[0].message.content[0].text == "Response"

    def test_removes_tool_result(self):
        """Tool result entries are filtered out."""
        messages = [_make_tool_result("some result")]
        filtered = _filter_for_indexing(messages)
        assert len(filtered) == 0

    def test_removes_tool_use(self):
        """Tool use entries are filtered out."""
        messages = [_make_tool_use("Read")]
        filtered = _filter_for_indexing(messages)
        assert len(filtered) == 0

    def test_mixed_messages(self):
        """Mixed messages are properly filtered."""
        messages = [
            _make_user_text("Read file.txt"),
            _make_tool_use("Read"),
            _make_tool_result("file contents"),
            _make_assistant_text("Done"),
        ]
        filtered = _filter_for_indexing(messages)
        assert len(filtered) == 2
        # First should be user text
        assert filtered[0].message.content == "Read file.txt"
        # Second should be assistant text
        assert filtered[1].message.content[0].text == "Done"

    def test_empty_list(self):
        """Empty list returns empty list."""
        filtered = _filter_for_indexing([])
        assert filtered == []

    def test_only_tools_returns_empty(self):
        """Turn with only tool calls returns empty list."""
        messages = [
            _make_tool_use("Read"),
            _make_tool_result("result"),
        ]
        filtered = _filter_for_indexing(messages)
        assert len(filtered) == 0


class TestChunkByTurns:
    """Test chunk_by_turns function."""

    def test_single_turn(self):
        """One user message + one assistant response = one passage."""
        messages = [
            _make_user_text("Hi"),
            _make_assistant_text("Hello!"),
        ]
        passages = list(chunk_by_turns(iter(messages)))
        assert len(passages) == 1
        assert passages[0].turn_index == 0
        assert passages[0].session_id == "test"
        assert "Hi" in passages[0].text
        assert "Hello!" in passages[0].text

    def test_two_turns(self):
        """Two user messages = two passages."""
        messages = [
            _make_user_text("First question", uuid="u1"),
            _make_assistant_text("First answer", uuid="a1"),
            _make_user_text("Second question", uuid="u2"),
            _make_assistant_text("Second answer", uuid="a2"),
        ]
        passages = list(chunk_by_turns(iter(messages)))
        assert len(passages) == 2
        assert passages[0].turn_index == 0
        assert passages[1].turn_index == 1
        assert "First question" in passages[0].text
        assert "Second question" in passages[1].text

    def test_turn_with_tool_calls(self):
        """Turn with tool use/result filters out tool calls, keeps text only."""
        messages = [
            _make_user_text("Read file.txt"),
            _make_tool_use("Read"),
            _make_tool_result("file contents"),
            _make_assistant_text("Done"),
        ]
        passages = list(chunk_by_turns(iter(messages)))
        assert len(passages) == 1
        # User text and assistant text should be present
        assert "Read file.txt" in passages[0].text
        assert "Done" in passages[0].text
        # Tool calls should be filtered out
        assert "Tool:" not in passages[0].text
        assert "Result:" not in passages[0].text

    def test_metadata_preserved(self):
        """Session ID, timestamp, branch should be in passage."""
        messages = [_make_user_text("test")]
        passages = list(chunk_by_turns(iter(messages)))
        assert passages[0].session_id == "test"
        assert passages[0].git_branch == "master"
        assert passages[0].timestamp is not None

    def test_token_count_set(self):
        """Token count should be positive."""
        messages = [
            _make_user_text("Hello"),
            _make_assistant_text("World"),
        ]
        passages = list(chunk_by_turns(iter(messages)))
        assert passages[0].token_count > 0

    def test_empty_messages(self):
        """Empty iterator should yield nothing."""
        passages = list(chunk_by_turns(iter([])))
        assert passages == []

    def test_oversized_turn_split(self):
        """Turn exceeding token limit should be split."""
        # Create a very large turn
        messages = [_make_user_text("Start")]
        for i in range(20):
            messages.append(
                _make_assistant_text(
                    f"This is a long response number {i} " * 20,
                    uuid=f"a-{i}",
                )
            )

        # Use a low token limit to force splitting
        passages = list(chunk_by_turns(iter(messages), token_limit=100))
        assert len(passages) > 1

        # All passages should have reasonable token counts
        for p in passages:
            assert p.token_count > 0

    def test_sample_session_fixture(self):
        """Parse sample session fixture end-to-end."""
        entries = read_jsonl(FIXTURES_DIR / "sample_session.jsonl")
        messages = filter_messages(entries)
        passages = list(chunk_by_turns(messages))

        # Sample session has 2 user-text messages = 2 turns
        assert len(passages) == 2
        assert passages[0].turn_index == 0
        assert passages[1].turn_index == 1
        assert passages[0].session_id == "test-session-001"


class TestUserAssistantSplit:
    """Test user_text and assistant_text field extraction."""

    def test_user_text_extracted(self):
        """User text should be extracted to user_text field."""
        messages = [
            _make_user_text("Hello world"),
            _make_assistant_text("Hi there"),
        ]
        passages = list(chunk_by_turns(iter(messages)))
        assert len(passages) == 1
        assert passages[0].user_text == "Hello world"

    def test_assistant_text_extracted(self):
        """Assistant text should be extracted to assistant_text field."""
        messages = [
            _make_user_text("Question?"),
            _make_assistant_text("Answer!"),
        ]
        passages = list(chunk_by_turns(iter(messages)))
        assert len(passages) == 1
        assert passages[0].assistant_text == "Answer!"

    def test_multiple_user_messages_concatenated(self):
        """Multiple user messages in one turn should be concatenated."""
        # This is a rare case but should handle gracefully
        messages = [
            _make_user_text("First"),
            _make_assistant_text("Response"),
        ]
        passages = list(chunk_by_turns(iter(messages)))
        assert "First" in passages[0].user_text

    def test_multiple_assistant_blocks_concatenated(self):
        """Multiple assistant text blocks should be concatenated."""
        # Create assistant entry with multiple text blocks
        entry = AssistantEntry(
            type="assistant",
            message=AssistantMessage(
                role="assistant",
                content=[
                    TextContent(type="text", text="Part 1"),
                    TextContent(type="text", text="Part 2"),
                ],
            ),
            sessionId="test",
            timestamp="2026-02-05T10:00:01Z",
            uuid="a1",
        )
        messages = [_make_user_text("Q"), entry]
        passages = list(chunk_by_turns(iter(messages)))
        assert "Part 1" in passages[0].assistant_text
        assert "Part 2" in passages[0].assistant_text

    def test_tool_calls_excluded_from_split_texts(self):
        """Tool calls should not appear in user_text or assistant_text."""
        messages = [
            _make_user_text("Read file.txt"),
            _make_tool_use("Read"),
            _make_tool_result("file contents"),
            _make_assistant_text("Done reading"),
        ]
        passages = list(chunk_by_turns(iter(messages)))
        assert passages[0].user_text == "Read file.txt"
        assert passages[0].assistant_text == "Done reading"
        # No tool stuff in either field
        assert "file contents" not in passages[0].user_text
        assert "Tool:" not in passages[0].assistant_text

    def test_empty_user_text(self):
        """Passage with no user text should have empty user_text."""
        # Edge case: only assistant response (shouldn't happen normally)
        messages = [_make_assistant_text("Response")]
        passages = list(chunk_by_turns(iter(messages)))
        assert len(passages) == 1
        assert passages[0].user_text == ""
        assert passages[0].assistant_text == "Response"

    def test_empty_assistant_text(self):
        """Passage with no assistant response should have empty assistant_text."""
        messages = [_make_user_text("Hello")]
        passages = list(chunk_by_turns(iter(messages)))
        assert len(passages) == 1
        assert passages[0].user_text == "Hello"
        assert passages[0].assistant_text == ""

    def test_two_turns_with_split_texts(self):
        """Multiple turns should each have their own user/assistant texts."""
        messages = [
            _make_user_text("Q1", uuid="u1"),
            _make_assistant_text("A1", uuid="a1"),
            _make_user_text("Q2", uuid="u2"),
            _make_assistant_text("A2", uuid="a2"),
        ]
        passages = list(chunk_by_turns(iter(messages)))
        assert len(passages) == 2

        assert passages[0].user_text == "Q1"
        assert passages[0].assistant_text == "A1"
        assert passages[1].user_text == "Q2"
        assert passages[1].assistant_text == "A2"
