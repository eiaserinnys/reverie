"""Integration tests: parse real JSONL session files."""

from pathlib import Path

import pytest

from reverie.parser import parse_session, serialize_passage
from reverie.parser.serializer import serialize_messages

# Test fixture paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_SESSION = FIXTURES_DIR / "sample_session.jsonl"
SESSIONS_DIR = FIXTURES_DIR / "sessions"  # Directory with multiple session files


class TestParseSessionPipeline:
    """End-to-end pipeline tests."""

    def test_parse_sample_fixture(self):
        """Parse sample session fixture through full pipeline."""
        passages = list(parse_session(SAMPLE_SESSION))
        assert len(passages) == 2

        # First turn: "claude.md를 읽어봐." + tool interactions + response
        p0 = passages[0]
        assert p0.session_id == "test-session-001"
        assert p0.turn_index == 0
        assert p0.git_branch == "master"
        assert p0.token_count > 0
        assert "claude.md를 읽어봐" in p0.text

        # Second turn: "좋아. 이제 테스트를 실행해줘."
        p1 = passages[1]
        assert p1.turn_index == 1
        assert "테스트를 실행" in p1.text

    def test_serialize_sample_passages(self):
        """Serialize passages with metadata header."""
        passages = list(parse_session(SAMPLE_SESSION))
        for p in passages:
            text = serialize_passage(p)
            assert "[Session: test-session-001]" in text
            assert f"[Turn: {p.turn_index}]" in text
            assert "Timestamp:" in text

    def test_parse_with_config(self):
        """Pipeline respects config parameters."""
        passages_normal = list(parse_session(SAMPLE_SESSION, token_limit=2000))
        passages_small = list(parse_session(SAMPLE_SESSION, token_limit=50))

        # With small token limit, should produce more passages (splitting)
        assert len(passages_small) >= len(passages_normal)

    def test_parse_multiple_sessions_from_directory(self):
        """Parse multiple session files from a directory."""
        # Find session files sorted by size
        jsonl_files = sorted(
            SESSIONS_DIR.glob("*.jsonl"),
            key=lambda f: f.stat().st_size,
        )
        # Filter out empty files
        jsonl_files = [f for f in jsonl_files if f.stat().st_size > 100]

        assert len(jsonl_files) >= 2, "Expected at least 2 session fixtures"

        # Parse the smallest file (session-small.jsonl)
        smallest_file = jsonl_files[0]
        passages = list(parse_session(smallest_file))

        assert len(passages) > 0
        for p in passages:
            assert p.session_id
            assert p.token_count > 0
            assert p.text

        # Parse the larger file (session-medium.jsonl)
        larger_file = jsonl_files[1]
        passages_medium = list(parse_session(larger_file))

        assert len(passages_medium) >= 2  # Should have multiple turns
        assert passages_medium[0].session_id == "session-medium"


class TestSerializerEdgeCases:
    """Extra serializer tests for coverage."""

    def test_tool_result_with_error(self):
        """Tool result with error flag should show [ERROR] tag."""
        from reverie.parser.models import (
            UserEntry,
            UserMessage,
            ToolResultContent,
        )

        entry = UserEntry(
            type="user",
            message=UserMessage(
                role="user",
                content=[
                    ToolResultContent(
                        type="tool_result",
                        tool_use_id="t1",
                        content="Command failed: exit code 1",
                        is_error=True,
                    )
                ],
            ),
            sessionId="test",
            timestamp="2026-02-05T10:00:00Z",
            uuid="u1",
        )
        text = serialize_messages([entry])
        assert "[ERROR]" in text
        assert "Command failed" in text

    def test_tool_use_with_long_input(self):
        """Tool use with long input should be truncated."""
        from reverie.parser.models import (
            AssistantEntry,
            AssistantMessage,
            ToolUseContent,
        )

        entry = AssistantEntry(
            type="assistant",
            message=AssistantMessage(
                role="assistant",
                content=[
                    ToolUseContent(
                        type="tool_use",
                        id="t1",
                        name="Write",
                        input={
                            "file_path": "/test.txt",
                            "content": "x" * 200,
                        },
                    )
                ],
            ),
            sessionId="test",
            timestamp="2026-02-05T10:00:00Z",
            uuid="a1",
        )
        text = serialize_messages([entry])
        assert "Tool: Write(" in text
        assert "file_path=" in text

    def test_tool_use_with_empty_input(self):
        """Tool use with empty input."""
        from reverie.parser.models import (
            AssistantEntry,
            AssistantMessage,
            ToolUseContent,
        )

        entry = AssistantEntry(
            type="assistant",
            message=AssistantMessage(
                role="assistant",
                content=[
                    ToolUseContent(
                        type="tool_use",
                        id="t1",
                        name="Bash",
                        input={},
                    )
                ],
            ),
            sessionId="test",
            timestamp="2026-02-05T10:00:00Z",
            uuid="a1",
        )
        text = serialize_messages([entry])
        assert "Tool: Bash()" in text

    def test_tool_use_with_non_string_input(self):
        """Tool use with complex non-string input values."""
        from reverie.parser.models import (
            AssistantEntry,
            AssistantMessage,
            ToolUseContent,
        )

        entry = AssistantEntry(
            type="assistant",
            message=AssistantMessage(
                role="assistant",
                content=[
                    ToolUseContent(
                        type="tool_use",
                        id="t1",
                        name="Test",
                        input={
                            "flag": True,
                            "count": 42,
                        },
                    )
                ],
            ),
            sessionId="test",
            timestamp="2026-02-05T10:00:00Z",
            uuid="a1",
        )
        text = serialize_messages([entry])
        assert "Tool: Test(" in text
