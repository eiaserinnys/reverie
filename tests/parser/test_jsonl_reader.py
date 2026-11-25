"""Tests for JSONL file reader."""

from pathlib import Path

import pytest

from reverie.parser.jsonl_reader import filter_messages, read_jsonl
from reverie.parser.models import AssistantEntry, QueueOperation, UserEntry

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestReadJsonl:
    """Test read_jsonl function."""

    def test_read_sample_session(self):
        """Should parse all entries from sample session."""
        entries = list(read_jsonl(FIXTURES_DIR / "sample_session.jsonl"))
        assert len(entries) == 8  # 1 queue-op + 7 messages

        assert isinstance(entries[0], QueueOperation)
        assert entries[0].operation == "dequeue"

        assert isinstance(entries[1], UserEntry)
        assert entries[1].message.content == "claude.md를 읽어봐."

        assert isinstance(entries[2], AssistantEntry)

    def test_empty_file(self, tmp_path):
        """Should return empty iterator for empty file."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")
        entries = list(read_jsonl(empty_file))
        assert entries == []

    def test_invalid_json_skipped(self, tmp_path):
        """Should skip invalid JSON lines without raising."""
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text(
            '{"type":"queue-operation","operation":"dequeue","timestamp":"2026-02-05T10:00:00Z","sessionId":"test"}\n'
            "not valid json\n"
            '{"type":"user","message":{"role":"user","content":"hi"},"sessionId":"test","timestamp":"2026-02-05T10:00:00Z","uuid":"u1"}\n'
        )
        entries = list(read_jsonl(bad_file))
        assert len(entries) == 2

    def test_unknown_type_skipped(self, tmp_path):
        """Should skip entries with unknown type."""
        f = tmp_path / "unknown.jsonl"
        f.write_text(
            '{"type":"unknown-type","data":"test"}\n'
            '{"type":"user","message":{"role":"user","content":"hi"},"sessionId":"test","timestamp":"2026-02-05T10:00:00Z","uuid":"u1"}\n'
        )
        entries = list(read_jsonl(f))
        assert len(entries) == 1
        assert isinstance(entries[0], UserEntry)

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            list(read_jsonl(Path("/nonexistent/file.jsonl")))


class TestFilterMessages:
    """Test filter_messages function."""

    def test_filters_queue_operations(self):
        entries = list(read_jsonl(FIXTURES_DIR / "sample_session.jsonl"))
        messages = list(filter_messages(iter(entries)))

        # Should have no QueueOperation entries
        for msg in messages:
            assert not isinstance(msg, QueueOperation)

        # Should only have UserEntry and AssistantEntry
        assert len(messages) == 7

    def test_preserves_order(self):
        entries = list(read_jsonl(FIXTURES_DIR / "sample_session.jsonl"))
        messages = list(filter_messages(iter(entries)))

        # First message should be user text
        assert isinstance(messages[0], UserEntry)
        assert messages[0].message.content == "claude.md를 읽어봐."
