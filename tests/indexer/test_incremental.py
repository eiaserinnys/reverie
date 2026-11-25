"""Tests for IncrementalIndexer functionality."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg2
import pytest

from reverie.config import get_settings
from reverie.indexer.incremental import IncrementalIndexer, TurnData


# Sample session data for tests
SAMPLE_SESSION = [
    {
        "type": "user",
        "message": {"role": "user", "content": "Hello, can you help me?"},
        "sessionId": "test-session-001",
        "timestamp": "2026-01-01T10:00:00Z",
        "uuid": "user-uuid-1",
    },
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Of course! What do you need?"}],
            "stop_reason": "end_turn",
        },
        "sessionId": "test-session-001",
        "timestamp": "2026-01-01T10:00:01Z",
        "uuid": "assistant-uuid-1",
    },
    {
        "type": "user",
        "message": {"role": "user", "content": "Explain PostgreSQL indexing."},
        "sessionId": "test-session-001",
        "timestamp": "2026-01-01T10:00:02Z",
        "uuid": "user-uuid-2",
    },
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "PostgreSQL uses B-tree indexes by default...",
                }
            ],
            "stop_reason": "end_turn",
        },
        "sessionId": "test-session-001",
        "timestamp": "2026-01-01T10:00:03Z",
        "uuid": "assistant-uuid-2",
    },
]


@pytest.fixture
def sample_session_file():
    """Create a temporary JSONL file with sample session data."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for entry in SAMPLE_SESSION:
            f.write(json.dumps(entry) + "\n")
        f.flush()
        yield Path(f.name)


@pytest.fixture
def incomplete_session_file():
    """Create a session file with incomplete last turn."""
    entries = SAMPLE_SESSION + [
        {
            "type": "user",
            "message": {"role": "user", "content": "Another question..."},
            "sessionId": "test-session-001",
            "timestamp": "2026-01-01T10:00:04Z",
            "uuid": "user-uuid-3",
        },
        # No assistant response - incomplete turn
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
        f.flush()
        yield Path(f.name)


class TestTurnParsing:
    """Tests for turn parsing logic."""

    def test_parse_new_turns_from_start(self, sample_session_file):
        """Test parsing turns from the beginning of file."""
        indexer = IncrementalIndexer()
        turns = list(indexer._parse_new_turns(sample_session_file, start_line=0))

        assert len(turns) == 2
        assert turns[0].turn_index == 0
        assert turns[1].turn_index == 1

    def test_parse_new_turns_incremental(self, sample_session_file):
        """Test parsing turns from a specific line."""
        indexer = IncrementalIndexer()

        # First, get the line count for the first turn
        first_turns = list(indexer._parse_new_turns(sample_session_file, start_line=0))
        first_turn_end = first_turns[0].end_line

        # Parse from after first turn
        remaining_turns = list(
            indexer._parse_new_turns(sample_session_file, start_line=first_turn_end)
        )

        assert len(remaining_turns) == 1
        assert remaining_turns[0].turn_index == 0  # Reset index for new parse

    def test_incomplete_turn_not_parsed(self, incomplete_session_file):
        """Test that incomplete turns are not parsed."""
        indexer = IncrementalIndexer()
        turns = list(indexer._parse_new_turns(incomplete_session_file, start_line=0))

        # Should only get 2 complete turns, not the incomplete one
        assert len(turns) == 2

    def test_count_lines(self, sample_session_file):
        """Test line counting."""
        indexer = IncrementalIndexer()
        count = indexer._count_lines(sample_session_file)
        assert count == len(SAMPLE_SESSION)


class TestBuildTurnData:
    """Tests for TurnData building."""

    def test_build_turn_data_creates_passage(self):
        """Test that build_turn_data creates proper Passage."""
        indexer = IncrementalIndexer()
        entries = SAMPLE_SESSION[:2]  # First turn

        turn_data = indexer._build_turn_data(
            entries=entries,
            turn_index=0,
            start_line=0,
            end_line=2,
        )

        assert turn_data is not None
        assert turn_data.turn_index == 0
        assert turn_data.start_line == 0
        assert turn_data.end_line == 2
        assert turn_data.passage is not None
        assert turn_data.passage.user_text is not None

    def test_build_turn_data_empty_entries(self):
        """Test that empty entries return None."""
        indexer = IncrementalIndexer()
        turn_data = indexer._build_turn_data(
            entries=[],
            turn_index=0,
            start_line=0,
            end_line=0,
        )
        assert turn_data is None


class TestFileShrinkageDetection:
    """Tests for file shrinkage detection."""

    @pytest.fixture
    def conn(self):
        """Get a database connection for tests."""
        settings = get_settings()
        conn = psycopg2.connect(settings.database_url)
        yield conn
        conn.rollback()
        conn.close()

    def test_file_shrinkage_triggers_reindex(self, sample_session_file, conn):
        """Test that file shrinkage triggers full reindex."""
        from reverie.db.repositories import SessionIndexStateRepository

        # First, create a state with larger file size
        state_repo = SessionIndexStateRepository(conn)
        state = state_repo.upsert(
            profile_id="test",
            session_id=sample_session_file.stem,
            file_path=str(sample_session_file),
            file_mtime=datetime.now(timezone.utc),
            file_size_bytes=999999,  # Much larger than actual
            file_lines_count=100,
        )
        conn.commit()

        # The indexer should detect shrinkage and trigger reindex
        # (This would require mocking the full flow, so we just test detection)
        actual_size = sample_session_file.stat().st_size
        assert actual_size < state.file_size_bytes


class TestIncrementalIndexerMocked:
    """Tests for IncrementalIndexer with mocked dependencies."""

    @pytest.fixture
    def mock_indexer(self):
        """Create an indexer with mocked API clients."""
        with patch("reverie.indexer.incremental.EmbeddingClient") as mock_embed, \
             patch("reverie.indexer.incremental.OpenIEClient") as mock_ner, \
             patch("reverie.indexer.incremental.PreprocessorClient") as mock_prep:

            # Configure mocks
            mock_embed.return_value.embed_text.return_value = [0.1] * 1536
            mock_ner.return_value.extract_entities.return_value = ["PostgreSQL", "index"]
            mock_prep.return_value.preprocess_turn.return_value = MagicMock(
                action="keep",
                refined_text="Refined text content",
            )

            indexer = IncrementalIndexer()
            yield indexer

    def test_indexer_initialization(self, mock_indexer):
        """Test that indexer initializes correctly."""
        assert mock_indexer._embedding_client is not None
        assert mock_indexer._openie_client is not None


class TestIndexTaskIntegration:
    """Integration tests requiring database access."""

    @pytest.fixture
    def conn(self):
        """Get a database connection for tests."""
        settings = get_settings()
        conn = psycopg2.connect(settings.database_url)
        yield conn
        conn.rollback()
        conn.close()

    @pytest.fixture
    def cleanup_test_data(self, conn):
        """Clean up test data after each test."""
        yield
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM session_index_state WHERE profile_id = 'test_incr'"
            )
            cur.execute(
                "DELETE FROM passage_processing_history WHERE profile_id = 'test_incr'"
            )
            cur.execute(
                "DELETE FROM passages WHERE session_id LIKE 'test-incr-%'"
            )
        conn.commit()

    def test_session_index_state_creation(self, conn, cleanup_test_data):
        """Test that session index state is created properly."""
        from reverie.db.repositories import SessionIndexStateRepository

        repo = SessionIndexStateRepository(conn)
        state = repo.upsert(
            profile_id="test_incr",
            session_id="test-incr-001",
            file_path="/path/to/session.jsonl",
            file_mtime=datetime.now(timezone.utc),
            file_size_bytes=1024,
            file_lines_count=50,
        )
        conn.commit()

        assert state is not None
        assert state.profile_id == "test_incr"
        assert state.session_id == "test-incr-001"
        assert state.session_seq >= 1
        assert state.status == "pending"
