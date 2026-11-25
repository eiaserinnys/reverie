"""Tests for SessionIndexStateRepository."""

from datetime import datetime, timezone

import psycopg2
import pytest

from reverie.config import get_settings
from reverie.db.repositories import SessionIndexStateRepository


@pytest.fixture
def conn():
    """Get a database connection for tests."""
    settings = get_settings()
    conn = psycopg2.connect(settings.database_url)
    yield conn
    conn.rollback()
    conn.close()


@pytest.fixture
def repo(conn):
    """Get a SessionIndexStateRepository instance."""
    return SessionIndexStateRepository(conn)


@pytest.fixture
def clean_session_state(conn):
    """Clean up test session states before and after each test."""
    # Clean before test (in case previous run left data)
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM session_index_state WHERE profile_id = 'test_profile'"
        )
    conn.commit()
    yield
    # Clean after test
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM session_index_state WHERE profile_id = 'test_profile'"
        )
    conn.commit()


class TestSessionIndexStateRepository:
    """Tests for SessionIndexStateRepository."""

    def test_upsert_creates_new_state(self, repo, conn, clean_session_state):
        """Test that upsert creates a new state record."""
        now = datetime.now(timezone.utc)
        state = repo.upsert(
            profile_id="test_profile",
            session_id="test_session_001",
            file_path="/path/to/session.jsonl",
            file_mtime=now,
            file_size_bytes=1024,
            file_lines_count=100,
        )

        assert state is not None
        assert state.profile_id == "test_profile"
        assert state.session_id == "test_session_001"
        assert state.session_seq == 1  # First session for this profile
        assert state.file_path == "/path/to/session.jsonl"
        assert state.file_size_bytes == 1024
        assert state.status == "pending"
        conn.commit()

    def test_upsert_auto_increments_session_seq(self, repo, conn, clean_session_state):
        """Test that session_seq auto-increments for each new session."""
        now = datetime.now(timezone.utc)

        # Create first session
        state1 = repo.upsert(
            profile_id="test_profile",
            session_id="session_001",
            file_path="/path/1.jsonl",
            file_mtime=now,
            file_size_bytes=100,
        )
        conn.commit()

        # Create second session
        state2 = repo.upsert(
            profile_id="test_profile",
            session_id="session_002",
            file_path="/path/2.jsonl",
            file_mtime=now,
            file_size_bytes=200,
        )
        conn.commit()

        assert state1.session_seq == 1
        assert state2.session_seq == 2

    def test_upsert_updates_existing(self, repo, conn, clean_session_state):
        """Test that upsert updates existing state."""
        now = datetime.now(timezone.utc)

        # Create initial state
        state1 = repo.upsert(
            profile_id="test_profile",
            session_id="test_session",
            file_path="/path/v1.jsonl",
            file_mtime=now,
            file_size_bytes=100,
        )
        conn.commit()

        # Update with new file info
        state2 = repo.upsert(
            profile_id="test_profile",
            session_id="test_session",
            file_path="/path/v2.jsonl",
            file_mtime=now,
            file_size_bytes=200,
        )
        conn.commit()

        assert state1.id == state2.id  # Same record
        assert state1.session_seq == state2.session_seq  # Seq preserved
        assert state2.file_path == "/path/v2.jsonl"
        assert state2.file_size_bytes == 200

    def test_get_by_session(self, repo, conn, clean_session_state):
        """Test getting state by profile and session."""
        now = datetime.now(timezone.utc)
        repo.upsert(
            profile_id="test_profile",
            session_id="find_me",
            file_path="/path/find.jsonl",
            file_mtime=now,
            file_size_bytes=100,
        )
        conn.commit()

        state = repo.get_by_session("test_profile", "find_me")
        assert state is not None
        assert state.session_id == "find_me"

        # Non-existent
        assert repo.get_by_session("test_profile", "not_here") is None

    def test_update_progress(self, repo, conn, clean_session_state):
        """Test updating indexing progress."""
        now = datetime.now(timezone.utc)
        repo.upsert(
            profile_id="test_profile",
            session_id="progress_test",
            file_path="/path/p.jsonl",
            file_mtime=now,
            file_size_bytes=100,
        )
        conn.commit()

        repo.update_progress(
            profile_id="test_profile",
            session_id="progress_test",
            last_indexed_turn=5,
            last_indexed_line=100,
            last_passage_id=42,
            passages_indexed_delta=3,
            passages_dropped_delta=1,
        )
        conn.commit()

        state = repo.get_by_session("test_profile", "progress_test")
        assert state.last_indexed_turn == 5
        assert state.last_indexed_line == 100
        assert state.last_passage_id == 42
        assert state.total_passages_indexed == 3
        assert state.total_passages_dropped == 1
        assert state.status == "in_progress"

    def test_mark_completed(self, repo, conn, clean_session_state):
        """Test marking session as completed."""
        now = datetime.now(timezone.utc)
        repo.upsert(
            profile_id="test_profile",
            session_id="complete_test",
            file_path="/path/c.jsonl",
            file_mtime=now,
            file_size_bytes=100,
        )
        conn.commit()

        repo.mark_completed("test_profile", "complete_test")
        conn.commit()

        state = repo.get_by_session("test_profile", "complete_test")
        assert state.status == "completed"
        assert state.error_message is None

    def test_mark_failed(self, repo, conn, clean_session_state):
        """Test marking session as failed."""
        now = datetime.now(timezone.utc)
        repo.upsert(
            profile_id="test_profile",
            session_id="fail_test",
            file_path="/path/f.jsonl",
            file_mtime=now,
            file_size_bytes=100,
        )
        conn.commit()

        repo.mark_failed("test_profile", "fail_test", "Something went wrong")
        conn.commit()

        state = repo.get_by_session("test_profile", "fail_test")
        assert state.status == "failed"
        assert state.error_message == "Something went wrong"

    def test_get_stats(self, repo, conn, clean_session_state):
        """Test getting profile statistics."""
        now = datetime.now(timezone.utc)

        # Create sessions with different statuses
        for i, status in enumerate(["pending", "in_progress", "completed", "failed"]):
            repo.upsert(
                profile_id="test_profile",
                session_id=f"stats_test_{i}",
                file_path=f"/path/{i}.jsonl",
                file_mtime=now,
                file_size_bytes=100,
            )
            conn.commit()
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE session_index_state SET status = %s
                       WHERE session_id = %s AND profile_id = 'test_profile'""",
                    (status, f"stats_test_{i}"),
                )
            conn.commit()

        stats = repo.get_stats("test_profile")
        assert stats["pending"] == 1
        assert stats["in_progress"] == 1
        assert stats["completed"] == 1
        assert stats["failed"] == 1

    def test_reset_for_full_reindex(self, repo, conn, clean_session_state):
        """Test resetting session for full reindex."""
        now = datetime.now(timezone.utc)
        repo.upsert(
            profile_id="test_profile",
            session_id="reset_test",
            file_path="/path/r.jsonl",
            file_mtime=now,
            file_size_bytes=100,
        )
        conn.commit()

        # Set some progress
        repo.update_progress(
            profile_id="test_profile",
            session_id="reset_test",
            last_indexed_turn=10,
            last_indexed_line=200,
            last_passage_id=50,
            passages_indexed_delta=5,
        )
        conn.commit()

        # Reset
        repo.reset_for_full_reindex("test_profile", "reset_test")
        conn.commit()

        state = repo.get_by_session("test_profile", "reset_test")
        assert state.last_indexed_turn == -1
        assert state.last_indexed_line == 0
        assert state.last_passage_id is None
        assert state.total_passages_indexed == 0
        assert state.status == "pending"
        # session_seq should be preserved
        assert state.session_seq is not None

    def test_upsert_with_indexing_path(self, repo, conn, clean_session_state):
        """Test that indexing_path is stored and returned correctly."""
        now = datetime.now(timezone.utc)
        state = repo.upsert(
            profile_id="test_profile",
            session_id="test_session_path",
            file_path="/path/to/session.jsonl",
            file_mtime=now,
            file_size_bytes=1024,
            indexing_path="/home/user/claude-sessions",
        )
        conn.commit()

        assert state.indexing_path == "/home/user/claude-sessions"

        # Retrieve and verify
        fetched = repo.get_by_session(
            "test_profile", "test_session_path", indexing_path="/home/user/claude-sessions"
        )
        assert fetched is not None
        assert fetched.indexing_path == "/home/user/claude-sessions"

    def test_same_session_different_indexing_paths(self, repo, conn, clean_session_state):
        """Test that same session can exist with different indexing_path values."""
        now = datetime.now(timezone.utc)

        # Same session_id but different indexing paths
        state1 = repo.upsert(
            profile_id="test_profile",
            session_id="shared_session",
            file_path="/path/folder1/session.jsonl",
            file_mtime=now,
            file_size_bytes=100,
            indexing_path="/home/user/folder1",
        )
        conn.commit()

        state2 = repo.upsert(
            profile_id="test_profile",
            session_id="shared_session",
            file_path="/path/folder2/session.jsonl",
            file_mtime=now,
            file_size_bytes=200,
            indexing_path="/home/user/folder2",
        )
        conn.commit()

        # Should be different records
        assert state1.id != state2.id
        assert state1.indexing_path == "/home/user/folder1"
        assert state2.indexing_path == "/home/user/folder2"

    def test_indexing_path_default_empty_string(self, repo, conn, clean_session_state):
        """Test that indexing_path defaults to empty string."""
        now = datetime.now(timezone.utc)
        state = repo.upsert(
            profile_id="test_profile",
            session_id="default_path_test",
            file_path="/path/session.jsonl",
            file_mtime=now,
            file_size_bytes=100,
        )
        conn.commit()

        assert state.indexing_path == ""

    def test_get_by_profile_with_indexing_path(self, repo, conn, clean_session_state):
        """Test filtering sessions by indexing_path."""
        now = datetime.now(timezone.utc)

        # Create sessions in different folders
        repo.upsert(
            profile_id="test_profile",
            session_id="session_a",
            file_path="/path/a.jsonl",
            file_mtime=now,
            file_size_bytes=100,
            indexing_path="/folder1",
        )
        repo.upsert(
            profile_id="test_profile",
            session_id="session_b",
            file_path="/path/b.jsonl",
            file_mtime=now,
            file_size_bytes=100,
            indexing_path="/folder1",
        )
        repo.upsert(
            profile_id="test_profile",
            session_id="session_c",
            file_path="/path/c.jsonl",
            file_mtime=now,
            file_size_bytes=100,
            indexing_path="/folder2",
        )
        conn.commit()

        # Get only sessions from folder1
        folder1_sessions = repo.get_by_profile(
            "test_profile", indexing_path="/folder1"
        )
        assert len(folder1_sessions) == 2
        assert all(s.indexing_path == "/folder1" for s in folder1_sessions)

        # Get only sessions from folder2
        folder2_sessions = repo.get_by_profile(
            "test_profile", indexing_path="/folder2"
        )
        assert len(folder2_sessions) == 1
        assert folder2_sessions[0].indexing_path == "/folder2"

    def test_get_stats_with_indexing_path(self, repo, conn, clean_session_state):
        """Test getting stats filtered by indexing_path."""
        now = datetime.now(timezone.utc)

        # Create sessions in folder1
        repo.upsert(
            profile_id="test_profile",
            session_id="folder1_session1",
            file_path="/path/f1s1.jsonl",
            file_mtime=now,
            file_size_bytes=100,
            indexing_path="/folder1",
        )
        repo.upsert(
            profile_id="test_profile",
            session_id="folder1_session2",
            file_path="/path/f1s2.jsonl",
            file_mtime=now,
            file_size_bytes=100,
            indexing_path="/folder1",
        )
        # Create session in folder2
        repo.upsert(
            profile_id="test_profile",
            session_id="folder2_session1",
            file_path="/path/f2s1.jsonl",
            file_mtime=now,
            file_size_bytes=100,
            indexing_path="/folder2",
        )
        conn.commit()

        # Mark one as completed in folder1
        repo.mark_completed(
            "test_profile", "folder1_session1", indexing_path="/folder1"
        )
        conn.commit()

        # Stats for folder1 only
        stats = repo.get_stats("test_profile", indexing_path="/folder1")
        assert stats["pending"] == 1
        assert stats["completed"] == 1

        # Stats for folder2 only
        stats2 = repo.get_stats("test_profile", indexing_path="/folder2")
        assert stats2["pending"] == 1
        assert stats2["completed"] == 0
