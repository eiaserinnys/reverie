"""Tests for PassageProcessingHistoryRepository."""

import psycopg2
import pytest

from reverie.config import get_settings
from reverie.db.repositories import PassageProcessingHistoryRepository


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
    """Get a PassageProcessingHistoryRepository instance."""
    return PassageProcessingHistoryRepository(conn)


@pytest.fixture
def clean_history(conn):
    """Clean up test history after each test."""
    yield
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM passage_processing_history WHERE profile_id = 'test_profile'"
        )
    conn.commit()


class TestPassageProcessingHistoryRepository:
    """Tests for PassageProcessingHistoryRepository."""

    def test_record_indexed(self, repo, conn, clean_history):
        """Test recording an indexed turn."""
        # Record without passage_id (no FK constraint issue)
        record_id = repo.record(
            profile_id="test_profile",
            session_id="test_session",
            turn_index=0,
            action="indexed",
            preprocessed_text="This is the processed text",
            entities_extracted=["entity1", "entity2"],
            # passage_id is optional, skip it to avoid FK constraint
        )
        conn.commit()

        assert record_id is not None
        history = repo.get_by_turn("test_profile", "test_session", 0)
        assert history is not None
        assert history.action == "indexed"
        assert history.passage_id is None  # Not set in this test
        assert history.entities_extracted == ["entity1", "entity2"]

    def test_record_dropped(self, repo, conn, clean_history):
        """Test recording a dropped turn."""
        repo.record(
            profile_id="test_profile",
            session_id="test_session",
            turn_index=1,
            action="dropped",
            drop_reason="Too short",
            raw_text="Hi",
        )
        conn.commit()

        history = repo.get_by_turn("test_profile", "test_session", 1)
        assert history.action == "dropped"
        assert history.drop_reason == "Too short"
        assert history.raw_text == "Hi"

    def test_record_upsert_updates(self, repo, conn, clean_history):
        """Test that recording same turn updates existing record."""
        repo.record(
            profile_id="test_profile",
            session_id="test_session",
            turn_index=2,
            action="failed",
            drop_reason="API error",
        )
        conn.commit()

        # Re-process the same turn (success this time)
        repo.record(
            profile_id="test_profile",
            session_id="test_session",
            turn_index=2,
            action="indexed",
            preprocessed_text="Now successful",
            # Skip passage_id to avoid FK constraint
        )
        conn.commit()

        history = repo.get_by_turn("test_profile", "test_session", 2)
        assert history.action == "indexed"
        assert history.preprocessed_text == "Now successful"
        assert history.drop_reason is None  # Updated to null

    def test_get_by_session(self, repo, conn, clean_history):
        """Test getting all history for a session."""
        for i in range(3):
            repo.record(
                profile_id="test_profile",
                session_id="multi_turn",
                turn_index=i,
                action="indexed" if i % 2 == 0 else "dropped",
            )
        conn.commit()

        history = repo.get_by_session("test_profile", "multi_turn")
        assert len(history) == 3
        assert history[0].turn_index == 0
        assert history[1].turn_index == 1
        assert history[2].turn_index == 2

    def test_get_dropped(self, repo, conn, clean_history):
        """Test getting dropped turns."""
        repo.record(
            profile_id="test_profile",
            session_id="s1",
            turn_index=0,
            action="indexed",
        )
        repo.record(
            profile_id="test_profile",
            session_id="s1",
            turn_index=1,
            action="dropped",
            drop_reason="Too short",
        )
        repo.record(
            profile_id="test_profile",
            session_id="s2",
            turn_index=0,
            action="dropped",
            drop_reason="No content",
        )
        conn.commit()

        dropped = repo.get_dropped("test_profile", limit=10)
        assert len(dropped) == 2
        assert all(h.action == "dropped" for h in dropped)

    def test_get_stats(self, repo, conn, clean_history):
        """Test getting processing statistics."""
        actions = ["indexed", "indexed", "dropped", "skipped", "failed"]
        for i, action in enumerate(actions):
            repo.record(
                profile_id="test_profile",
                session_id=f"stats_{i}",
                turn_index=0,
                action=action,
            )
        conn.commit()

        stats = repo.get_stats("test_profile")
        assert stats["indexed"] == 2
        assert stats["dropped"] == 1
        assert stats["skipped"] == 1
        assert stats["failed"] == 1

    def test_get_drop_reasons(self, repo, conn, clean_history):
        """Test getting drop reason distribution."""
        reasons = ["Too short", "Too short", "No content", "Too short"]
        for i, reason in enumerate(reasons):
            repo.record(
                profile_id="test_profile",
                session_id=f"reason_{i}",
                turn_index=0,
                action="dropped",
                drop_reason=reason,
            )
        conn.commit()

        drop_reasons = repo.get_drop_reasons("test_profile")
        assert len(drop_reasons) == 2
        assert drop_reasons[0] == ("Too short", 3)
        assert drop_reasons[1] == ("No content", 1)

    def test_delete_by_session(self, repo, conn, clean_history):
        """Test deleting history for a session."""
        for i in range(3):
            repo.record(
                profile_id="test_profile",
                session_id="to_delete",
                turn_index=i,
                action="indexed",
            )
        conn.commit()

        deleted = repo.delete_by_session("test_profile", "to_delete")
        conn.commit()

        assert deleted == 3
        history = repo.get_by_session("test_profile", "to_delete")
        assert len(history) == 0
