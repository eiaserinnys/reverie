"""Tests for AdjacencyRepository.

Tests for passage adjacency (Passage↔Passage temporal edges) for MemoryLink v2.
Uses actual PostgreSQL with reverie database.
"""

import pytest

from reverie.db.connection import close_pool, get_connection
from reverie.db.repositories.adjacency_repo import AdjacencyRepository, PassageAdjacency
from reverie.db.repositories.passage_repo import PassageRepository


@pytest.fixture
def conn():
    """Provide a database connection that rolls back after each test."""
    with get_connection() as connection:
        yield connection
    close_pool()


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Clean up test data after each test."""
    yield
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM passage_adjacency")
            cur.execute("DELETE FROM entity_passages")
            cur.execute("DELETE FROM passages")
    close_pool()


class TestAdjacencyRepository:
    """Test AdjacencyRepository CRUD operations."""

    def test_create_adjacency(self, conn):
        """Should create a passage adjacency edge."""
        # Create test passages
        passage_repo = PassageRepository(conn)
        pid_a = passage_repo.insert("sess-adj-1", 0, "Turn 0")
        pid_b = passage_repo.insert("sess-adj-1", 1, "Turn 1")

        # Create adjacency
        adj_repo = AdjacencyRepository(conn)
        adjacency = adj_repo.create(pid_a, pid_b)

        assert adjacency is not None
        assert adjacency.passage_id_a == pid_a
        assert adjacency.passage_id_b == pid_b

    def test_create_adjacency_auto_order(self, conn):
        """Should auto-order passage IDs (smaller first)."""
        passage_repo = PassageRepository(conn)
        pid_a = passage_repo.insert("sess-adj-2", 0, "Turn 0")
        pid_b = passage_repo.insert("sess-adj-2", 1, "Turn 1")

        # Create with b, a order (reversed)
        adj_repo = AdjacencyRepository(conn)
        adjacency = adj_repo.create(pid_b, pid_a)

        # Should auto-order: a < b
        assert adjacency.passage_id_a == min(pid_a, pid_b)
        assert adjacency.passage_id_b == max(pid_a, pid_b)

    def test_create_adjacency_idempotent(self, conn):
        """Should not fail on duplicate adjacency creation."""
        passage_repo = PassageRepository(conn)
        pid_a = passage_repo.insert("sess-adj-3", 0, "Turn 0")
        pid_b = passage_repo.insert("sess-adj-3", 1, "Turn 1")

        adj_repo = AdjacencyRepository(conn)
        adj1 = adj_repo.create(pid_a, pid_b)
        adj2 = adj_repo.create(pid_a, pid_b)  # Duplicate

        # Should return same adjacency (idempotent)
        assert adj1.passage_id_a == adj2.passage_id_a
        assert adj1.passage_id_b == adj2.passage_id_b

    def test_get_adjacent_passages(self, conn):
        """Should get all adjacent passages for a given passage."""
        passage_repo = PassageRepository(conn)
        pid_0 = passage_repo.insert("sess-adj-4", 0, "Turn 0")
        pid_1 = passage_repo.insert("sess-adj-4", 1, "Turn 1")
        pid_2 = passage_repo.insert("sess-adj-4", 2, "Turn 2")
        pid_3 = passage_repo.insert("sess-adj-4", 3, "Turn 3")

        # Create adjacencies: 0-1, 1-2, 2-3
        adj_repo = AdjacencyRepository(conn)
        adj_repo.create(pid_0, pid_1)
        adj_repo.create(pid_1, pid_2)
        adj_repo.create(pid_2, pid_3)

        # Get adjacent to pid_1 (should return pid_0 and pid_2)
        adjacent = adj_repo.get_adjacent_passages(pid_1)
        assert len(adjacent) == 2
        assert pid_0 in adjacent
        assert pid_2 in adjacent

        # Get adjacent to pid_0 (should return only pid_1)
        adjacent_0 = adj_repo.get_adjacent_passages(pid_0)
        assert len(adjacent_0) == 1
        assert pid_1 in adjacent_0

    def test_get_all(self, conn):
        """Should return all adjacency records."""
        passage_repo = PassageRepository(conn)
        pid_0 = passage_repo.insert("sess-adj-5", 0, "Turn 0")
        pid_1 = passage_repo.insert("sess-adj-5", 1, "Turn 1")
        pid_2 = passage_repo.insert("sess-adj-5", 2, "Turn 2")

        adj_repo = AdjacencyRepository(conn)
        adj_repo.create(pid_0, pid_1)
        adj_repo.create(pid_1, pid_2)

        all_adjacencies = adj_repo.get_all()
        assert len(all_adjacencies) == 2

    def test_delete_by_passage(self, conn):
        """Should delete all adjacencies involving a specific passage."""
        passage_repo = PassageRepository(conn)
        pid_0 = passage_repo.insert("sess-adj-6", 0, "Turn 0")
        pid_1 = passage_repo.insert("sess-adj-6", 1, "Turn 1")
        pid_2 = passage_repo.insert("sess-adj-6", 2, "Turn 2")

        adj_repo = AdjacencyRepository(conn)
        adj_repo.create(pid_0, pid_1)
        adj_repo.create(pid_1, pid_2)

        # Delete adjacencies involving pid_1
        deleted = adj_repo.delete_by_passage(pid_1)
        assert deleted == 2

        # Verify all adjacencies are gone
        all_adjacencies = adj_repo.get_all()
        assert len(all_adjacencies) == 0

    def test_delete_all(self, conn):
        """Should delete all adjacency records."""
        passage_repo = PassageRepository(conn)
        pid_0 = passage_repo.insert("sess-adj-7", 0, "Turn 0")
        pid_1 = passage_repo.insert("sess-adj-7", 1, "Turn 1")

        adj_repo = AdjacencyRepository(conn)
        adj_repo.create(pid_0, pid_1)

        deleted = adj_repo.delete_all()
        assert deleted == 1

        all_adjacencies = adj_repo.get_all()
        assert len(all_adjacencies) == 0

    def test_bulk_create(self, conn):
        """Should efficiently create multiple adjacencies at once."""
        passage_repo = PassageRepository(conn)
        pids = [
            passage_repo.insert("sess-adj-8", i, f"Turn {i}")
            for i in range(5)
        ]

        # Create adjacency pairs: (0,1), (1,2), (2,3), (3,4)
        pairs = [(pids[i], pids[i + 1]) for i in range(4)]

        adj_repo = AdjacencyRepository(conn)
        created = adj_repo.bulk_create(pairs)

        assert created == 4
        all_adjacencies = adj_repo.get_all()
        assert len(all_adjacencies) == 4

    def test_bulk_create_with_duplicates(self, conn):
        """Should handle duplicates in bulk create gracefully."""
        passage_repo = PassageRepository(conn)
        pid_0 = passage_repo.insert("sess-adj-9", 0, "Turn 0")
        pid_1 = passage_repo.insert("sess-adj-9", 1, "Turn 1")

        # Include duplicate pair
        pairs = [(pid_0, pid_1), (pid_0, pid_1)]

        adj_repo = AdjacencyRepository(conn)
        created = adj_repo.bulk_create(pairs)

        # Only 1 unique pair should be created
        all_adjacencies = adj_repo.get_all()
        assert len(all_adjacencies) == 1

    def test_create_for_session_within_window(self, conn):
        """Should create adjacencies for passages within window."""
        passage_repo = PassageRepository(conn)
        # Create passages for session
        pids = [
            passage_repo.insert("sess-adj-10", i, f"Turn {i}")
            for i in range(5)
        ]

        adj_repo = AdjacencyRepository(conn)
        # Window = 1: adjacent turns only
        created = adj_repo.create_for_session("sess-adj-10", window=1)

        assert created == 4  # 0-1, 1-2, 2-3, 3-4

        # Verify adjacencies
        all_adjacencies = adj_repo.get_all()
        assert len(all_adjacencies) == 4

    def test_create_for_session_larger_window(self, conn):
        """Should create adjacencies for larger window size."""
        passage_repo = PassageRepository(conn)
        pids = [
            passage_repo.insert("sess-adj-11", i, f"Turn {i}")
            for i in range(4)
        ]

        adj_repo = AdjacencyRepository(conn)
        # Window = 2: connect turns within 2 steps
        created = adj_repo.create_for_session("sess-adj-11", window=2)

        # Expected: 0-1, 0-2, 1-2, 1-3, 2-3 = 5 adjacencies
        assert created == 5
