"""Tests for out_degree maintenance in Lazy PPR.

Tests that out_degree values are correctly updated during:
- Passage insertion with entities
- Adjacency creation
- Deletion operations

Uses actual PostgreSQL with reverie_test database.
"""

import pytest

from reverie.db.connection import close_pool, get_connection
from reverie.db.repositories.adjacency_repo import AdjacencyRepository
from reverie.db.repositories.entity_passage_repo import EntityPassageRepository
from reverie.db.repositories.entity_repo import EntityRepository
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
            cur.execute("DELETE FROM passages WHERE session_id LIKE 'test-outdeg-%'")
            cur.execute("DELETE FROM entities WHERE name LIKE 'TestEntity%'")
        conn.commit()
    close_pool()


class TestPassageOutDegree:
    """Tests for passage out_degree maintenance."""

    def test_passage_insert_with_out_degree(self, conn):
        """Should store initial out_degree when inserting passage."""
        passage_repo = PassageRepository(conn)

        # Insert passage with explicit out_degree
        pid = passage_repo.insert(
            "test-outdeg-1",
            0,
            "Test passage",
            out_degree=5,
        )
        conn.commit()

        passage = passage_repo.get_by_id(pid)
        assert passage.out_degree == 5

    def test_passage_out_degree_default_zero(self, conn):
        """Should default to 0 out_degree."""
        passage_repo = PassageRepository(conn)

        pid = passage_repo.insert("test-outdeg-2", 0, "Test passage")
        conn.commit()

        passage = passage_repo.get_by_id(pid)
        assert passage.out_degree == 0

    def test_adjacency_increments_both_passage_out_degrees(self, conn):
        """Creating adjacency should increment out_degree of both passages."""
        passage_repo = PassageRepository(conn)
        adj_repo = AdjacencyRepository(conn)

        # Create two passages with initial out_degree=0
        pid_a = passage_repo.insert("test-outdeg-3", 0, "Passage A", out_degree=0)
        pid_b = passage_repo.insert("test-outdeg-3", 1, "Passage B", out_degree=0)
        conn.commit()

        # Create adjacency (this should increment both out_degrees)
        adj_repo.create(pid_a, pid_b)

        # Verify both passages have out_degree=1
        p_a = passage_repo.get_by_id(pid_a)
        p_b = passage_repo.get_by_id(pid_b)
        assert p_a.out_degree == 1, f"Expected 1, got {p_a.out_degree}"
        assert p_b.out_degree == 1, f"Expected 1, got {p_b.out_degree}"

    def test_duplicate_adjacency_no_double_increment(self, conn):
        """Duplicate adjacency creation should not double-increment out_degree."""
        passage_repo = PassageRepository(conn)
        adj_repo = AdjacencyRepository(conn)

        pid_a = passage_repo.insert("test-outdeg-4", 0, "Passage A", out_degree=0)
        pid_b = passage_repo.insert("test-outdeg-4", 1, "Passage B", out_degree=0)
        conn.commit()

        # Create adjacency twice
        adj_repo.create(pid_a, pid_b)
        adj_repo.create(pid_a, pid_b)  # Duplicate

        # Should still be 1
        p_a = passage_repo.get_by_id(pid_a)
        p_b = passage_repo.get_by_id(pid_b)
        assert p_a.out_degree == 1
        assert p_b.out_degree == 1

    def test_update_out_degree(self, conn):
        """Should update out_degree by delta."""
        passage_repo = PassageRepository(conn)

        pid = passage_repo.insert("test-outdeg-5", 0, "Test passage", out_degree=3)
        conn.commit()

        # Increment by 2
        passage_repo.update_out_degree(pid, 2)
        conn.commit()

        passage = passage_repo.get_by_id(pid)
        assert passage.out_degree == 5

        # Decrement by 1
        passage_repo.update_out_degree(pid, -1)
        conn.commit()

        passage = passage_repo.get_by_id(pid)
        assert passage.out_degree == 4


class TestEntityOutDegree:
    """Tests for entity out_degree maintenance."""

    def test_entity_out_degree_default_zero(self, conn):
        """Entity out_degree should default to 0."""
        entity_repo = EntityRepository(conn)

        entity_id = entity_repo.upsert("TestEntity001")
        conn.commit()

        entity = entity_repo.get_by_id(entity_id)
        assert entity.out_degree == 0

    def test_entity_passage_link_increments_entity_out_degree(self, conn):
        """Linking entity to passage should increment entity out_degree."""
        passage_repo = PassageRepository(conn)
        entity_repo = EntityRepository(conn)

        pid = passage_repo.insert("test-outdeg-6", 0, "Test passage")
        entity_id = entity_repo.upsert("TestEntity002")
        conn.commit()

        # Link entity to passage with manual out_degree update
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO entity_passages (entity_id, passage_id)
                   VALUES (%s, %s) ON CONFLICT DO NOTHING RETURNING entity_id""",
                (entity_id, pid),
            )
            if cur.fetchone() is not None:
                cur.execute(
                    "UPDATE entities SET out_degree = out_degree + 1 WHERE id = %s",
                    (entity_id,),
                )
        conn.commit()

        entity = entity_repo.get_by_id(entity_id)
        assert entity.out_degree == 1

    def test_update_out_degree(self, conn):
        """Should update entity out_degree by delta."""
        entity_repo = EntityRepository(conn)

        entity_id = entity_repo.upsert("TestEntity003")
        conn.commit()

        # Increment by 3
        entity_repo.update_out_degree(entity_id, 3)
        conn.commit()

        entity = entity_repo.get_by_id(entity_id)
        assert entity.out_degree == 3

        # Decrement by 1
        entity_repo.update_out_degree(entity_id, -1)
        conn.commit()

        entity = entity_repo.get_by_id(entity_id)
        assert entity.out_degree == 2


class TestOutDegreeOnDelete:
    """Tests for out_degree updates on deletion."""

    def test_delete_entity_passage_link_decrements_out_degree(self, conn):
        """Deleting entity-passage link should decrement out_degrees."""
        passage_repo = PassageRepository(conn)
        entity_repo = EntityRepository(conn)
        ep_repo = EntityPassageRepository(conn)

        # Setup: passage with out_degree=1, entity with out_degree=1
        pid = passage_repo.insert("test-outdeg-7", 0, "Test passage", out_degree=1)
        entity_id = entity_repo.upsert("TestEntity004")
        conn.commit()

        # Create link
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO entity_passages (entity_id, passage_id) VALUES (%s, %s)",
                (entity_id, pid),
            )
            cur.execute(
                "UPDATE entities SET out_degree = 1 WHERE id = %s",
                (entity_id,),
            )
        conn.commit()

        # Delete link with out_degree update
        ep_repo.delete(entity_id, pid, update_out_degree=True)

        # Verify decremented
        entity = entity_repo.get_by_id(entity_id)
        passage = passage_repo.get_by_id(pid)
        assert entity.out_degree == 0
        assert passage.out_degree == 0

    def test_delete_adjacency_decrements_passage_out_degree(self, conn):
        """Deleting adjacency should decrement out_degree of adjacent passages."""
        passage_repo = PassageRepository(conn)
        adj_repo = AdjacencyRepository(conn)

        # Create passages with adjacency
        pid_a = passage_repo.insert("test-outdeg-8", 0, "Passage A", out_degree=0)
        pid_b = passage_repo.insert("test-outdeg-8", 1, "Passage B", out_degree=0)
        pid_c = passage_repo.insert("test-outdeg-8", 2, "Passage C", out_degree=0)
        conn.commit()

        # Create adjacencies: A-B, B-C
        adj_repo.create(pid_a, pid_b)
        adj_repo.create(pid_b, pid_c)

        # B should have out_degree=2
        p_b = passage_repo.get_by_id(pid_b)
        assert p_b.out_degree == 2

        # Delete all adjacencies involving B
        adj_repo.delete_by_passage(pid_b, update_out_degree=True)

        # A and C should have out_degree decremented
        p_a = passage_repo.get_by_id(pid_a)
        p_c = passage_repo.get_by_id(pid_c)
        assert p_a.out_degree == 0
        assert p_c.out_degree == 0


class TestOutDegreeConsistency:
    """Integration tests for out_degree consistency."""

    def test_full_indexing_scenario(self, conn):
        """Simulate full indexing scenario and verify out_degree consistency."""
        passage_repo = PassageRepository(conn)
        entity_repo = EntityRepository(conn)
        adj_repo = AdjacencyRepository(conn)

        # Simulate indexing 3 turns with 2 entities each
        entities_per_turn = [
            ["TestEntityA1", "TestEntityA2"],
            ["TestEntityB1", "TestEntityA1"],  # A1 shared
            ["TestEntityC1", "TestEntityC2"],
        ]

        prev_pid = None
        for turn_idx, entity_names in enumerate(entities_per_turn):
            # Insert passage with initial out_degree = entity count
            pid = passage_repo.insert(
                f"test-outdeg-full",
                turn_idx,
                f"Turn {turn_idx}",
                out_degree=len(entity_names),
            )
            conn.commit()

            # Link entities and update their out_degree
            for name in entity_names:
                eid = entity_repo.upsert(name)
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO entity_passages (entity_id, passage_id)
                           VALUES (%s, %s) ON CONFLICT DO NOTHING RETURNING entity_id""",
                        (eid, pid),
                    )
                    if cur.fetchone() is not None:
                        cur.execute(
                            "UPDATE entities SET out_degree = out_degree + 1 WHERE id = %s",
                            (eid,),
                        )
                conn.commit()

            # Create adjacency
            if prev_pid is not None:
                adj_repo.create(prev_pid, pid)

            prev_pid = pid

        # Verify out_degree consistency
        # Get all passages
        passages = passage_repo.get_by_session("test-outdeg-full")

        # Turn 0: 2 entities + 1 adjacency (to turn 1) = 3
        assert passages[0].out_degree == 3, f"Turn 0: expected 3, got {passages[0].out_degree}"

        # Turn 1: 2 entities + 2 adjacencies (to turn 0 and 2) = 4
        assert passages[1].out_degree == 4, f"Turn 1: expected 4, got {passages[1].out_degree}"

        # Turn 2: 2 entities + 1 adjacency (to turn 1) = 3
        assert passages[2].out_degree == 3, f"Turn 2: expected 3, got {passages[2].out_degree}"

        # Verify entity out_degrees
        # TestEntityA1 appears in turn 0 and 1 -> out_degree=2
        ea1 = entity_repo.get_by_name("TestEntityA1")
        assert ea1.out_degree == 2, f"TestEntityA1: expected 2, got {ea1.out_degree}"

        # TestEntityA2 appears only in turn 0 -> out_degree=1
        ea2 = entity_repo.get_by_name("TestEntityA2")
        assert ea2.out_degree == 1, f"TestEntityA2: expected 1, got {ea2.out_degree}"
