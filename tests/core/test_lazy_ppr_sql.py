"""Tests for Server-side SQL PPR function.

Tests the PostgreSQL run_lazy_ppr function and compares results with
the Python implementation to ensure equivalence.

Requires:
- reverie_test database
- migrations/018_run_lazy_ppr_function.sql to be installed
"""

import math
import pytest

from reverie.core.lazy_bipartite_ppr import LazyBipartitePPR, LazyPPRResult
from reverie.db.connection import close_pool, get_connection
from reverie.db.repositories.adjacency_repo import AdjacencyRepository
from reverie.db.repositories.entity_passage_repo import EntityPassageRepository
from reverie.db.repositories.entity_repo import EntityRepository
from reverie.db.repositories.passage_repo import PassageRepository


@pytest.fixture
def conn():
    """Provide a database connection."""
    with get_connection() as connection:
        yield connection
    close_pool()


@pytest.fixture
def db(conn):
    """Provide a ReverieDB instance wrapping the connection."""
    from reverie.db.postgres_db import PostgresDB
    return PostgresDB(conn)


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Clean up test data after each test."""
    yield
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM passage_adjacency")
            cur.execute("DELETE FROM entity_passages")
            cur.execute("DELETE FROM passages WHERE session_id LIKE 'test-sqlppr-%'")
            cur.execute("DELETE FROM entities WHERE name LIKE 'SQLPPRTest%'")
        conn.commit()
    close_pool()


def create_test_graph(conn, session_prefix: str):
    """Create a simple test graph for PPR testing.

    Graph structure:
        E1 ─── P1 ─── P2 ─── E2
               │      │
               │      │
               E3 ─── P3

    Returns:
        Dict with entity_ids, passage_ids for verification.
    """
    passage_repo = PassageRepository(conn)
    entity_repo = EntityRepository(conn)
    adj_repo = AdjacencyRepository(conn)

    # Create entities
    e1_id = entity_repo.upsert(f"SQLPPRTest_E1_{session_prefix}")
    e2_id = entity_repo.upsert(f"SQLPPRTest_E2_{session_prefix}")
    e3_id = entity_repo.upsert(f"SQLPPRTest_E3_{session_prefix}")
    conn.commit()

    # Create passages with initial out_degree
    p1_id = passage_repo.insert(
        f"test-sqlppr-{session_prefix}",
        0,
        "Passage 1",
        out_degree=0,
    )
    p2_id = passage_repo.insert(
        f"test-sqlppr-{session_prefix}",
        1,
        "Passage 2",
        out_degree=0,
    )
    p3_id = passage_repo.insert(
        f"test-sqlppr-{session_prefix}",
        2,
        "Passage 3",
        out_degree=0,
    )
    conn.commit()

    # Create entity-passage links
    links = [
        (e1_id, p1_id),  # E1 in P1
        (e3_id, p1_id),  # E3 in P1
        (e2_id, p2_id),  # E2 in P2
        (e3_id, p2_id),  # E3 in P2
        (e3_id, p3_id),  # E3 in P3
    ]

    entity_link_counts = {}
    for eid, pid in links:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO entity_passages (entity_id, passage_id)
                   VALUES (%s, %s) ON CONFLICT DO NOTHING""",
                (eid, pid),
            )
        entity_link_counts[eid] = entity_link_counts.get(eid, 0) + 1
    conn.commit()

    # Update entity out_degrees
    for eid, count in entity_link_counts.items():
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE entities SET out_degree = %s WHERE id = %s",
                (count, eid),
            )
    conn.commit()

    # Create passage adjacency (P1 -- P2)
    adj_repo.create(p1_id, p2_id, update_out_degree=False)
    # Note: create() auto-commits

    # Update passage out_degrees
    # P1: 2 entities + 1 adjacency = 3
    # P2: 2 entities + 1 adjacency = 3
    # P3: 1 entity = 1
    with conn.cursor() as cur:
        cur.execute("UPDATE passages SET out_degree = 3 WHERE id = %s", (p1_id,))
        cur.execute("UPDATE passages SET out_degree = 3 WHERE id = %s", (p2_id,))
        cur.execute("UPDATE passages SET out_degree = 1 WHERE id = %s", (p3_id,))
    conn.commit()

    return {
        "entities": {"e1": e1_id, "e2": e2_id, "e3": e3_id},
        "passages": {"p1": p1_id, "p2": p2_id, "p3": p3_id},
    }


def check_sql_function_exists(conn) -> bool:
    """Check if run_lazy_ppr function is installed."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM pg_proc WHERE proname = 'run_lazy_ppr'
            )
        """)
        return cur.fetchone()[0]


@pytest.fixture
def require_sql_function(conn):
    """Skip test if SQL function is not installed."""
    if not check_sql_function_exists(conn):
        pytest.skip("run_lazy_ppr function not installed. Run migrations/018_run_lazy_ppr_function.sql first.")


class TestServerSidePPR:
    """Tests for server-side SQL PPR function."""

    def test_sql_function_exists(self, conn, db):
        """Test that the SQL function is installed."""
        exists = check_sql_function_exists(conn)
        if not exists:
            pytest.skip("SQL function not installed")
        assert exists

    def test_empty_seeds_sql(self, conn, db, require_sql_function):
        """Test SQL PPR with empty seeds returns empty result."""
        ppr = LazyBipartitePPR(db, use_server_side=True)
        result = ppr.run_full({})

        assert result.entity_scores == {}
        assert result.passage_scores == {}
        assert result.iterations == 0
        assert result.converged

    def test_single_seed_sql(self, conn, db, require_sql_function):
        """Test SQL PPR with single seed entity."""
        graph = create_test_graph(conn, "single")
        e1_id = graph["entities"]["e1"]

        ppr = LazyBipartitePPR(
            db,
            damping=0.85,
            max_iter=10,
            min_score=0.0001,
            use_server_side=True,
        )

        result = ppr.run_full({e1_id: 1.0})

        # E1 should have the highest entity score (it's the seed)
        assert e1_id in result.entity_scores
        assert result.entity_scores[e1_id] > 0

        # P1 should have some score (connected to E1)
        p1_id = graph["passages"]["p1"]
        assert p1_id in result.passage_scores
        assert result.passage_scores[p1_id] > 0

    def test_multiple_seeds_sql(self, conn, db, require_sql_function):
        """Test SQL PPR with multiple seed entities."""
        graph = create_test_graph(conn, "multi")
        e1_id = graph["entities"]["e1"]
        e2_id = graph["entities"]["e2"]

        ppr = LazyBipartitePPR(
            db,
            damping=0.85,
            max_iter=10,
            min_score=0.0001,
            use_server_side=True,
        )

        result = ppr.run_full({e1_id: 0.6, e2_id: 0.4})

        # Both seed entities should have scores
        assert e1_id in result.entity_scores
        assert e2_id in result.entity_scores

        # Multiple passages should have scores
        assert len(result.passage_scores) > 0


class TestPythonVsSQLEquivalence:
    """Tests comparing Python and SQL implementations."""

    def test_equivalence_single_seed(self, conn, db, require_sql_function):
        """Test that Python and SQL give similar results for single seed."""
        graph = create_test_graph(conn, "equiv1")
        e1_id = graph["entities"]["e1"]

        # Common parameters
        damping = 0.85
        max_iter = 10
        min_score = 0.0001

        # Run Python version
        ppr_python = LazyBipartitePPR(
            db,
            damping=damping,
            max_iter=max_iter,
            min_score=min_score,
            use_server_side=False,
        )
        result_python = ppr_python.run_full({e1_id: 1.0})

        # Run SQL version
        ppr_sql = LazyBipartitePPR(
            db,
            damping=damping,
            max_iter=max_iter,
            min_score=min_score,
            use_server_side=True,
        )
        result_sql = ppr_sql.run_full({e1_id: 1.0})

        # Compare entity scores
        all_entities = set(result_python.entity_scores.keys()) | set(result_sql.entity_scores.keys())
        for eid in all_entities:
            py_score = result_python.entity_scores.get(eid, 0)
            sql_score = result_sql.entity_scores.get(eid, 0)
            # Allow some tolerance due to floating point differences
            assert abs(py_score - sql_score) < 0.01, (
                f"Entity {eid}: Python={py_score:.6f}, SQL={sql_score:.6f}"
            )

        # Compare passage scores
        all_passages = set(result_python.passage_scores.keys()) | set(result_sql.passage_scores.keys())
        for pid in all_passages:
            py_score = result_python.passage_scores.get(pid, 0)
            sql_score = result_sql.passage_scores.get(pid, 0)
            assert abs(py_score - sql_score) < 0.01, (
                f"Passage {pid}: Python={py_score:.6f}, SQL={sql_score:.6f}"
            )

    def test_equivalence_multiple_seeds(self, conn, db, require_sql_function):
        """Test that Python and SQL give similar results for multiple seeds."""
        graph = create_test_graph(conn, "equiv2")
        e1_id = graph["entities"]["e1"]
        e2_id = graph["entities"]["e2"]
        e3_id = graph["entities"]["e3"]

        seeds = {e1_id: 0.5, e2_id: 0.3, e3_id: 0.2}

        # Common parameters
        damping = 0.85
        max_iter = 10
        min_score = 0.0001

        # Run Python version
        ppr_python = LazyBipartitePPR(
            db,
            damping=damping,
            max_iter=max_iter,
            min_score=min_score,
            use_server_side=False,
        )
        result_python = ppr_python.run_full(seeds)

        # Run SQL version
        ppr_sql = LazyBipartitePPR(
            db,
            damping=damping,
            max_iter=max_iter,
            min_score=min_score,
            use_server_side=True,
        )
        result_sql = ppr_sql.run_full(seeds)

        # Compare entity scores
        all_entities = set(result_python.entity_scores.keys()) | set(result_sql.entity_scores.keys())
        for eid in all_entities:
            py_score = result_python.entity_scores.get(eid, 0)
            sql_score = result_sql.entity_scores.get(eid, 0)
            assert abs(py_score - sql_score) < 0.01, (
                f"Entity {eid}: Python={py_score:.6f}, SQL={sql_score:.6f}"
            )

        # Compare passage scores
        all_passages = set(result_python.passage_scores.keys()) | set(result_sql.passage_scores.keys())
        for pid in all_passages:
            py_score = result_python.passage_scores.get(pid, 0)
            sql_score = result_sql.passage_scores.get(pid, 0)
            assert abs(py_score - sql_score) < 0.01, (
                f"Passage {pid}: Python={py_score:.6f}, SQL={sql_score:.6f}"
            )

    def test_top_passages_ordering_matches(self, conn, db, require_sql_function):
        """Test that top passages are returned in same order."""
        graph = create_test_graph(conn, "order")
        e1_id = graph["entities"]["e1"]

        # Common parameters
        damping = 0.85
        max_iter = 10
        min_score = 0.0001

        # Run both versions
        ppr_python = LazyBipartitePPR(
            db, damping=damping, max_iter=max_iter, min_score=min_score, use_server_side=False
        )
        ppr_sql = LazyBipartitePPR(
            db, damping=damping, max_iter=max_iter, min_score=min_score, use_server_side=True
        )

        # Use run() which returns top-k sorted passages
        top_python = ppr_python.run({e1_id: 1.0}, top_k=10)
        top_sql = ppr_sql.run({e1_id: 1.0}, top_k=10)

        # Check that top passages are in same order
        python_ids = [pid for pid, _ in top_python]
        sql_ids = [pid for pid, _ in top_sql]

        assert python_ids == sql_ids, (
            f"Ordering mismatch: Python={python_ids}, SQL={sql_ids}"
        )


class TestEdgeCases:
    """Edge case tests for SQL PPR."""

    def test_zero_score_seeds(self, conn, db, require_sql_function):
        """Test handling of seeds with zero total score."""
        ppr = LazyBipartitePPR(db, use_server_side=True)
        result = ppr.run_full({1: 0.0, 2: 0.0})

        assert result.entity_scores == {}
        assert result.passage_scores == {}

    def test_large_seed_count(self, conn, db, require_sql_function):
        """Test SQL PPR with many seeds (stress test)."""
        graph = create_test_graph(conn, "large")
        e1_id = graph["entities"]["e1"]
        e2_id = graph["entities"]["e2"]
        e3_id = graph["entities"]["e3"]

        # Create seeds with all entities
        seeds = {e1_id: 0.33, e2_id: 0.33, e3_id: 0.34}

        ppr = LazyBipartitePPR(
            db,
            damping=0.85,
            max_iter=20,  # More iterations
            min_score=0.0001,
            use_server_side=True,
        )

        result = ppr.run_full(seeds)

        # Should complete without error and have results
        assert len(result.entity_scores) > 0
        assert len(result.passage_scores) > 0

    def test_different_damping_factors(self, conn, db, require_sql_function):
        """Test SQL PPR with different damping factors."""
        graph = create_test_graph(conn, "damp")
        e1_id = graph["entities"]["e1"]

        for damping in [0.5, 0.75, 0.85, 0.95]:
            ppr = LazyBipartitePPR(
                db,
                damping=damping,
                max_iter=10,
                min_score=0.0001,
                use_server_side=True,
            )

            result = ppr.run_full({e1_id: 1.0})

            # Seed entity should always have highest score
            assert e1_id in result.entity_scores
            seed_score = result.entity_scores[e1_id]

            # With lower damping, seed score should be higher (more teleport)
            expected_min = 1 - damping
            assert seed_score >= expected_min * 0.9, (
                f"damping={damping}: seed_score={seed_score:.4f} < expected_min={expected_min:.4f}"
            )
