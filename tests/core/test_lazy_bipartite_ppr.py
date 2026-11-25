"""Tests for LazyBipartitePPR (Lazy PPR Phase 4).

Tests the lazy iteration algorithm that doesn't require loading the full graph.
Uses actual PostgreSQL with reverie_test database.
"""

import pytest

from reverie.core.lazy_bipartite_ppr import LazyBipartitePPR, LazyPPRResult
from reverie.db.connection import close_pool, get_connection
from reverie.db.postgres_db import PostgresDB
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
    return PostgresDB(conn)


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Clean up test data after each test."""
    yield
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM passage_adjacency")
            cur.execute("DELETE FROM entity_passages")
            cur.execute("DELETE FROM passages WHERE session_id LIKE 'test-lazyppr-%'")
            cur.execute("DELETE FROM entities WHERE name LIKE 'LazyPPRTest%'")
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

    # Create entities (out_degree will be updated as we add links)
    e1_id = entity_repo.upsert(f"LazyPPRTest_E1_{session_prefix}")
    e2_id = entity_repo.upsert(f"LazyPPRTest_E2_{session_prefix}")
    e3_id = entity_repo.upsert(f"LazyPPRTest_E3_{session_prefix}")
    conn.commit()

    # Create passages
    # P1: connected to E1, E3
    p1_id = passage_repo.insert(
        f"test-lazyppr-{session_prefix}",
        0,
        "Passage 1",
        out_degree=2,  # 2 entities
    )
    # P2: connected to E2, E3
    p2_id = passage_repo.insert(
        f"test-lazyppr-{session_prefix}",
        1,
        "Passage 2",
        out_degree=2,  # 2 entities
    )
    # P3: connected to E3
    p3_id = passage_repo.insert(
        f"test-lazyppr-{session_prefix}",
        2,
        "Passage 3",
        out_degree=1,  # 1 entity
    )
    conn.commit()

    # Create entity-passage links and update entity out_degrees
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
        entity_repo.update_out_degree(eid, count)
    conn.commit()

    # Create adjacencies: P1-P2, P2-P3
    adj_repo.create(p1_id, p2_id)  # This increments P1 and P2 out_degree
    adj_repo.create(p2_id, p3_id)  # This increments P2 and P3 out_degree

    return {
        "entity_ids": {"E1": e1_id, "E2": e2_id, "E3": e3_id},
        "passage_ids": {"P1": p1_id, "P2": p2_id, "P3": p3_id},
    }


class TestLazyBipartitePPRBasic:
    """Basic functionality tests for LazyBipartitePPR."""

    def test_empty_seeds_returns_empty_result(self, conn, db):
        """Should return empty result for empty seeds."""
        ppr = LazyBipartitePPR(db)
        result = ppr.run_full({})

        assert result.entity_scores == {}
        assert result.passage_scores == {}
        assert result.iterations == 0
        assert result.converged is True

    def test_zero_activation_returns_empty_result(self, conn, db):
        """Should return empty result when all activations are zero."""
        ppr = LazyBipartitePPR(db)
        result = ppr.run_full({1: 0.0, 2: 0.0})

        assert result.entity_scores == {}
        assert result.passage_scores == {}
        assert result.converged is True

    def test_single_seed_propagates_to_passages(self, conn, db):
        """Single seed entity should propagate to connected passages."""
        graph = create_test_graph(conn, "single")
        e1_id = graph["entity_ids"]["E1"]
        p1_id = graph["passage_ids"]["P1"]

        ppr = LazyBipartitePPR(db, damping=0.85, max_iter=50)
        result = ppr.run_full({e1_id: 1.0})

        # E1 connects to P1, so P1 should have a score
        assert p1_id in result.passage_scores
        assert result.passage_scores[p1_id] > 0

        # Check metadata
        assert result.iterations > 0
        assert isinstance(result.converged, bool)

    def test_multiple_seeds_propagate(self, conn, db):
        """Multiple seeds should both contribute to PPR."""
        graph = create_test_graph(conn, "multi")
        e1_id = graph["entity_ids"]["E1"]
        e2_id = graph["entity_ids"]["E2"]
        p1_id = graph["passage_ids"]["P1"]
        p2_id = graph["passage_ids"]["P2"]

        ppr = LazyBipartitePPR(db, damping=0.85)
        result = ppr.run_full({e1_id: 1.0, e2_id: 1.0})

        # Both P1 and P2 should have scores
        assert p1_id in result.passage_scores
        assert p2_id in result.passage_scores
        assert result.passage_scores[p1_id] > 0
        assert result.passage_scores[p2_id] > 0


class TestLazyBipartitePPRPropagation:
    """Tests for correct propagation behavior."""

    def test_hub_entity_increases_passage_scores(self, conn, db):
        """Entity connected to multiple passages should spread scores."""
        graph = create_test_graph(conn, "hub")
        e3_id = graph["entity_ids"]["E3"]  # E3 connects to P1, P2, P3

        ppr = LazyBipartitePPR(db, damping=0.85)
        result = ppr.run_full({e3_id: 1.0})

        # All three passages should have scores (E3 connects to all)
        for pname in ["P1", "P2", "P3"]:
            pid = graph["passage_ids"][pname]
            assert pid in result.passage_scores, f"{pname} should have a score"
            assert result.passage_scores[pid] > 0, f"{pname} should have positive score"

    def test_adjacent_passages_share_score(self, conn, db):
        """Passage adjacency should allow score propagation between passages."""
        graph = create_test_graph(conn, "adj")
        e1_id = graph["entity_ids"]["E1"]  # E1 only in P1
        p2_id = graph["passage_ids"]["P2"]  # P2 adjacent to P1

        ppr = LazyBipartitePPR(db, damping=0.85, max_iter=50)
        result = ppr.run_full({e1_id: 1.0})

        # P2 should get some score via P1 adjacency
        assert p2_id in result.passage_scores
        assert result.passage_scores[p2_id] > 0

    def test_damping_factor_effect(self, conn, db):
        """Higher damping should spread more to neighbors."""
        graph = create_test_graph(conn, "damp")
        e1_id = graph["entity_ids"]["E1"]
        p1_id = graph["passage_ids"]["P1"]
        p2_id = graph["passage_ids"]["P2"]

        # Low damping (more reset to seed)
        ppr_low = LazyBipartitePPR(db, damping=0.5, max_iter=50)
        result_low = ppr_low.run_full({e1_id: 1.0})

        # High damping (more spread to neighbors)
        ppr_high = LazyBipartitePPR(db, damping=0.9, max_iter=50)
        result_high = ppr_high.run_full({e1_id: 1.0})

        # With higher damping, P2 (which is further from E1) should get more score
        # relative to P1 compared to lower damping
        ratio_low = result_low.passage_scores.get(p2_id, 0) / (
            result_low.passage_scores.get(p1_id, 0.001)
        )
        ratio_high = result_high.passage_scores.get(p2_id, 0) / (
            result_high.passage_scores.get(p1_id, 0.001)
        )

        assert ratio_high > ratio_low, "Higher damping should spread more to distant passages"


class TestLazyBipartitePPRConvergence:
    """Tests for convergence behavior."""

    def test_converges_in_reasonable_iterations(self, conn, db):
        """PPR should converge within max_iter."""
        graph = create_test_graph(conn, "conv")
        e1_id = graph["entity_ids"]["E1"]

        ppr = LazyBipartitePPR(db, damping=0.85, max_iter=100, tol=1e-6)
        result = ppr.run_full({e1_id: 1.0})

        # PPR typically converges in 10-20 iterations for small graphs
        assert result.iterations < 50, f"Too many iterations: {result.iterations}"
        # Should converge (not hit max_iter)
        # Note: with small graphs and proper setup, it should converge

    def test_max_iter_respected(self, conn, db):
        """Should stop at max_iter even if not converged."""
        graph = create_test_graph(conn, "maxiter")
        e1_id = graph["entity_ids"]["E1"]

        # Very tight tolerance to prevent convergence
        ppr = LazyBipartitePPR(db, damping=0.85, max_iter=3, tol=1e-20)
        result = ppr.run_full({e1_id: 1.0})

        assert result.iterations <= 3


class TestLazyBipartitePPRRunMethod:
    """Tests for the run() method (top-k results)."""

    def test_run_returns_top_k_passages(self, conn, db):
        """run() should return top-k passages sorted by score."""
        graph = create_test_graph(conn, "topk")
        e3_id = graph["entity_ids"]["E3"]

        ppr = LazyBipartitePPR(db, damping=0.85)
        results = ppr.run({e3_id: 1.0}, top_k=2)

        # Should return at most 2 passages
        assert len(results) <= 2

        # Should be sorted by score descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_run_returns_passage_id_score_tuples(self, conn, db):
        """run() should return list of (passage_id, score) tuples."""
        graph = create_test_graph(conn, "tuples")
        e1_id = graph["entity_ids"]["E1"]

        ppr = LazyBipartitePPR(db)
        results = ppr.run({e1_id: 1.0}, top_k=10)

        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], int)  # passage_id
            assert isinstance(item[1], float)  # score


class TestLazyBipartitePPRWithGraph:
    """Tests for run_with_graph() visualization support."""

    def test_run_with_graph_returns_all_fields(self, conn, db):
        """run_with_graph() should return passages, nodes, edges, metadata."""
        graph = create_test_graph(conn, "graph")
        e1_id = graph["entity_ids"]["E1"]

        ppr = LazyBipartitePPR(db, damping=0.85)
        result = ppr.run_with_graph({e1_id: 1.0}, top_k_passages=5)

        assert "passages" in result
        assert "graph_nodes" in result
        assert "graph_edges" in result
        assert "iterations" in result
        assert "converged" in result

    def test_run_with_graph_includes_entities_and_passages(self, conn, db):
        """Graph nodes should include both entities and passages."""
        graph = create_test_graph(conn, "nodes")
        e1_id = graph["entity_ids"]["E1"]

        ppr = LazyBipartitePPR(db, damping=0.85)
        result = ppr.run_with_graph({e1_id: 1.0}, score_threshold=0.0)

        node_types = {node["type"] for node in result["graph_nodes"]}
        assert "entity" in node_types
        assert "passage" in node_types

    def test_run_with_graph_includes_contain_edges(self, conn, db):
        """Graph edges should include entity-passage contain edges."""
        graph = create_test_graph(conn, "edges")
        e1_id = graph["entity_ids"]["E1"]

        ppr = LazyBipartitePPR(db, damping=0.85)
        result = ppr.run_with_graph({e1_id: 1.0}, score_threshold=0.0)

        edge_types = {edge["type"] for edge in result["graph_edges"]}
        assert "contain" in edge_types

    def test_run_with_graph_score_threshold_filters_nodes(self, conn, db):
        """score_threshold should filter out low-score nodes."""
        graph = create_test_graph(conn, "thresh")
        e1_id = graph["entity_ids"]["E1"]

        ppr = LazyBipartitePPR(db, damping=0.85)

        # Low threshold - more nodes
        result_low = ppr.run_with_graph({e1_id: 1.0}, score_threshold=0.0)

        # High threshold - fewer nodes
        result_high = ppr.run_with_graph({e1_id: 1.0}, score_threshold=0.1)

        assert len(result_high["graph_nodes"]) <= len(result_low["graph_nodes"])


class TestPPRNeighborRepository:
    """Tests for the PPR neighbor query repository."""

    def test_get_all_neighbors_empty_inputs(self, conn, db):
        """Should return empty list for empty inputs."""
        from reverie.db.repositories.ppr_neighbor_repo import PPRNeighborRepository

        repo = PPRNeighborRepository(conn)
        result = repo.get_all_neighbors([], [])
        assert result == []

    def test_get_all_neighbors_returns_edges(self, conn, db):
        """Should return edges for active nodes."""
        from reverie.db.repositories.ppr_neighbor_repo import PPRNeighborRepository

        graph = create_test_graph(conn, "neigh")
        e1_id = graph["entity_ids"]["E1"]
        p1_id = graph["passage_ids"]["P1"]

        repo = PPRNeighborRepository(conn)
        edges = repo.get_all_neighbors([e1_id], [p1_id])

        assert len(edges) > 0

        # Check edge structure
        edge_types = {e.edge_type for e in edges}
        assert "e2p" in edge_types  # E1 → P1
        # p2e, p2p edges should also be present since P1 is active
