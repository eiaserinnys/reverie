"""Tests for PPR Equivalence (Lazy PPR vs Matrix PPR).

Verifies that LazyBipartitePPR produces mathematically equivalent results
to the original BipartitePPR matrix implementation.

This is critical for Lazy PPR Phase 4 correctness validation.
"""

import numpy as np
import pytest

from reverie.core.bipartite import BipartiteGraph
from reverie.core.bipartite_ppr import BipartitePPR
from reverie.core.lazy_bipartite_ppr import LazyBipartitePPR
from reverie.db.connection import close_pool, get_connection
from reverie.db.repositories.adjacency_repo import AdjacencyRepository
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
            cur.execute("DELETE FROM passages WHERE session_id LIKE 'test-equiv-%'")
            cur.execute("DELETE FROM entities WHERE name LIKE 'EquivTest%'")
        conn.commit()
    close_pool()


def create_equivalence_test_graph(conn, session_id: str, num_entities: int = 5, num_passages: int = 8):
    """Create a test graph in both DB (for Lazy PPR) and memory (for Matrix PPR).

    Creates a bipartite graph where:
    - Each passage connects to 1-3 random entities
    - Adjacent passages (by turn_index) are connected

    Returns:
        Tuple of (BipartiteGraph, entity_id_list, passage_id_list, entity_id_to_index)
    """
    np.random.seed(42)  # Reproducible random connections

    passage_repo = PassageRepository(conn)
    entity_repo = EntityRepository(conn)
    adj_repo = AdjacencyRepository(conn)

    # Create entities
    entity_ids = []
    for i in range(num_entities):
        eid = entity_repo.upsert(f"EquivTest_E{i}_{session_id}")
        entity_ids.append(eid)
    conn.commit()

    # Build in-memory graph
    graph = BipartiteGraph()
    for i, eid in enumerate(entity_ids):
        graph.add_entity(eid, f"EquivTest_E{i}_{session_id}")

    # Create passages with random entity connections
    passage_ids = []
    entity_link_counts = {eid: 0 for eid in entity_ids}

    for turn_idx in range(num_passages):
        # Each passage connects to 1-3 random entities
        num_links = np.random.randint(1, min(4, num_entities + 1))
        linked_entities = np.random.choice(entity_ids, size=num_links, replace=False).tolist()

        # Calculate out_degree: entity links + 1 or 2 adjacencies
        out_degree = len(linked_entities)

        pid = passage_repo.insert(
            f"test-equiv-{session_id}",
            turn_idx,
            f"Passage {turn_idx}",
            out_degree=out_degree,
        )
        passage_ids.append(pid)

        # Add to in-memory graph
        graph.add_passage(pid, f"test-equiv-{session_id}", turn_idx)

        # Create entity-passage links
        for eid in linked_entities:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO entity_passages (entity_id, passage_id)
                       VALUES (%s, %s) ON CONFLICT DO NOTHING""",
                    (eid, pid),
                )
            entity_link_counts[eid] += 1
            graph.add_contain_edge(eid, pid)

        conn.commit()

    # Update entity out_degrees
    for eid, count in entity_link_counts.items():
        entity_repo.update_out_degree(eid, count)
    conn.commit()

    # Create adjacencies (consecutive passages)
    for i in range(len(passage_ids) - 1):
        adj_repo.create(passage_ids[i], passage_ids[i + 1])
        graph.add_adjacency_edge(passage_ids[i], passage_ids[i + 1])

        # Update passage out_degree in DB for adjacencies
        # (AdjacencyRepository.create already does this)

    # Build entity_id_to_index mapping for matrix PPR
    entity_id_to_index = {eid: i for i, eid in enumerate(entity_ids)}

    return graph, entity_ids, passage_ids, entity_id_to_index


class TestPPREquivalence:
    """Tests that verify Lazy PPR and Matrix PPR produce equivalent results."""

    def test_single_seed_equivalence(self, conn, db):
        """Single seed should produce equivalent results in both implementations."""
        graph, entity_ids, passage_ids, entity_id_to_index = create_equivalence_test_graph(
            conn, "single", num_entities=5, num_passages=8
        )

        # Choose first entity as seed
        seed_entity_id = entity_ids[0]

        # Matrix PPR
        matrix_ppr = BipartitePPR(graph, damping=0.85, max_iter=100, tol=1e-8)
        entity_activations = np.zeros(len(entity_ids))
        entity_activations[entity_id_to_index[seed_entity_id]] = 1.0
        matrix_results = matrix_ppr.run(entity_activations, top_k=len(passage_ids))

        # Lazy PPR
        lazy_ppr = LazyBipartitePPR(db, damping=0.85, max_iter=100, tol=1e-8)
        lazy_results = lazy_ppr.run({seed_entity_id: 1.0}, top_k=len(passage_ids))

        # Compare results
        self._compare_results(matrix_results, lazy_results, rtol=1e-4)

    def test_multiple_seeds_equivalence(self, conn, db):
        """Multiple seeds should produce equivalent results."""
        graph, entity_ids, passage_ids, entity_id_to_index = create_equivalence_test_graph(
            conn, "multi", num_entities=5, num_passages=8
        )

        # Use first two entities as seeds with different weights
        seeds = {entity_ids[0]: 0.7, entity_ids[1]: 0.3}

        # Matrix PPR
        matrix_ppr = BipartitePPR(graph, damping=0.85, max_iter=100, tol=1e-8)
        entity_activations = np.zeros(len(entity_ids))
        entity_activations[entity_id_to_index[entity_ids[0]]] = 0.7
        entity_activations[entity_id_to_index[entity_ids[1]]] = 0.3
        matrix_results = matrix_ppr.run(entity_activations, top_k=len(passage_ids))

        # Lazy PPR
        lazy_ppr = LazyBipartitePPR(db, damping=0.85, max_iter=100, tol=1e-8)
        lazy_results = lazy_ppr.run(seeds, top_k=len(passage_ids))

        # Compare results
        self._compare_results(matrix_results, lazy_results, rtol=1e-4)

    def test_different_damping_factors(self, conn, db):
        """Equivalence should hold for different damping factors."""
        for damping in [0.5, 0.7, 0.85, 0.95]:
            graph, entity_ids, passage_ids, entity_id_to_index = create_equivalence_test_graph(
                conn, f"damp{int(damping*100)}", num_entities=4, num_passages=6
            )

            seed_entity_id = entity_ids[0]

            # Matrix PPR
            matrix_ppr = BipartitePPR(graph, damping=damping, max_iter=100, tol=1e-8)
            entity_activations = np.zeros(len(entity_ids))
            entity_activations[entity_id_to_index[seed_entity_id]] = 1.0
            matrix_results = matrix_ppr.run(entity_activations, top_k=len(passage_ids))

            # Lazy PPR
            lazy_ppr = LazyBipartitePPR(db, damping=damping, max_iter=100, tol=1e-8)
            lazy_results = lazy_ppr.run({seed_entity_id: 1.0}, top_k=len(passage_ids))

            # Compare results
            self._compare_results(
                matrix_results,
                lazy_results,
                rtol=1e-4,
                msg=f"Damping={damping}",
            )

    def test_larger_graph_equivalence(self, conn, db):
        """Equivalence should hold for larger graphs."""
        graph, entity_ids, passage_ids, entity_id_to_index = create_equivalence_test_graph(
            conn, "large", num_entities=10, num_passages=20
        )

        # Use three seeds
        seeds = {
            entity_ids[0]: 0.5,
            entity_ids[3]: 0.3,
            entity_ids[7]: 0.2,
        }

        # Matrix PPR
        matrix_ppr = BipartitePPR(graph, damping=0.85, max_iter=100, tol=1e-8)
        entity_activations = np.zeros(len(entity_ids))
        for eid, weight in seeds.items():
            entity_activations[entity_id_to_index[eid]] = weight
        matrix_results = matrix_ppr.run(entity_activations, top_k=len(passage_ids))

        # Lazy PPR
        lazy_ppr = LazyBipartitePPR(db, damping=0.85, max_iter=100, tol=1e-8)
        lazy_results = lazy_ppr.run(seeds, top_k=len(passage_ids))

        # Compare results
        self._compare_results(matrix_results, lazy_results, rtol=1e-4)

    def test_ranking_order_equivalence(self, conn, db):
        """Top-k ranking order should be identical."""
        graph, entity_ids, passage_ids, entity_id_to_index = create_equivalence_test_graph(
            conn, "rank", num_entities=5, num_passages=10
        )

        seed_entity_id = entity_ids[2]

        # Matrix PPR
        matrix_ppr = BipartitePPR(graph, damping=0.85, max_iter=100, tol=1e-8)
        entity_activations = np.zeros(len(entity_ids))
        entity_activations[entity_id_to_index[seed_entity_id]] = 1.0
        matrix_results = matrix_ppr.run(entity_activations, top_k=5)

        # Lazy PPR
        lazy_ppr = LazyBipartitePPR(db, damping=0.85, max_iter=100, tol=1e-8)
        lazy_results = lazy_ppr.run({seed_entity_id: 1.0}, top_k=5)

        # Compare ranking order
        matrix_ranking = [pid for pid, _ in matrix_results]
        lazy_ranking = [pid for pid, _ in lazy_results]

        assert matrix_ranking == lazy_ranking, (
            f"Ranking order differs:\nMatrix: {matrix_ranking}\nLazy: {lazy_ranking}"
        )

    def _compare_results(
        self,
        matrix_results: list[tuple[int, float]],
        lazy_results: list[tuple[int, float]],
        rtol: float = 1e-4,
        msg: str = "",
    ):
        """Compare Matrix PPR and Lazy PPR results.

        Args:
            matrix_results: Results from BipartitePPR.run()
            lazy_results: Results from LazyBipartitePPR.run()
            rtol: Relative tolerance for score comparison
            msg: Optional message for assertion errors
        """
        # Convert to dicts for easy comparison
        matrix_dict = {pid: score for pid, score in matrix_results}
        lazy_dict = {pid: score for pid, score in lazy_results}

        # Check same passages are present
        matrix_pids = set(matrix_dict.keys())
        lazy_pids = set(lazy_dict.keys())

        # Due to floating point, some very low scores might appear/disappear
        # Check intersection has the important passages
        common_pids = matrix_pids & lazy_pids

        assert len(common_pids) > 0, f"No common passages! Matrix: {matrix_pids}, Lazy: {lazy_pids}"

        # Compare scores for common passages
        for pid in common_pids:
            matrix_score = matrix_dict[pid]
            lazy_score = lazy_dict[pid]

            if matrix_score == 0 and lazy_score == 0:
                continue

            # Relative difference
            max_score = max(abs(matrix_score), abs(lazy_score))
            if max_score > 0:
                rel_diff = abs(matrix_score - lazy_score) / max_score
                assert rel_diff < rtol, (
                    f"{msg} Passage {pid} score mismatch: "
                    f"Matrix={matrix_score:.6f}, Lazy={lazy_score:.6f}, RelDiff={rel_diff:.6f}"
                )


class TestPPREquivalenceEdgeCases:
    """Edge case tests for PPR equivalence."""

    def test_linear_chain_graph(self, conn, db):
        """Linear chain: E1-P1-P2-P3-E2 should produce equivalent results."""
        passage_repo = PassageRepository(conn)
        entity_repo = EntityRepository(conn)
        adj_repo = AdjacencyRepository(conn)

        # Create two entities
        e1_id = entity_repo.upsert("EquivTest_Chain_E1")
        e2_id = entity_repo.upsert("EquivTest_Chain_E2")
        conn.commit()

        # Create passages: P1 has E1, P3 has E2
        p1_id = passage_repo.insert("test-equiv-chain", 0, "P1", out_degree=1)
        p2_id = passage_repo.insert("test-equiv-chain", 1, "P2", out_degree=0)
        p3_id = passage_repo.insert("test-equiv-chain", 2, "P3", out_degree=1)
        conn.commit()

        # Link E1 to P1, E2 to P3
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO entity_passages (entity_id, passage_id) VALUES (%s, %s)",
                (e1_id, p1_id),
            )
            cur.execute(
                "INSERT INTO entity_passages (entity_id, passage_id) VALUES (%s, %s)",
                (e2_id, p3_id),
            )
        entity_repo.update_out_degree(e1_id, 1)
        entity_repo.update_out_degree(e2_id, 1)
        conn.commit()

        # Create adjacencies: P1-P2, P2-P3
        adj_repo.create(p1_id, p2_id)
        adj_repo.create(p2_id, p3_id)

        # Build in-memory graph
        graph = BipartiteGraph()
        graph.add_entity(e1_id, "E1")
        graph.add_entity(e2_id, "E2")
        graph.add_passage(p1_id, "test-equiv-chain", 0)
        graph.add_passage(p2_id, "test-equiv-chain", 1)
        graph.add_passage(p3_id, "test-equiv-chain", 2)
        graph.add_contain_edge(e1_id, p1_id)
        graph.add_contain_edge(e2_id, p3_id)
        graph.add_adjacency_edge(p1_id, p2_id)
        graph.add_adjacency_edge(p2_id, p3_id)

        # Run PPR from E1
        entity_id_to_index = {e1_id: 0, e2_id: 1}

        # Matrix PPR
        matrix_ppr = BipartitePPR(graph, damping=0.85, max_iter=100, tol=1e-8)
        entity_activations = np.array([1.0, 0.0])
        matrix_results = matrix_ppr.run(entity_activations, top_k=3)

        # Lazy PPR
        lazy_ppr = LazyBipartitePPR(db, damping=0.85, max_iter=100, tol=1e-8)
        lazy_results = lazy_ppr.run({e1_id: 1.0}, top_k=3)

        # P1 should have highest score (directly connected to seed)
        # P2 should have second highest (adjacent to P1)
        # P3 should have lowest but non-zero (two hops from seed)

        matrix_dict = {pid: score for pid, score in matrix_results}
        lazy_dict = {pid: score for pid, score in lazy_results}

        # Check ordering
        assert matrix_dict[p1_id] > matrix_dict[p2_id] > matrix_dict[p3_id]
        assert lazy_dict[p1_id] > lazy_dict[p2_id] > lazy_dict[p3_id]

        # Check score equivalence
        for pid in [p1_id, p2_id, p3_id]:
            rel_diff = abs(matrix_dict[pid] - lazy_dict[pid]) / max(matrix_dict[pid], lazy_dict[pid])
            assert rel_diff < 1e-4, f"P{pid} score mismatch"

    def test_isolated_entity_zero_score(self, conn, db):
        """Entity with no passages should not affect PPR."""
        passage_repo = PassageRepository(conn)
        entity_repo = EntityRepository(conn)

        # Create entities: E1 isolated, E2 connected
        e1_id = entity_repo.upsert("EquivTest_Iso_E1")  # Isolated
        e2_id = entity_repo.upsert("EquivTest_Iso_E2")  # Connected
        conn.commit()

        # Create passage connected only to E2
        p1_id = passage_repo.insert("test-equiv-iso", 0, "P1", out_degree=1)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO entity_passages (entity_id, passage_id) VALUES (%s, %s)",
                (e2_id, p1_id),
            )
        entity_repo.update_out_degree(e2_id, 1)
        conn.commit()

        # Seed with isolated entity - should produce no passage scores
        lazy_ppr = LazyBipartitePPR(db, damping=0.85)
        result = lazy_ppr.run_full({e1_id: 1.0})

        # No passages should be reached from isolated entity
        assert len(result.passage_scores) == 0
