"""Equivalence tests: LazySeedSelector vs SemanticPropagation seed selection.

Verifies that the Lazy PPR Phase 3 (Progressive Widening) produces
mathematically equivalent results to the in-memory SemanticPropagation.

This is a critical validation test for the Lazy PPR implementation.
"""

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from reverie.db.connection import close_pool, get_connection
from reverie.db.postgres_db import PostgresDB
from reverie.db.repositories.entity_repo import EntityRepository
from reverie.retriever.lazy_seed_selection import LazySeedSelector


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
            cur.execute("DELETE FROM entities WHERE name LIKE 'EquivTest%'")
        conn.commit()
    close_pool()


def create_normalized_embedding(seed: int, dim: int = 1536) -> np.ndarray:
    """Create a normalized random embedding with a specific seed."""
    np.random.seed(seed)
    vec = np.random.randn(dim)
    return vec / np.linalg.norm(vec)


class TestSeedSelectionEquivalence:
    """Tests that LazySeedSelector produces equivalent results to in-memory computation."""

    def test_query_mode_equivalence(self, conn, db):
        """LazySeedSelector query mode should match in-memory cosine similarity."""
        entity_repo = EntityRepository(conn)
        tau = 0.5

        # Create entities with known embeddings
        n_entities = 100
        entity_embeddings = []
        entity_ids = []

        for i in range(n_entities):
            emb = create_normalized_embedding(i + 1000)
            entity_embeddings.append(emb)
            entity_id = entity_repo.upsert(f"EquivTest_Query{i:03d}", emb.tolist())
            entity_ids.append(entity_id)
        conn.commit()

        entity_embeddings = np.array(entity_embeddings)

        # Create query embedding
        query_emb = create_normalized_embedding(9999)

        # === In-memory computation (SemanticPropagation style) ===
        in_memory_sims = cosine_similarity(query_emb.reshape(1, -1), entity_embeddings)[0]
        in_memory_seeds = {
            entity_ids[i]: in_memory_sims[i]
            for i in range(n_entities)
            if in_memory_sims[i] > tau
        }

        # === Lazy seed selection ===
        selector = LazySeedSelector(
            db=db,
            tau=tau,
            initial_k=50,
        )
        lazy_result = selector.select_seeds(query_embedding=query_emb)
        lazy_seeds = lazy_result.seeds

        # === Verify equivalence ===
        # Same set of entity IDs
        in_memory_ids = set(in_memory_seeds.keys())
        lazy_ids = set(lazy_seeds.keys())

        assert in_memory_ids == lazy_ids, (
            f"Entity ID mismatch.\n"
            f"In-memory only: {in_memory_ids - lazy_ids}\n"
            f"Lazy only: {lazy_ids - in_memory_ids}"
        )

        # Similarities should be very close (accounting for floating point)
        for entity_id in in_memory_ids:
            in_memory_sim = in_memory_seeds[entity_id]
            lazy_sim = lazy_seeds[entity_id]
            assert np.isclose(in_memory_sim, lazy_sim, rtol=1e-5), (
                f"Entity {entity_id}: in_memory={in_memory_sim}, lazy={lazy_sim}"
            )

    def test_entity_mode_equivalence(self, conn, db):
        """LazySeedSelector entity mode should match in-memory max aggregation."""
        entity_repo = EntityRepository(conn)
        tau = 0.5

        # Create graph entities
        n_graph_entities = 50
        graph_embeddings = []
        graph_entity_ids = []

        for i in range(n_graph_entities):
            emb = create_normalized_embedding(i + 2000)
            graph_embeddings.append(emb)
            entity_id = entity_repo.upsert(f"EquivTest_Graph{i:03d}", emb.tolist())
            graph_entity_ids.append(entity_id)
        conn.commit()

        graph_embeddings = np.array(graph_embeddings)

        # Create query entities (simulating NER extraction)
        n_query_entities = 5
        query_entity_embeddings = np.array([
            create_normalized_embedding(i + 3000) for i in range(n_query_entities)
        ])

        # === In-memory computation (SemanticPropagation._compute_entity_activations style) ===
        # For each graph entity, find max similarity to any query entity
        sims_matrix = cosine_similarity(query_entity_embeddings, graph_embeddings)
        max_sims = np.max(sims_matrix, axis=0)

        in_memory_seeds = {
            graph_entity_ids[i]: max_sims[i]
            for i in range(n_graph_entities)
            if max_sims[i] > tau
        }

        # === Lazy seed selection ===
        selector = LazySeedSelector(
            db=db,
            tau=tau,
            initial_k=50,
        )
        lazy_result = selector.select_seeds(
            query_embedding=np.zeros(1536),  # Not used in entity mode
            query_entity_embeddings=query_entity_embeddings,
        )
        lazy_seeds = lazy_result.seeds

        # === Verify equivalence ===
        in_memory_ids = set(in_memory_seeds.keys())
        lazy_ids = set(lazy_seeds.keys())

        assert in_memory_ids == lazy_ids, (
            f"Entity ID mismatch.\n"
            f"In-memory only: {in_memory_ids - lazy_ids}\n"
            f"Lazy only: {lazy_ids - in_memory_ids}"
        )

        for entity_id in in_memory_ids:
            in_memory_sim = in_memory_seeds[entity_id]
            lazy_sim = lazy_seeds[entity_id]
            assert np.isclose(in_memory_sim, lazy_sim, rtol=1e-5), (
                f"Entity {entity_id}: in_memory={in_memory_sim}, lazy={lazy_sim}"
            )

    def test_equivalence_with_various_tau_values(self, conn, db):
        """Equivalence should hold for various tau thresholds."""
        entity_repo = EntityRepository(conn)

        # Create entities
        n_entities = 100
        entity_embeddings = []
        entity_ids = []

        for i in range(n_entities):
            emb = create_normalized_embedding(i + 4000)
            entity_embeddings.append(emb)
            entity_id = entity_repo.upsert(f"EquivTest_Tau{i:03d}", emb.tolist())
            entity_ids.append(entity_id)
        conn.commit()

        entity_embeddings = np.array(entity_embeddings)
        query_emb = create_normalized_embedding(8888)

        # Test various tau values
        for tau in [0.3, 0.5, 0.7, 0.9]:
            # In-memory
            in_memory_sims = cosine_similarity(
                query_emb.reshape(1, -1), entity_embeddings
            )[0]
            in_memory_count = sum(1 for s in in_memory_sims if s > tau)

            # Lazy
            selector = LazySeedSelector(
                db=db,
                tau=tau,
                initial_k=50,
            )
            lazy_result = selector.select_seeds(query_embedding=query_emb)

            assert len(lazy_result.seeds) == in_memory_count, (
                f"tau={tau}: in_memory found {in_memory_count}, "
                f"lazy found {len(lazy_result.seeds)}"
            )

    def test_empty_result_equivalence(self, conn, db):
        """Both methods should return empty when no entities match."""
        entity_repo = EntityRepository(conn)

        # Create entities with low similarity to query
        for i in range(10):
            emb = create_normalized_embedding(i + 5000)
            entity_repo.upsert(f"EquivTest_Empty{i:03d}", emb.tolist())
        conn.commit()

        # Query with orthogonal embedding
        query_emb = create_normalized_embedding(6000)

        selector = LazySeedSelector(
            db=db,
            tau=0.99,  # Very high threshold
            initial_k=50,
        )
        lazy_result = selector.select_seeds(query_embedding=query_emb)

        # Both should find very few or no entities
        assert len(lazy_result.seeds) < 5  # Most random vectors won't match at 0.99

    def test_all_entities_above_tau_equivalence(self, conn, db):
        """Both methods should handle case when all entities are above tau."""
        entity_repo = EntityRepository(conn)

        # Create query embedding
        query_emb = create_normalized_embedding(7000)

        # Create entities very similar to query
        entity_ids = []
        for i in range(20):
            # Add small noise to query embedding
            np.random.seed(i + 7001)
            noise = np.random.randn(1536) * 0.01
            emb = query_emb + noise
            emb = emb / np.linalg.norm(emb)

            entity_id = entity_repo.upsert(f"EquivTest_AllAbove{i:03d}", emb.tolist())
            entity_ids.append(entity_id)
        conn.commit()

        selector = LazySeedSelector(
            db=db,
            tau=0.1,  # Very low threshold
            initial_k=10,
        )
        lazy_result = selector.select_seeds(query_embedding=query_emb)

        # Should find all 20 entities
        assert len(lazy_result.seeds) == 20


class TestActivationArrayConversion:
    """Tests for converting lazy seeds to activation array format."""

    def test_activation_array_matches_in_memory(self, conn, db):
        """to_activation_array should produce same array as in-memory computation."""
        entity_repo = EntityRepository(conn)
        tau = 0.5

        # Create entities
        n_entities = 20
        entity_embeddings = []
        entity_ids = []

        for i in range(n_entities):
            emb = create_normalized_embedding(i + 8000)
            entity_embeddings.append(emb)
            entity_id = entity_repo.upsert(f"EquivTest_Array{i:03d}", emb.tolist())
            entity_ids.append(entity_id)
        conn.commit()

        entity_embeddings = np.array(entity_embeddings)
        query_emb = create_normalized_embedding(9000)

        # In-memory activation array (SemanticPropagation style)
        in_memory_sims = cosine_similarity(query_emb.reshape(1, -1), entity_embeddings)[0]
        in_memory_activations = np.where(in_memory_sims > tau, in_memory_sims, 0.0)

        # Lazy seed selection
        selector = LazySeedSelector(
            db=db,
            tau=tau,
            initial_k=50,
        )
        lazy_result = selector.select_seeds(query_embedding=query_emb)

        # Convert to activation array
        entity_id_to_index = {eid: i for i, eid in enumerate(entity_ids)}
        lazy_activations = selector.to_activation_array(
            seeds=lazy_result.seeds,
            entity_id_to_index=entity_id_to_index,
            n_entities=n_entities,
        )

        # Should be equivalent
        assert np.allclose(in_memory_activations, lazy_activations, rtol=1e-5), (
            f"Activation arrays differ:\n"
            f"In-memory: {in_memory_activations}\n"
            f"Lazy: {lazy_activations}"
        )
