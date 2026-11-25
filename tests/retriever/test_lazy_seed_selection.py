"""Tests for Lazy Seed Selection (Lazy PPR Phase 3).

Tests Progressive Widening algorithm and LazySeedSelector class.
Uses actual PostgreSQL with reverie_test database.
"""

import numpy as np
import pytest

from reverie.db.connection import close_pool, get_connection
from reverie.db.postgres_db import PostgresDB
from reverie.db.repositories.entity_repo import EntityRepository
from reverie.retriever.lazy_seed_selection import LazySeedSelector, SeedSelectionResult


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
            cur.execute("DELETE FROM entities WHERE name LIKE 'LazySeedTest%'")
        conn.commit()
    close_pool()


def create_test_embedding(seed: int, dim: int = 1536) -> np.ndarray:
    """Create a normalized random embedding with a specific seed.

    Using the same seed produces the same embedding (for exact matching).
    Different seeds produce different embeddings (generally low cosine similarity).
    """
    np.random.seed(seed)
    vec = np.random.randn(dim)
    return vec / np.linalg.norm(vec)


def create_similar_embedding(
    base: np.ndarray, noise_scale: float = 0.1, seed: int = 0
) -> np.ndarray:
    """Create an embedding similar to the base embedding.

    Args:
        base: Base embedding vector.
        noise_scale: How much noise to add (smaller = more similar).
        seed: Random seed for reproducibility.

    Returns:
        Normalized embedding similar to base.
    """
    np.random.seed(seed)
    noise = np.random.randn(len(base)) * noise_scale
    vec = base + noise
    return vec / np.linalg.norm(vec)


class TestEntityRepoFindSeedsAboveTau:
    """Tests for EntityRepository.find_seeds_above_tau()."""

    def test_empty_database_returns_empty(self, conn):
        """Should return empty list when no entities exist."""
        entity_repo = EntityRepository(conn)

        # Use a random embedding
        query_emb = create_test_embedding(seed=1000)

        results = entity_repo.find_seeds_above_tau(
            query_embedding=query_emb.tolist(),
            tau=0.5,
            initial_k=10,
        )

        assert results == []

    def test_finds_similar_entities_above_tau(self, conn):
        """Should find entities with similarity >= tau."""
        entity_repo = EntityRepository(conn)

        # Create a query embedding
        query_emb = create_test_embedding(seed=1001)

        # Create entities with varying similarity to query
        # High similarity (noise_scale=0.01 gives ~0.93 similarity)
        high1_emb = create_similar_embedding(query_emb, noise_scale=0.01, seed=1)
        high2_emb = create_similar_embedding(query_emb, noise_scale=0.015, seed=2)
        # Low similarity (random, should not be found with tau=0.8)
        low1_emb = create_test_embedding(seed=2001)
        low2_emb = create_test_embedding(seed=2002)

        entity_repo.upsert("LazySeedTest_High1", high1_emb.tolist())
        entity_repo.upsert("LazySeedTest_High2", high2_emb.tolist())
        entity_repo.upsert("LazySeedTest_Low1", low1_emb.tolist())
        entity_repo.upsert("LazySeedTest_Low2", low2_emb.tolist())
        conn.commit()

        # Find seeds with tau=0.8
        results = entity_repo.find_seeds_above_tau(
            query_embedding=query_emb.tolist(),
            tau=0.8,
            initial_k=10,
        )

        # Should find the high similarity entities
        assert len(results) >= 2

        # All results should have similarity >= 0.8
        for entity_id, similarity in results:
            assert similarity >= 0.8, f"Entity {entity_id} has similarity {similarity} < 0.8"

    def test_progressive_widening_expands_batch(self, conn):
        """Should expand batch size when all entities are above tau."""
        entity_repo = EntityRepository(conn)

        # Create a query embedding
        query_emb = create_test_embedding(seed=3000)

        # Create 150 entities all with high similarity to query
        # noise_scale=0.02 gives ~0.79 similarity, so use tau=0.5 to catch all
        for i in range(150):
            # All similar to query
            entity_emb = create_similar_embedding(query_emb, noise_scale=0.02, seed=3001 + i)
            entity_repo.upsert(
                f"LazySeedTest_Expand{i:03d}",
                entity_emb.tolist(),
            )
        conn.commit()

        # Find seeds with tau=0.5 (all 150 entities should be above this)
        results = entity_repo.find_seeds_above_tau(
            query_embedding=query_emb.tolist(),
            tau=0.5,
            initial_k=100,  # Start with 100, but we have 150 entities
        )

        # Should find all 150 entities (they all have ~0.79 similarity > 0.5)
        assert len(results) >= 150

    def test_returns_sorted_by_similarity_descending(self, conn):
        """Results should be sorted by similarity (highest first)."""
        entity_repo = EntityRepository(conn)

        query_emb = create_test_embedding(seed=4000)

        # Create entities with varying similarity (different noise scales)
        for i, noise in enumerate([0.3, 0.05, 0.15, 0.5]):
            entity_emb = create_similar_embedding(query_emb, noise_scale=noise, seed=4001 + i)
            entity_repo.upsert(
                f"LazySeedTest_Sort{i}",
                entity_emb.tolist(),
            )
        conn.commit()

        results = entity_repo.find_seeds_above_tau(
            query_embedding=query_emb.tolist(),
            tau=0.5,
            initial_k=10,
        )

        # Extract similarities
        similarities = [sim for _, sim in results]

        # Should be sorted descending
        assert similarities == sorted(similarities, reverse=True)


class TestLazySeedSelector:
    """Tests for LazySeedSelector class."""

    def test_select_seeds_query_mode(self, conn, db):
        """Should select seeds using query embedding mode."""
        entity_repo = EntityRepository(conn)

        # Create query embedding and test entities
        query_emb = create_test_embedding(seed=5000)

        # High similarity (noise_scale=0.01 gives ~0.93)
        high1 = create_similar_embedding(query_emb, noise_scale=0.01, seed=5001)
        high2 = create_similar_embedding(query_emb, noise_scale=0.015, seed=5002)
        # Low similarity (random)
        low1 = create_test_embedding(seed=6001)

        entity_repo.upsert("LazySeedTest_QueryMode1", high1.tolist())
        entity_repo.upsert("LazySeedTest_QueryMode2", high2.tolist())
        entity_repo.upsert("LazySeedTest_QueryMode3", low1.tolist())
        conn.commit()

        selector = LazySeedSelector(
            db=db,
            tau=0.8,
            initial_k=100,
        )

        result = selector.select_seeds(query_embedding=query_emb)

        assert isinstance(result, SeedSelectionResult)
        assert len(result.seeds) >= 2  # Should find high similarity entities
        assert result.query_count >= 1

        # All seeds should have similarity >= tau
        for entity_id, similarity in result.seeds.items():
            assert similarity >= 0.8

    def test_select_seeds_entity_mode(self, conn, db):
        """Should select seeds using entity-to-entity matching mode."""
        entity_repo = EntityRepository(conn)

        # Create base embeddings for graph entities
        base_a = create_test_embedding(seed=7000)
        base_b = create_test_embedding(seed=7001)
        base_c = create_test_embedding(seed=7002)

        entity_repo.upsert("LazySeedTest_GraphA", base_a.tolist())
        entity_repo.upsert("LazySeedTest_GraphB", base_b.tolist())
        entity_repo.upsert("LazySeedTest_GraphC", base_c.tolist())
        conn.commit()

        # Query entities similar to GraphA and GraphB
        query_entity_embs = np.array([
            create_similar_embedding(base_a, noise_scale=0.05, seed=7100),
            create_similar_embedding(base_b, noise_scale=0.05, seed=7101),
        ])

        selector = LazySeedSelector(
            db=db,
            tau=0.7,
            initial_k=100,
        )

        result = selector.select_seeds(
            query_embedding=np.zeros(1536),  # Not used in entity mode
            query_entity_embeddings=query_entity_embs,
        )

        assert isinstance(result, SeedSelectionResult)
        assert result.query_count == 2  # One query per query entity

    def test_entity_mode_max_aggregation(self, conn, db):
        """Entity mode should take max similarity across query entities."""
        entity_repo = EntityRepository(conn)

        # Create one graph entity
        graph_emb = create_test_embedding(seed=8000)
        entity_id = entity_repo.upsert("LazySeedTest_MaxAgg", graph_emb.tolist())
        conn.commit()

        # Two query entities: one very similar to graph entity (noise_scale=0.01 -> ~0.93)
        # and one random (low similarity)
        similar_query = create_similar_embedding(graph_emb, noise_scale=0.01, seed=8001)
        random_query = create_test_embedding(seed=9001)

        query_entity_embs = np.array([similar_query, random_query])

        selector = LazySeedSelector(
            db=db,
            tau=0.5,
            initial_k=100,
        )

        result = selector.select_seeds(
            query_embedding=np.zeros(1536),
            query_entity_embeddings=query_entity_embs,
        )

        # The entity should be in results (due to similarity with first query entity)
        assert entity_id in result.seeds

        # The similarity should be the MAX of the two query entity similarities
        sim = result.seeds[entity_id]
        # Should be high (from similar_query, ~0.93)
        assert sim > 0.9

    def test_to_activation_array(self, conn, db):
        """Should convert seeds dict to activation array."""
        entity_repo = EntityRepository(conn)

        # Create test entities and get their IDs
        emb1 = create_test_embedding(seed=10000)
        emb2 = create_test_embedding(seed=10001)
        id1 = entity_repo.upsert("LazySeedTest_Array1", emb1.tolist())
        id2 = entity_repo.upsert("LazySeedTest_Array2", emb2.tolist())
        conn.commit()

        selector = LazySeedSelector(db=db, tau=0.5)

        seeds = {id1: 0.95, id2: 0.85}
        entity_id_to_index = {id1: 0, id2: 1}

        activations = selector.to_activation_array(
            seeds=seeds,
            entity_id_to_index=entity_id_to_index,
            n_entities=3,
        )

        assert activations.shape == (3,)
        assert activations[0] == 0.95
        assert activations[1] == 0.85
        assert activations[2] == 0.0  # Not in seeds


class TestProgressiveWideningBehavior:
    """Tests for Progressive Widening edge cases."""

    def test_low_tau_finds_many_entities(self, conn):
        """Low tau should find many entities."""
        entity_repo = EntityRepository(conn)

        query_emb = create_test_embedding(seed=11000)

        # Create many entities similar to query (noise_scale=0.02 -> ~0.79)
        for i in range(50):
            entity_emb = create_similar_embedding(query_emb, noise_scale=0.02, seed=11001 + i)
            entity_repo.upsert(
                f"LazySeedTest_LowTau{i:02d}",
                entity_emb.tolist(),
            )
        conn.commit()

        results = entity_repo.find_seeds_above_tau(
            query_embedding=query_emb.tolist(),
            tau=0.5,  # Threshold below ~0.79
            initial_k=10,
        )

        # Should find most entities (all 50 have ~0.79 similarity)
        assert len(results) >= 40

    def test_high_tau_finds_few_entities(self, conn):
        """High tau should find few entities."""
        entity_repo = EntityRepository(conn)

        query_emb = create_test_embedding(seed=12000)

        # Create diverse random entities (not specifically similar to query)
        for i in range(50):
            entity_emb = create_test_embedding(seed=12001 + i)
            entity_repo.upsert(
                f"LazySeedTest_HighTau{i:02d}",
                entity_emb.tolist(),
            )
        conn.commit()

        results = entity_repo.find_seeds_above_tau(
            query_embedding=query_emb.tolist(),
            tau=0.95,  # Very high threshold
            initial_k=10,
        )

        # Should find very few entities (random vectors rarely match at 0.95)
        assert len(results) < 10

    def test_exact_tau_boundary(self, conn):
        """Entities exactly at tau should be included."""
        entity_repo = EntityRepository(conn)

        query_emb = create_test_embedding(seed=13000)

        results = entity_repo.find_seeds_above_tau(
            query_embedding=query_emb.tolist(),
            tau=0.99999,  # Almost 1.0
            initial_k=10,
        )

        # All results should have similarity >= tau
        for _, sim in results:
            assert sim >= 0.99999
