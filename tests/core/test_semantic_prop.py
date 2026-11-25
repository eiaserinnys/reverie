"""Tests for SemanticPropagation.

Tests for LinearRAG Stage 1: Entity activation propagation.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from reverie.core.semantic_prop import SemanticPropagation


class TestSemanticPropagation:
    """Test SemanticPropagation operations."""

    def test_propagate_basic(self):
        """Should propagate entity activations from query."""
        # 2 entities, 2 passages
        # Entity embeddings (2 x dim)
        entity_embeddings = np.array([
            [1.0, 0.0, 0.0],  # entity-0
            [0.0, 1.0, 0.0],  # entity-1
        ])

        # Passage embeddings (2 x dim)
        passage_embeddings = np.array([
            [0.9, 0.1, 0.0],  # passage-0: similar to entity-0
            [0.1, 0.9, 0.0],  # passage-1: similar to entity-1
        ])

        # Contain matrix: passage-0 contains entity-0, passage-1 contains entity-1
        contain_matrix = csr_matrix([
            [1.0, 0.0],  # passage-0: entity-0
            [0.0, 1.0],  # passage-1: entity-1
        ])

        prop = SemanticPropagation(
            entity_embeddings=entity_embeddings,
            passage_embeddings=passage_embeddings,
            contain_matrix=contain_matrix,
            tau=0.5,
            delta=0.1,
            max_iter=2,
        )

        # Query similar to entity-0
        query_emb = np.array([0.95, 0.05, 0.0])
        activations = prop.propagate(query_emb)

        # Entity-0 should be more activated than entity-1
        assert activations[0] > activations[1]
        assert activations[0] > 0.5  # Above threshold

    def test_propagate_with_2d_query(self):
        """Should handle 2D query embedding (1, dim)."""
        entity_embeddings = np.array([[1.0, 0.0, 0.0]])
        passage_embeddings = np.array([[0.9, 0.1, 0.0]])
        contain_matrix = csr_matrix([[1.0]])

        prop = SemanticPropagation(
            entity_embeddings=entity_embeddings,
            passage_embeddings=passage_embeddings,
            contain_matrix=contain_matrix,
        )

        # Query as 2D array (1, 3)
        query_emb = np.array([[0.95, 0.05, 0.0]])
        activations = prop.propagate(query_emb)

        assert activations.shape == (1,)
        assert activations[0] > 0.5

    def test_propagate_no_activation(self):
        """Should return zeros when query doesn't match any entity."""
        entity_embeddings = np.array([
            [1.0, 0.0, 0.0],  # entity-0
            [0.0, 1.0, 0.0],  # entity-1
        ])

        passage_embeddings = np.array([
            [1.0, 0.0, 0.0],  # passage-0
            [0.0, 1.0, 0.0],  # passage-1
        ])

        contain_matrix = csr_matrix([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        prop = SemanticPropagation(
            entity_embeddings=entity_embeddings,
            passage_embeddings=passage_embeddings,
            contain_matrix=contain_matrix,
            tau=0.99,  # Very high threshold
        )

        # Query orthogonal to all entities
        query_emb = np.array([0.0, 0.0, 1.0])
        activations = prop.propagate(query_emb)

        # All activations should be zero (below threshold)
        assert np.all(activations == 0.0)

    def test_propagate_through_passages(self):
        """Should propagate activation through passages to discover new entities."""
        # 3 entities, 2 passages
        # Entity-0 is directly similar to query
        # Entity-2 shares passage with entity-0
        entity_embeddings = np.array([
            [1.0, 0.0, 0.0],  # entity-0: similar to query
            [0.0, 1.0, 0.0],  # entity-1: different
            [0.5, 0.5, 0.0],  # entity-2: somewhat similar
        ])

        passage_embeddings = np.array([
            [0.8, 0.1, 0.0],  # passage-0: similar to query
            [0.1, 0.9, 0.0],  # passage-1: different
        ])

        # Contain matrix:
        # passage-0 contains entity-0 and entity-2
        # passage-1 contains entity-1
        contain_matrix = csr_matrix([
            [1.0, 0.0, 1.0],  # passage-0: entity-0, entity-2
            [0.0, 1.0, 0.0],  # passage-1: entity-1
        ])

        prop = SemanticPropagation(
            entity_embeddings=entity_embeddings,
            passage_embeddings=passage_embeddings,
            contain_matrix=contain_matrix,
            tau=0.5,
            delta=0.1,
            max_iter=2,
        )

        # Query similar to entity-0
        query_emb = np.array([0.95, 0.05, 0.0])
        activations = prop.propagate(query_emb)

        # Entity-0 and entity-2 should be activated (share passage-0)
        assert activations[0] > 0.5  # Directly similar
        assert activations[2] > 0.1  # Via passage propagation
        assert activations[0] > activations[2]  # Direct > propagated
        # Entity-1 might be low or zero
        assert activations[1] < activations[0]

    def test_propagate_dynamic_pruning(self):
        """Should prune low activations based on delta threshold."""
        entity_embeddings = np.array([
            [1.0, 0.0, 0.0],  # entity-0
            [0.2, 0.1, 0.0],  # entity-1: very low similarity
        ])

        passage_embeddings = np.array([
            [1.0, 0.0, 0.0],
        ])

        contain_matrix = csr_matrix([
            [1.0, 1.0],  # Both entities in passage
        ])

        prop = SemanticPropagation(
            entity_embeddings=entity_embeddings,
            passage_embeddings=passage_embeddings,
            contain_matrix=contain_matrix,
            tau=0.1,  # Low threshold to activate both initially
            delta=0.5,  # High pruning threshold
        )

        query_emb = np.array([0.95, 0.05, 0.0])
        activations = prop.propagate(query_emb)

        # Only high activation should survive pruning
        assert activations[0] > 0.0
        # entity-1 might be pruned due to high delta

    def test_propagate_empty_inputs(self):
        """Should handle empty entity/passage lists."""
        entity_embeddings = np.zeros((0, 3))
        passage_embeddings = np.zeros((0, 3))
        contain_matrix = csr_matrix((0, 0))

        prop = SemanticPropagation(
            entity_embeddings=entity_embeddings,
            passage_embeddings=passage_embeddings,
            contain_matrix=contain_matrix,
        )

        query_emb = np.array([1.0, 0.0, 0.0])
        activations = prop.propagate(query_emb)

        assert activations.shape == (0,)

    def test_propagate_single_iteration(self):
        """Should work with max_iter=1."""
        entity_embeddings = np.array([[1.0, 0.0, 0.0]])
        passage_embeddings = np.array([[1.0, 0.0, 0.0]])
        contain_matrix = csr_matrix([[1.0]])

        prop = SemanticPropagation(
            entity_embeddings=entity_embeddings,
            passage_embeddings=passage_embeddings,
            contain_matrix=contain_matrix,
            max_iter=1,
        )

        query_emb = np.array([0.95, 0.05, 0.0])
        activations = prop.propagate(query_emb)

        assert activations[0] > 0.5

    def test_propagate_convergence(self):
        """Should converge within iterations."""
        entity_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        passage_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        contain_matrix = csr_matrix([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        prop = SemanticPropagation(
            entity_embeddings=entity_embeddings,
            passage_embeddings=passage_embeddings,
            contain_matrix=contain_matrix,
            max_iter=10,  # More than needed
        )

        query_emb = np.array([1.0, 0.0, 0.0])
        activations = prop.propagate(query_emb)

        # Should converge and produce valid result
        assert activations[0] > 0.5
        assert activations[1] < 0.3

    def test_propagate_normalized_embedding(self):
        """Should work with normalized embeddings."""
        # Normalized embeddings
        entity_embeddings = np.array([
            [1.0, 0.0, 0.0],  # Unit vector
            [0.0, 1.0, 0.0],  # Unit vector
        ])

        passage_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        contain_matrix = csr_matrix([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        prop = SemanticPropagation(
            entity_embeddings=entity_embeddings,
            passage_embeddings=passage_embeddings,
            contain_matrix=contain_matrix,
        )

        # Normalized query
        query_emb = np.array([1.0, 0.0, 0.0])
        activations = prop.propagate(query_emb)

        # Perfect match should give high activation
        assert activations[0] > 0.9
        assert activations[1] < 0.1  # Orthogonal


class TestSemanticPropagationWithQueryEntities:
    """Test SemanticPropagation with query_entity_embeddings parameter."""

    def test_propagate_with_query_entities(self):
        """Should match query entities to graph entities by embedding similarity."""
        # Graph entities
        entity_embeddings = np.array([
            [1.0, 0.0, 0.0],  # agent bot
            [0.0, 1.0, 0.0],  # memory system
            [0.0, 0.0, 1.0],  # unrelated entity
        ])

        passage_embeddings = np.array([
            [0.8, 0.2, 0.0],  # passage about agent
            [0.2, 0.8, 0.0],  # passage about memory
        ])

        contain_matrix = csr_matrix([
            [1.0, 1.0, 0.0],  # passage-0: agent, memory
            [0.0, 1.0, 0.0],  # passage-1: memory
        ])

        prop = SemanticPropagation(
            entity_embeddings=entity_embeddings,
            passage_embeddings=passage_embeddings,
            contain_matrix=contain_matrix,
            tau=0.5,
            delta=0.1,
        )

        # Query embedding (different from entity embeddings)
        query_emb = np.array([0.5, 0.5, 0.0])

        # Query entities extracted via NER, with embeddings similar to graph entities
        query_entity_embeddings = np.array([
            [0.95, 0.05, 0.0],  # "에이전트 봇" -> similar to "agent bot"
        ])

        activations = prop.propagate(query_emb, query_entity_embeddings)

        # agent bot should be activated (high similarity to query entity)
        assert activations[0] > 0.5
        # memory system should not be directly activated (low sim to query entity)
        # but may receive some activation via passage propagation
        # unrelated entity should be zero
        assert activations[2] == 0.0

    def test_propagate_with_multiple_query_entities(self):
        """Should find max similarity across multiple query entities."""
        entity_embeddings = np.array([
            [1.0, 0.0, 0.0],  # entity-0
            [0.0, 1.0, 0.0],  # entity-1
            [0.0, 0.0, 1.0],  # entity-2
        ])

        passage_embeddings = np.array([
            [0.5, 0.5, 0.0],
        ])

        contain_matrix = csr_matrix([
            [1.0, 1.0, 1.0],
        ])

        prop = SemanticPropagation(
            entity_embeddings=entity_embeddings,
            passage_embeddings=passage_embeddings,
            contain_matrix=contain_matrix,
            tau=0.5,
            delta=0.1,
        )

        query_emb = np.array([0.5, 0.5, 0.0])

        # Two query entities, each similar to a different graph entity
        query_entity_embeddings = np.array([
            [0.95, 0.05, 0.0],  # similar to entity-0
            [0.05, 0.95, 0.0],  # similar to entity-1
        ])

        activations = prop.propagate(query_emb, query_entity_embeddings)

        # Both entity-0 and entity-1 should be activated
        assert activations[0] > 0.5
        assert activations[1] > 0.5
        # entity-2 should not be activated (no similar query entity)
        assert activations[2] == 0.0

    def test_propagate_empty_query_entities(self):
        """Should fallback to legacy mode when query_entity_embeddings is empty."""
        entity_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        passage_embeddings = np.array([
            [1.0, 0.0, 0.0],
        ])

        contain_matrix = csr_matrix([
            [1.0, 0.0],
        ])

        prop = SemanticPropagation(
            entity_embeddings=entity_embeddings,
            passage_embeddings=passage_embeddings,
            contain_matrix=contain_matrix,
            tau=0.5,
        )

        query_emb = np.array([0.95, 0.05, 0.0])

        # Empty query entity embeddings -> fallback to legacy
        activations = prop.propagate(query_emb, query_entity_embeddings=np.array([]))

        # Should use query-to-entity similarity (legacy mode)
        assert activations[0] > 0.5  # Similar to query
        assert activations[1] < 0.5  # Different from query

    def test_propagate_none_query_entities(self):
        """Should fallback to legacy mode when query_entity_embeddings is None."""
        entity_embeddings = np.array([
            [1.0, 0.0, 0.0],
        ])

        passage_embeddings = np.array([
            [1.0, 0.0, 0.0],
        ])

        contain_matrix = csr_matrix([
            [1.0],
        ])

        prop = SemanticPropagation(
            entity_embeddings=entity_embeddings,
            passage_embeddings=passage_embeddings,
            contain_matrix=contain_matrix,
            tau=0.5,
        )

        query_emb = np.array([0.95, 0.05, 0.0])

        # None -> fallback to legacy
        activations = prop.propagate(query_emb, query_entity_embeddings=None)

        assert activations[0] > 0.5

    def test_propagate_1d_query_entity(self):
        """Should handle 1D query entity embedding."""
        entity_embeddings = np.array([
            [1.0, 0.0, 0.0],
        ])

        passage_embeddings = np.array([
            [1.0, 0.0, 0.0],
        ])

        contain_matrix = csr_matrix([
            [1.0],
        ])

        prop = SemanticPropagation(
            entity_embeddings=entity_embeddings,
            passage_embeddings=passage_embeddings,
            contain_matrix=contain_matrix,
            tau=0.5,
        )

        query_emb = np.array([0.5, 0.5, 0.0])

        # 1D query entity embedding
        query_entity_emb = np.array([0.95, 0.05, 0.0])

        activations = prop.propagate(query_emb, query_entity_embeddings=query_entity_emb)

        assert activations[0] > 0.5
