"""Tests for BipartitePPR.

Tests for LinearRAG Stage 2: Personalized PageRank on bipartite graph.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from reverie.core.bipartite import BipartiteGraph
from reverie.core.bipartite_ppr import BipartitePPR


class TestBipartitePPR:
    """Test BipartitePPR operations."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph.

        2 entities, 3 passages:
        - entity-0 appears in passage-0
        - entity-1 appears in passage-1 and passage-2
        - passage-0 adjacent to passage-1
        - passage-1 adjacent to passage-2
        """
        graph = BipartiteGraph()

        # Add entities
        graph.add_entity(1, "entity-0")
        graph.add_entity(2, "entity-1")

        # Add passages
        graph.add_passage(10, "session-1", 0)
        graph.add_passage(11, "session-1", 1)
        graph.add_passage(12, "session-1", 2)

        # Contain edges
        graph.add_contain_edge(1, 10)  # entity-0 in passage-0
        graph.add_contain_edge(2, 11)  # entity-1 in passage-1
        graph.add_contain_edge(2, 12)  # entity-1 in passage-2

        # Adjacency edges
        graph.add_adjacency_edge(10, 11)  # passage-0 ↔ passage-1
        graph.add_adjacency_edge(11, 12)  # passage-1 ↔ passage-2

        return graph

    def test_run_basic(self, simple_graph):
        """Should run PPR and return passage scores."""
        ppr = BipartitePPR(simple_graph, damping=0.85)

        # Activate entity-0
        entity_activations = np.array([1.0, 0.0])

        results = ppr.run(entity_activations, top_k=3)

        # Should return list of (passage_id, score)
        assert len(results) == 3
        assert all(isinstance(r[0], int) for r in results)
        assert all(isinstance(r[1], float) for r in results)

        # Passage IDs should be 10, 11, 12
        passage_ids = [r[0] for r in results]
        assert set(passage_ids) == {10, 11, 12}

        # passage-0 (10) should have highest score (contains activated entity)
        top_passage_id = results[0][0]
        assert top_passage_id == 10

    def test_run_propagation(self, simple_graph):
        """Should propagate activation to neighboring passages."""
        ppr = BipartitePPR(simple_graph, damping=0.85)

        # Activate entity-0 only
        entity_activations = np.array([1.0, 0.0])

        results = ppr.run(entity_activations, top_k=3)

        # Find scores for each passage
        scores = {r[0]: r[1] for r in results}

        # passage-0 (10) should have highest score
        # passage-1 (11) should have some score from adjacency
        # passage-2 (12) should have lower score (2 hops away)
        assert scores[10] > scores[11]
        assert scores[11] > 0  # Should receive some propagation

    def test_run_multi_entity_activation(self, simple_graph):
        """Should handle multiple activated entities."""
        ppr = BipartitePPR(simple_graph, damping=0.85)

        # Activate both entities
        entity_activations = np.array([0.5, 0.5])

        results = ppr.run(entity_activations, top_k=3)

        # All passages should have scores
        scores = {r[0]: r[1] for r in results}
        assert all(score > 0 for score in scores.values())

    def test_run_zero_activation(self, simple_graph):
        """Should handle zero entity activation."""
        ppr = BipartitePPR(simple_graph, damping=0.85)

        # No entity activated
        entity_activations = np.array([0.0, 0.0])

        results = ppr.run(entity_activations, top_k=3)

        # Should return results with zero scores
        assert len(results) == 3
        assert all(r[1] == 0.0 for r in results)

    def test_run_top_k(self, simple_graph):
        """Should return only top_k passages."""
        ppr = BipartitePPR(simple_graph, damping=0.85)

        entity_activations = np.array([1.0, 0.0])

        results = ppr.run(entity_activations, top_k=2)

        # Should return only 2 passages
        assert len(results) == 2

        # Should be sorted by score descending
        assert results[0][1] >= results[1][1]

    def test_run_damping_high(self, simple_graph):
        """High damping should spread activation more."""
        ppr_high = BipartitePPR(simple_graph, damping=0.95)
        ppr_low = BipartitePPR(simple_graph, damping=0.5)

        entity_activations = np.array([1.0, 0.0])

        results_high = ppr_high.run(entity_activations, top_k=3)
        results_low = ppr_low.run(entity_activations, top_k=3)

        scores_high = {r[0]: r[1] for r in results_high}
        scores_low = {r[0]: r[1] for r in results_low}

        # With high damping, remote passages should have higher relative scores
        # passage-2 (12) is 2 hops away
        ratio_high = scores_high[12] / scores_high[10] if scores_high[10] > 0 else 0
        ratio_low = scores_low[12] / scores_low[10] if scores_low[10] > 0 else 0

        # High damping should spread more (higher ratio for distant nodes)
        assert ratio_high > ratio_low

    def test_run_empty_graph(self):
        """Should handle empty graph."""
        graph = BipartiteGraph()
        ppr = BipartitePPR(graph, damping=0.85)

        entity_activations = np.array([])
        results = ppr.run(entity_activations, top_k=10)

        assert len(results) == 0

    def test_run_single_passage(self):
        """Should handle graph with single passage."""
        graph = BipartiteGraph()
        graph.add_entity(1, "entity-0")
        graph.add_passage(10, "session-1", 0)
        graph.add_contain_edge(1, 10)

        ppr = BipartitePPR(graph, damping=0.85)
        entity_activations = np.array([1.0])

        results = ppr.run(entity_activations, top_k=10)

        assert len(results) == 1
        assert results[0][0] == 10
        assert results[0][1] > 0

    def test_run_no_contain_edges(self):
        """Should handle graph with no contain edges."""
        graph = BipartiteGraph()
        graph.add_entity(1, "entity-0")
        graph.add_passage(10, "session-1", 0)
        graph.add_passage(11, "session-1", 1)
        graph.add_adjacency_edge(10, 11)
        # No contain edges

        ppr = BipartitePPR(graph, damping=0.85)
        entity_activations = np.array([1.0])

        results = ppr.run(entity_activations, top_k=10)

        # Should return passages (but with very low scores)
        assert len(results) == 2

    def test_convergence(self, simple_graph):
        """Should converge within max iterations."""
        ppr = BipartitePPR(simple_graph, damping=0.85, max_iter=100)

        entity_activations = np.array([1.0, 0.0])

        # Should not raise or hang
        results = ppr.run(entity_activations, top_k=3)

        assert len(results) == 3

    def test_passage_only_graph(self):
        """Should handle graph with passages but no entities."""
        graph = BipartiteGraph()
        graph.add_passage(10, "session-1", 0)
        graph.add_passage(11, "session-1", 1)
        graph.add_adjacency_edge(10, 11)

        ppr = BipartitePPR(graph, damping=0.85)
        entity_activations = np.array([])

        results = ppr.run(entity_activations, top_k=10)

        # No entities means no activation, should return passages with zero scores
        assert len(results) == 2
