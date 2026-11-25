"""Tests for BipartiteGraph.

Tests for Entity↔Passage + Passage↔Passage bipartite graph structure.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from reverie.core.bipartite import BipartiteGraph


class TestBipartiteGraph:
    """Test BipartiteGraph operations."""

    def test_add_entity(self):
        """Should add entity nodes."""
        graph = BipartiteGraph()
        idx1 = graph.add_entity(1, "entity-1")
        idx2 = graph.add_entity(2, "entity-2")

        assert idx1 == 0
        assert idx2 == 1
        assert graph.entity_count == 2
        assert graph.get_entity_id(idx1) == 1
        assert graph.get_entity_id(idx2) == 2

    def test_add_entity_duplicate(self):
        """Should return existing index for duplicate entity."""
        graph = BipartiteGraph()
        idx1 = graph.add_entity(1, "entity-1")
        idx2 = graph.add_entity(1, "entity-1")  # Same entity_id

        assert idx1 == idx2
        assert graph.entity_count == 1

    def test_add_passage(self):
        """Should add passage nodes."""
        graph = BipartiteGraph()
        idx1 = graph.add_passage(10, "session-1", 0)
        idx2 = graph.add_passage(11, "session-1", 1)

        assert idx1 == 0
        assert idx2 == 1
        assert graph.passage_count == 2
        assert graph.get_passage_id(idx1) == 10
        assert graph.get_passage_id(idx2) == 11

    def test_add_passage_duplicate(self):
        """Should return existing index for duplicate passage."""
        graph = BipartiteGraph()
        idx1 = graph.add_passage(10, "session-1", 0)
        idx2 = graph.add_passage(10, "session-1", 0)  # Same passage_id

        assert idx1 == idx2
        assert graph.passage_count == 1

    def test_add_contain_edge(self):
        """Should add entity-passage contain edge."""
        graph = BipartiteGraph()
        graph.add_entity(1, "entity-1")
        graph.add_passage(10, "session-1", 0)

        graph.add_contain_edge(1, 10)

        # Verify edge exists
        C = graph.get_contain_matrix()
        assert C.shape == (1, 1)  # (passages, entities)
        assert C[0, 0] == 1.0

    def test_add_contain_edge_multiple(self):
        """Should handle multiple contain edges."""
        graph = BipartiteGraph()
        graph.add_entity(1, "entity-1")
        graph.add_entity(2, "entity-2")
        graph.add_passage(10, "session-1", 0)
        graph.add_passage(11, "session-1", 1)

        # Entity 1 in passages 10, 11
        graph.add_contain_edge(1, 10)
        graph.add_contain_edge(1, 11)
        # Entity 2 only in passage 10
        graph.add_contain_edge(2, 10)

        C = graph.get_contain_matrix()
        assert C.shape == (2, 2)  # (passages, entities)

        # Row 0 (passage 10): entity 1 and 2
        assert C[0, 0] == 1.0
        assert C[0, 1] == 1.0
        # Row 1 (passage 11): only entity 1
        assert C[1, 0] == 1.0
        assert C[1, 1] == 0.0

    def test_add_adjacency_edge(self):
        """Should add passage-passage adjacency edge."""
        graph = BipartiteGraph()
        graph.add_passage(10, "session-1", 0)
        graph.add_passage(11, "session-1", 1)

        graph.add_adjacency_edge(10, 11)

        A = graph.get_adjacency_matrix()
        assert A.shape == (2, 2)  # (passages, passages)
        # Symmetric matrix
        assert A[0, 1] == 1.0
        assert A[1, 0] == 1.0
        # No self-loops
        assert A[0, 0] == 0.0
        assert A[1, 1] == 0.0

    def test_add_adjacency_edge_auto_order(self):
        """Should auto-order passage IDs for adjacency."""
        graph = BipartiteGraph()
        graph.add_passage(10, "session-1", 0)
        graph.add_passage(11, "session-1", 1)

        # Add in reversed order
        graph.add_adjacency_edge(11, 10)

        A = graph.get_adjacency_matrix()
        assert A[0, 1] == 1.0
        assert A[1, 0] == 1.0

    def test_get_unified_adjacency(self):
        """Should return unified adjacency matrix for PPR.

        Structure:
               Entity    Passage
        Entity [  0    |   Cᵀ   ]
        Passage[  C    |   A    ]
        """
        graph = BipartiteGraph()
        # 2 entities, 2 passages
        graph.add_entity(1, "entity-1")
        graph.add_entity(2, "entity-2")
        graph.add_passage(10, "session-1", 0)
        graph.add_passage(11, "session-1", 1)

        # Contain: entity-1 in passage-10, entity-2 in passage-11
        graph.add_contain_edge(1, 10)
        graph.add_contain_edge(2, 11)

        # Adjacency: passage-10 adjacent to passage-11
        graph.add_adjacency_edge(10, 11)

        Ã = graph.get_unified_adjacency()

        # Size should be (n_entities + n_passages) x (n_entities + n_passages)
        assert Ã.shape == (4, 4)

        # Convert to dense for easier assertion
        Ã_dense = Ã.toarray()

        # Entity-Entity block (top-left 2x2): zeros
        assert Ã_dense[0, 0] == 0.0
        assert Ã_dense[0, 1] == 0.0
        assert Ã_dense[1, 0] == 0.0
        assert Ã_dense[1, 1] == 0.0

        # Entity-Passage block (top-right 2x2): Cᵀ
        # C[0,0]=1 (passage-10 contains entity-1) → Cᵀ[0,0]=1
        # C[1,1]=1 (passage-11 contains entity-2) → Cᵀ[1,1]=1
        assert Ã_dense[0, 2] == 1.0  # entity-1 → passage-10
        assert Ã_dense[1, 3] == 1.0  # entity-2 → passage-11
        assert Ã_dense[0, 3] == 0.0
        assert Ã_dense[1, 2] == 0.0

        # Passage-Entity block (bottom-left 2x2): C
        assert Ã_dense[2, 0] == 1.0  # passage-10 → entity-1
        assert Ã_dense[3, 1] == 1.0  # passage-11 → entity-2

        # Passage-Passage block (bottom-right 2x2): A
        assert Ã_dense[2, 3] == 1.0  # passage-10 ↔ passage-11
        assert Ã_dense[3, 2] == 1.0

    def test_empty_graph(self):
        """Should handle empty graph gracefully."""
        graph = BipartiteGraph()

        assert graph.entity_count == 0
        assert graph.passage_count == 0

        C = graph.get_contain_matrix()
        assert C.shape == (0, 0)

        A = graph.get_adjacency_matrix()
        assert A.shape == (0, 0)

        Ã = graph.get_unified_adjacency()
        assert Ã.shape == (0, 0)

    def test_graph_with_only_entities(self):
        """Should handle graph with only entities."""
        graph = BipartiteGraph()
        graph.add_entity(1, "entity-1")
        graph.add_entity(2, "entity-2")

        assert graph.entity_count == 2
        assert graph.passage_count == 0

        C = graph.get_contain_matrix()
        assert C.shape == (0, 2)

        A = graph.get_adjacency_matrix()
        assert A.shape == (0, 0)

    def test_graph_with_only_passages(self):
        """Should handle graph with only passages."""
        graph = BipartiteGraph()
        graph.add_passage(10, "session-1", 0)
        graph.add_passage(11, "session-1", 1)

        assert graph.entity_count == 0
        assert graph.passage_count == 2

        C = graph.get_contain_matrix()
        assert C.shape == (2, 0)

    def test_get_entity_ids(self):
        """Should return all entity IDs."""
        graph = BipartiteGraph()
        graph.add_entity(1, "entity-1")
        graph.add_entity(5, "entity-5")
        graph.add_entity(3, "entity-3")

        entity_ids = graph.get_entity_ids()
        assert set(entity_ids) == {1, 5, 3}

    def test_get_passage_ids(self):
        """Should return all passage IDs."""
        graph = BipartiteGraph()
        graph.add_passage(10, "session-1", 0)
        graph.add_passage(20, "session-1", 1)

        passage_ids = graph.get_passage_ids()
        assert set(passage_ids) == {10, 20}

    def test_get_entity_index(self):
        """Should return vertex index for entity ID."""
        graph = BipartiteGraph()
        graph.add_entity(1, "entity-1")
        graph.add_entity(5, "entity-5")

        assert graph.get_entity_index(1) == 0
        assert graph.get_entity_index(5) == 1
        assert graph.get_entity_index(999) is None

    def test_get_passage_index(self):
        """Should return vertex index for passage ID."""
        graph = BipartiteGraph()
        graph.add_passage(10, "session-1", 0)
        graph.add_passage(20, "session-1", 1)

        assert graph.get_passage_index(10) == 0
        assert graph.get_passage_index(20) == 1
        assert graph.get_passage_index(999) is None

    def test_get_entity_passages(self):
        """Should return passages containing the entity."""
        graph = BipartiteGraph()
        graph.add_entity(1, "entity-1")
        graph.add_entity(2, "entity-2")
        graph.add_passage(10, "session-1", 0)
        graph.add_passage(11, "session-1", 1)
        graph.add_passage(12, "session-1", 2)

        # Entity 1 appears in passages 10 and 11
        graph.add_contain_edge(1, 10)
        graph.add_contain_edge(1, 11)
        # Entity 2 only in passage 12
        graph.add_contain_edge(2, 12)

        assert graph.get_entity_passages(1) == {10, 11}
        assert graph.get_entity_passages(2) == {12}
        assert graph.get_entity_passages(999) == set()  # Non-existent entity

    def test_get_passage_entities(self):
        """Should return entities mentioned in the passage."""
        graph = BipartiteGraph()
        graph.add_entity(1, "entity-1")
        graph.add_entity(2, "entity-2")
        graph.add_passage(10, "session-1", 0)
        graph.add_passage(11, "session-1", 1)

        # Passage 10 contains entities 1 and 2
        graph.add_contain_edge(1, 10)
        graph.add_contain_edge(2, 10)
        # Passage 11 only contains entity 1
        graph.add_contain_edge(1, 11)

        assert graph.get_passage_entities(10) == {1, 2}
        assert graph.get_passage_entities(11) == {1}
        assert graph.get_passage_entities(999) == set()  # Non-existent passage

    def test_get_adjacent_passages(self):
        """Should return adjacent passages."""
        graph = BipartiteGraph()
        graph.add_passage(10, "session-1", 0)
        graph.add_passage(11, "session-1", 1)
        graph.add_passage(12, "session-1", 2)

        # 10 ↔ 11 ↔ 12 (chain)
        graph.add_adjacency_edge(10, 11)
        graph.add_adjacency_edge(11, 12)

        assert graph.get_adjacent_passages(10) == {11}
        assert graph.get_adjacent_passages(11) == {10, 12}  # Both sides
        assert graph.get_adjacent_passages(12) == {11}
        assert graph.get_adjacent_passages(999) == set()  # Non-existent passage
