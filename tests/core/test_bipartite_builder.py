"""Tests for BipartiteGraphBuilder.

Tests for building BipartiteGraph from database repositories.
"""

import pytest
from unittest.mock import MagicMock, patch

from reverie.core.bipartite import BipartiteGraph
from reverie.core.bipartite_builder import BipartiteGraphBuilder


class MockEntityRow:
    """Mock for EntityRow."""
    def __init__(self, id: int, name: str, embedding=None):
        self.id = id
        self.name = name
        self.embedding = embedding


class MockPassageRow:
    """Mock for PassageRow."""
    def __init__(self, id: int, session_id: str, turn_index: int, embedding=None, session_seq: int = None):
        self.id = id
        self.session_id = session_id
        self.turn_index = turn_index
        self.embedding = embedding
        self.session_seq = session_seq


class MockEntityPassage:
    """Mock for EntityPassage."""
    def __init__(self, entity_id: int, passage_id: int):
        self.entity_id = entity_id
        self.passage_id = passage_id


class MockPassageAdjacency:
    """Mock for PassageAdjacency."""
    def __init__(self, passage_id_a: int, passage_id_b: int):
        self.passage_id_a = passage_id_a
        self.passage_id_b = passage_id_b


class TestBipartiteGraphBuilder:
    """Test BipartiteGraphBuilder."""

    def test_build_empty_graph(self):
        """Should build empty graph when no data."""
        # Mock repositories
        entity_repo = MagicMock()
        entity_repo.get_all.return_value = []

        passage_repo = MagicMock()
        passage_repo.get_all.return_value = []

        entity_passage_repo = MagicMock()
        entity_passage_repo.get_all.return_value = []

        adjacency_repo = MagicMock()
        adjacency_repo.get_all.return_value = []

        builder = BipartiteGraphBuilder(
            entity_repo=entity_repo,
            passage_repo=passage_repo,
            entity_passage_repo=entity_passage_repo,
            adjacency_repo=adjacency_repo,
        )

        graph = builder.build()

        assert graph.entity_count == 0
        assert graph.passage_count == 0

    def test_build_with_entities_and_passages(self):
        """Should build graph with entities and passages."""
        # Mock entities
        entity_repo = MagicMock()
        entity_repo.get_all.return_value = [
            MockEntityRow(1, "entity-1"),
            MockEntityRow(2, "entity-2"),
        ]

        # Mock passages
        passage_repo = MagicMock()
        passage_repo.get_all.return_value = [
            MockPassageRow(10, "session-1", 0),
            MockPassageRow(11, "session-1", 1),
        ]

        # Mock entity-passage links
        entity_passage_repo = MagicMock()
        entity_passage_repo.get_all.return_value = [
            MockEntityPassage(1, 10),  # entity-1 in passage-10
            MockEntityPassage(2, 11),  # entity-2 in passage-11
        ]

        # Mock adjacencies
        adjacency_repo = MagicMock()
        adjacency_repo.get_all.return_value = [
            MockPassageAdjacency(10, 11),  # passage-10 adjacent to passage-11
        ]

        builder = BipartiteGraphBuilder(
            entity_repo=entity_repo,
            passage_repo=passage_repo,
            entity_passage_repo=entity_passage_repo,
            adjacency_repo=adjacency_repo,
        )

        graph = builder.build()

        # Check counts
        assert graph.entity_count == 2
        assert graph.passage_count == 2

        # Check contain matrix
        C = graph.get_contain_matrix()
        assert C[0, 0] == 1.0  # passage-10 contains entity-1
        assert C[1, 1] == 1.0  # passage-11 contains entity-2
        assert C[0, 1] == 0.0
        assert C[1, 0] == 0.0

        # Check adjacency matrix
        A = graph.get_adjacency_matrix()
        assert A[0, 1] == 1.0  # passage-10 adjacent to passage-11
        assert A[1, 0] == 1.0  # symmetric

    def test_build_with_connection(self):
        """Should build graph using database connection."""
        # Mock connection and repositories
        conn = MagicMock()

        with patch('reverie.core.bipartite_builder.EntityRepository') as MockEntityRepo, \
             patch('reverie.core.bipartite_builder.PassageRepository') as MockPassageRepo, \
             patch('reverie.core.bipartite_builder.EntityPassageRepository') as MockEPRepo, \
             patch('reverie.core.bipartite_builder.AdjacencyRepository') as MockAdjRepo:

            # Configure mocks
            MockEntityRepo.return_value.get_all.return_value = [
                MockEntityRow(1, "test-entity"),
            ]
            MockPassageRepo.return_value.get_all.return_value = [
                MockPassageRow(10, "session-1", 0),
            ]
            MockEPRepo.return_value.get_all.return_value = [
                MockEntityPassage(1, 10),
            ]
            MockAdjRepo.return_value.get_all.return_value = []

            builder = BipartiteGraphBuilder.from_connection(conn)
            graph = builder.build()

            assert graph.entity_count == 1
            assert graph.passage_count == 1

    def test_get_entity_embeddings(self):
        """Should return entity embeddings matrix."""
        # Mock entities with embeddings
        entity_repo = MagicMock()
        entity_repo.get_all.return_value = [
            MockEntityRow(1, "entity-1", embedding=[0.1] * 1536),
            MockEntityRow(2, "entity-2", embedding=[0.2] * 1536),
        ]

        passage_repo = MagicMock()
        passage_repo.get_all.return_value = []

        entity_passage_repo = MagicMock()
        entity_passage_repo.get_all.return_value = []

        adjacency_repo = MagicMock()
        adjacency_repo.get_all.return_value = []

        builder = BipartiteGraphBuilder(
            entity_repo=entity_repo,
            passage_repo=passage_repo,
            entity_passage_repo=entity_passage_repo,
            adjacency_repo=adjacency_repo,
        )

        graph = builder.build()
        entity_embeddings = builder.get_entity_embeddings()

        assert entity_embeddings.shape == (2, 1536)
        assert entity_embeddings[0, 0] == pytest.approx(0.1)
        assert entity_embeddings[1, 0] == pytest.approx(0.2)

    def test_get_passage_embeddings(self):
        """Should return passage embeddings matrix."""
        entity_repo = MagicMock()
        entity_repo.get_all.return_value = []

        # Mock passages with embeddings
        passage_repo = MagicMock()
        passage_repo.get_all.return_value = [
            MockPassageRow(10, "session-1", 0, embedding=[0.3] * 1536),
            MockPassageRow(11, "session-1", 1, embedding=[0.4] * 1536),
        ]

        entity_passage_repo = MagicMock()
        entity_passage_repo.get_all.return_value = []

        adjacency_repo = MagicMock()
        adjacency_repo.get_all.return_value = []

        builder = BipartiteGraphBuilder(
            entity_repo=entity_repo,
            passage_repo=passage_repo,
            entity_passage_repo=entity_passage_repo,
            adjacency_repo=adjacency_repo,
        )

        graph = builder.build()
        passage_embeddings = builder.get_passage_embeddings()

        assert passage_embeddings.shape == (2, 1536)
        assert passage_embeddings[0, 0] == pytest.approx(0.3)
        assert passage_embeddings[1, 0] == pytest.approx(0.4)

    def test_get_embeddings_with_missing(self):
        """Should handle entities/passages without embeddings."""
        entity_repo = MagicMock()
        entity_repo.get_all.return_value = [
            MockEntityRow(1, "entity-1", embedding=[0.1] * 1536),
            MockEntityRow(2, "entity-2", embedding=None),  # No embedding
        ]

        passage_repo = MagicMock()
        passage_repo.get_all.return_value = [
            MockPassageRow(10, "session-1", 0, embedding=None),  # No embedding
            MockPassageRow(11, "session-1", 1, embedding=[0.5] * 1536),
        ]

        entity_passage_repo = MagicMock()
        entity_passage_repo.get_all.return_value = []

        adjacency_repo = MagicMock()
        adjacency_repo.get_all.return_value = []

        builder = BipartiteGraphBuilder(
            entity_repo=entity_repo,
            passage_repo=passage_repo,
            entity_passage_repo=entity_passage_repo,
            adjacency_repo=adjacency_repo,
        )

        graph = builder.build()

        # Entity embeddings: one has embedding, one is zeros
        entity_embeddings = builder.get_entity_embeddings()
        assert entity_embeddings.shape == (2, 1536)
        assert entity_embeddings[0, 0] == pytest.approx(0.1)
        assert entity_embeddings[1, 0] == pytest.approx(0.0)  # Missing → zeros

        # Passage embeddings: one is zeros, one has embedding
        passage_embeddings = builder.get_passage_embeddings()
        assert passage_embeddings.shape == (2, 1536)
        assert passage_embeddings[0, 0] == pytest.approx(0.0)  # Missing → zeros
        assert passage_embeddings[1, 0] == pytest.approx(0.5)
