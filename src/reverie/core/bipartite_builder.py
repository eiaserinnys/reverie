"""BipartiteGraphBuilder for MemoryLink v2.

Builds BipartiteGraph from database repositories.
"""

import logging
from typing import Optional

import numpy as np

from reverie.db.repositories.adjacency_repo import AdjacencyRepository
from reverie.db.repositories.entity_passage_repo import EntityPassageRepository
from reverie.db.repositories.entity_repo import EntityRepository
from reverie.db.repositories.passage_repo import PassageRepository
from reverie.core.bipartite import BipartiteGraph

logger = logging.getLogger(__name__)


class BipartiteGraphBuilder:
    """Builds BipartiteGraph from database.

    Loads entities, passages, contain edges, and adjacency edges
    from repositories and constructs the graph structure.
    """

    def __init__(
        self,
        entity_repo: EntityRepository,
        passage_repo: PassageRepository,
        entity_passage_repo: EntityPassageRepository,
        adjacency_repo: AdjacencyRepository,
    ):
        self._entity_repo = entity_repo
        self._passage_repo = passage_repo
        self._entity_passage_repo = entity_passage_repo
        self._adjacency_repo = adjacency_repo

        # Built graph and embedding caches
        self._graph: Optional[BipartiteGraph] = None
        self._entity_embeddings: Optional[np.ndarray] = None
        self._passage_embeddings: Optional[np.ndarray] = None

        # Ordered lists (matching graph indices)
        self._entity_list: list = []
        self._passage_list: list = []

    @classmethod
    def from_connection(cls, conn) -> "BipartiteGraphBuilder":
        """Create builder from database connection.

        Args:
            conn: Database connection.

        Returns:
            BipartiteGraphBuilder instance.
        """
        return cls(
            entity_repo=EntityRepository(conn),
            passage_repo=PassageRepository(conn),
            entity_passage_repo=EntityPassageRepository(conn),
            adjacency_repo=AdjacencyRepository(conn),
        )

    def build(self) -> BipartiteGraph:
        """Build and return the BipartiteGraph.

        Loads all data from repositories and constructs the graph.

        Returns:
            Built BipartiteGraph.
        """
        graph = BipartiteGraph()

        # Load entities
        entities = self._entity_repo.get_all()
        self._entity_list = []
        for entity in entities:
            graph.add_entity(entity.id, entity.name)
            self._entity_list.append(entity)

        logger.debug(f"Loaded {len(entities)} entities")

        # Load passages
        passages = self._passage_repo.get_all()
        self._passage_list = []
        for passage in passages:
            graph.add_passage(passage.id, passage.session_id, passage.turn_index)
            self._passage_list.append(passage)

        logger.debug(f"Loaded {len(passages)} passages")

        # Load contain edges (entity-passage links)
        entity_passages = self._entity_passage_repo.get_all()
        for ep in entity_passages:
            graph.add_contain_edge(ep.entity_id, ep.passage_id)

        logger.debug(f"Loaded {len(entity_passages)} contain edges")

        # Load adjacency edges (passage-passage links)
        adjacencies = self._adjacency_repo.get_all()
        for adj in adjacencies:
            graph.add_adjacency_edge(adj.passage_id_a, adj.passage_id_b)

        logger.debug(f"Loaded {len(adjacencies)} adjacency edges")

        # Cache the graph
        self._graph = graph

        # Invalidate embedding caches
        self._entity_embeddings = None
        self._passage_embeddings = None

        return graph

    def get_entity_embeddings(self, embedding_dim: int = 1536) -> np.ndarray:
        """Get entity embeddings matrix.

        Args:
            embedding_dim: Dimension of embeddings.

        Returns:
            numpy array of shape (n_entities, embedding_dim).
            Entities without embeddings have zero vectors.
        """
        if self._entity_embeddings is not None:
            return self._entity_embeddings

        if self._graph is None:
            raise RuntimeError("Must call build() before get_entity_embeddings()")

        n_entities = len(self._entity_list)
        embeddings = np.zeros((n_entities, embedding_dim), dtype=np.float64)

        for i, entity in enumerate(self._entity_list):
            if entity.embedding is not None:
                embeddings[i] = np.array(entity.embedding)

        self._entity_embeddings = embeddings
        return embeddings

    def get_passage_embeddings(self, embedding_dim: int = 1536) -> np.ndarray:
        """Get passage embeddings matrix.

        Args:
            embedding_dim: Dimension of embeddings.

        Returns:
            numpy array of shape (n_passages, embedding_dim).
            Passages without embeddings have zero vectors.
        """
        if self._passage_embeddings is not None:
            return self._passage_embeddings

        if self._graph is None:
            raise RuntimeError("Must call build() before get_passage_embeddings()")

        n_passages = len(self._passage_list)
        embeddings = np.zeros((n_passages, embedding_dim), dtype=np.float64)

        for i, passage in enumerate(self._passage_list):
            if passage.embedding is not None:
                embeddings[i] = np.array(passage.embedding)

        self._passage_embeddings = embeddings
        return embeddings
