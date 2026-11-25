"""BipartiteGraph for MemoryLink v2.

Entity↔Passage + Passage↔Passage bipartite graph structure.
All edge weights are 1.0 (no decay).
"""

import logging
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, bmat

logger = logging.getLogger(__name__)


class BipartiteGraph:
    """Entity↔Passage + Passage↔Passage bipartite graph.

    Nodes: Entity, Passage
    Edges:
        - Entity ↔ Passage (contain)
        - Passage ↔ Passage (session adjacency)
    All weights = 1
    """

    def __init__(self):
        # Entity mappings: entity_id ↔ vertex_index
        self._entity_idx: dict[int, int] = {}  # entity_id → vertex_idx
        self._idx_to_entity: dict[int, int] = {}  # vertex_idx → entity_id
        self._entity_names: dict[int, str] = {}  # entity_id → name

        # Passage mappings: passage_id ↔ vertex_index
        self._passage_idx: dict[int, int] = {}  # passage_id → vertex_idx
        self._idx_to_passage: dict[int, int] = {}  # vertex_idx → passage_id
        self._passage_info: dict[int, tuple[str, int]] = {}  # passage_id → (session_id, turn_index)

        # Edge lists (stored until matrix is built)
        self._contain_edges: list[tuple[int, int]] = []  # [(entity_id, passage_id), ...]
        self._adjacency_edges: list[tuple[int, int]] = []  # [(passage_id_a, passage_id_b), ...]

        # Cached matrices (lazy build)
        self._contain_matrix: Optional[csr_matrix] = None
        self._adjacency_matrix: Optional[csr_matrix] = None
        self._unified_adjacency: Optional[csr_matrix] = None

    # --- Node operations ---

    def add_entity(self, entity_id: int, name: str) -> int:
        """Add an entity node.

        Args:
            entity_id: Database entity ID.
            name: Entity name.

        Returns:
            Vertex index for this entity.
        """
        if entity_id in self._entity_idx:
            return self._entity_idx[entity_id]

        idx = len(self._entity_idx)
        self._entity_idx[entity_id] = idx
        self._idx_to_entity[idx] = entity_id
        self._entity_names[entity_id] = name
        self._invalidate_cache()
        return idx

    def add_passage(self, passage_id: int, session_id: str, turn_index: int) -> int:
        """Add a passage node.

        Args:
            passage_id: Database passage ID.
            session_id: Session identifier.
            turn_index: Turn index within session.

        Returns:
            Vertex index for this passage.
        """
        if passage_id in self._passage_idx:
            return self._passage_idx[passage_id]

        idx = len(self._passage_idx)
        self._passage_idx[passage_id] = idx
        self._idx_to_passage[idx] = passage_id
        self._passage_info[passage_id] = (session_id, turn_index)
        self._invalidate_cache()
        return idx

    # --- Edge operations ---

    def add_contain_edge(self, entity_id: int, passage_id: int) -> None:
        """Add entity-passage contain edge.

        Args:
            entity_id: Entity ID (must already be added).
            passage_id: Passage ID (must already be added).
        """
        if entity_id not in self._entity_idx:
            logger.warning(f"Entity {entity_id} not in graph, skipping contain edge")
            return
        if passage_id not in self._passage_idx:
            logger.warning(f"Passage {passage_id} not in graph, skipping contain edge")
            return

        self._contain_edges.append((entity_id, passage_id))
        self._invalidate_cache()

    def add_adjacency_edge(self, passage_id_a: int, passage_id_b: int) -> None:
        """Add passage-passage adjacency edge.

        Args:
            passage_id_a: First passage ID.
            passage_id_b: Second passage ID.
        """
        if passage_id_a not in self._passage_idx:
            logger.warning(f"Passage {passage_id_a} not in graph, skipping adjacency edge")
            return
        if passage_id_b not in self._passage_idx:
            logger.warning(f"Passage {passage_id_b} not in graph, skipping adjacency edge")
            return

        # Store in canonical order (smaller first)
        pid_a, pid_b = min(passage_id_a, passage_id_b), max(passage_id_a, passage_id_b)
        self._adjacency_edges.append((pid_a, pid_b))
        self._invalidate_cache()

    # --- Matrix accessors ---

    def get_contain_matrix(self) -> csr_matrix:
        """Get Contain matrix C (passages × entities).

        Returns:
            Sparse matrix of shape (n_passages, n_entities).
            C[p, e] = 1 if entity e appears in passage p.
        """
        if self._contain_matrix is not None:
            return self._contain_matrix

        n_passages = len(self._passage_idx)
        n_entities = len(self._entity_idx)

        if n_passages == 0 or n_entities == 0:
            self._contain_matrix = csr_matrix((n_passages, n_entities))
            return self._contain_matrix

        # Build using lil_matrix for efficient incremental construction
        C = lil_matrix((n_passages, n_entities), dtype=np.float64)

        for entity_id, passage_id in self._contain_edges:
            p_idx = self._passage_idx[passage_id]
            e_idx = self._entity_idx[entity_id]
            C[p_idx, e_idx] = 1.0

        self._contain_matrix = C.tocsr()
        return self._contain_matrix

    def get_adjacency_matrix(self) -> csr_matrix:
        """Get Passage adjacency matrix A (passages × passages).

        Returns:
            Symmetric sparse matrix of shape (n_passages, n_passages).
            A[i, j] = 1 if passages i and j are adjacent.
        """
        if self._adjacency_matrix is not None:
            return self._adjacency_matrix

        n_passages = len(self._passage_idx)

        if n_passages == 0:
            self._adjacency_matrix = csr_matrix((0, 0))
            return self._adjacency_matrix

        # Build symmetric adjacency matrix
        A = lil_matrix((n_passages, n_passages), dtype=np.float64)

        for pid_a, pid_b in self._adjacency_edges:
            i = self._passage_idx[pid_a]
            j = self._passage_idx[pid_b]
            A[i, j] = 1.0
            A[j, i] = 1.0

        self._adjacency_matrix = A.tocsr()
        return self._adjacency_matrix

    def get_unified_adjacency(self) -> csr_matrix:
        """Get unified adjacency matrix for PPR.

        Structure:
               Entity    Passage
        Entity [  0    |   Cᵀ   ]
        Passage[  C    |   A    ]

        Returns:
            Sparse matrix of shape (n_entities + n_passages, n_entities + n_passages).
        """
        if self._unified_adjacency is not None:
            return self._unified_adjacency

        n_entities = len(self._entity_idx)
        n_passages = len(self._passage_idx)
        n_total = n_entities + n_passages

        if n_total == 0:
            self._unified_adjacency = csr_matrix((0, 0))
            return self._unified_adjacency

        C = self.get_contain_matrix()  # (passages, entities)
        A = self.get_adjacency_matrix()  # (passages, passages)

        # Build block matrix
        # Top-left: zeros (entities × entities)
        top_left = csr_matrix((n_entities, n_entities), dtype=np.float64)

        # Top-right: Cᵀ (entities × passages)
        top_right = C.T if n_entities > 0 and n_passages > 0 else csr_matrix((n_entities, n_passages))

        # Bottom-left: C (passages × entities)
        bottom_left = C if n_entities > 0 and n_passages > 0 else csr_matrix((n_passages, n_entities))

        # Bottom-right: A (passages × passages)
        bottom_right = A

        # Combine using scipy.sparse.bmat
        self._unified_adjacency = bmat([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ], format='csr')

        return self._unified_adjacency

    # --- Query methods ---

    @property
    def entity_count(self) -> int:
        """Number of entity nodes."""
        return len(self._entity_idx)

    @property
    def passage_count(self) -> int:
        """Number of passage nodes."""
        return len(self._passage_idx)

    def get_entity_id(self, vertex_idx: int) -> int:
        """Get entity ID from vertex index."""
        return self._idx_to_entity[vertex_idx]

    def get_passage_id(self, vertex_idx: int) -> int:
        """Get passage ID from vertex index."""
        return self._idx_to_passage[vertex_idx]

    def get_entity_index(self, entity_id: int) -> Optional[int]:
        """Get vertex index for entity ID, or None if not found."""
        return self._entity_idx.get(entity_id)

    def get_passage_index(self, passage_id: int) -> Optional[int]:
        """Get vertex index for passage ID, or None if not found."""
        return self._passage_idx.get(passage_id)

    def get_entity_ids(self) -> list[int]:
        """Get all entity IDs in the graph."""
        return list(self._entity_idx.keys())

    def get_passage_ids(self) -> list[int]:
        """Get all passage IDs in the graph."""
        return list(self._passage_idx.keys())

    def get_entity_passages(self, entity_id: int) -> set[int]:
        """Get passages containing the given entity.

        Args:
            entity_id: Database entity ID.

        Returns:
            Set of passage IDs containing this entity.
        """
        passages = set()
        for eid, pid in self._contain_edges:
            if eid == entity_id:
                passages.add(pid)
        return passages

    def get_passage_entities(self, passage_id: int) -> set[int]:
        """Get entities mentioned in the given passage.

        Args:
            passage_id: Database passage ID.

        Returns:
            Set of entity IDs mentioned in this passage.
        """
        entities = set()
        for eid, pid in self._contain_edges:
            if pid == passage_id:
                entities.add(eid)
        return entities

    def get_adjacent_passages(self, passage_id: int) -> set[int]:
        """Get passages adjacent to the given passage.

        Args:
            passage_id: Database passage ID.

        Returns:
            Set of adjacent passage IDs.
        """
        adjacent = set()
        for pid_a, pid_b in self._adjacency_edges:
            if pid_a == passage_id:
                adjacent.add(pid_b)
            elif pid_b == passage_id:
                adjacent.add(pid_a)
        return adjacent

    # --- Internal ---

    def _invalidate_cache(self) -> None:
        """Invalidate cached matrices."""
        self._contain_matrix = None
        self._adjacency_matrix = None
        self._unified_adjacency = None
