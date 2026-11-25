"""BipartitePPR for MemoryLink v2.

LinearRAG Stage 2: Personalized PageRank on bipartite graph.

DEPRECATED: This module is deprecated in favor of LazyBipartitePPR.
LazyBipartitePPR provides the same functionality without loading
the full graph into memory. This module is kept for:
1. Reference implementation for equivalence testing
2. Backward compatibility

Use LazyBipartitePPR for production:
    from reverie.core.lazy_bipartite_ppr import LazyBipartitePPR
"""

import logging

import numpy as np
from scipy.sparse import csr_matrix

from reverie.core.bipartite import BipartiteGraph

logger = logging.getLogger(__name__)


def row_normalize(matrix: csr_matrix) -> csr_matrix:
    """Row-normalize a sparse matrix.

    Each row sums to 1 (or 0 if the row was all zeros).

    Args:
        matrix: Sparse matrix to normalize.

    Returns:
        Row-normalized sparse matrix.
    """
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    inv_row_sums = 1.0 / row_sums
    # Create diagonal matrix
    from scipy.sparse import diags
    D_inv = diags(inv_row_sums)
    return D_inv @ matrix


class BipartitePPR:
    """Personalized PageRank on bipartite graph (LinearRAG Stage 2).

    Runs PPR on the unified adjacency matrix to propagate entity activations
    to passages through the graph structure.

    Convention: r = (1-d)·v + d·Ã·r
    - d = damping factor (default 0.85)
    - d가 클수록 이웃으로 더 많이 전파 (85% to neighbors, 15% reset)
    - d가 작을수록 seed(활성화된 entity)에 집중

    The unified adjacency matrix Ã has structure:
           Entity    Passage
    Entity [  0    |   Cᵀ   ]
    Passage[  C    |   A    ]

    Where:
    - C is the contain matrix (passages × entities)
    - A is the passage adjacency matrix (passages × passages)
    """

    def __init__(
        self,
        graph: BipartiteGraph,
        damping: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-6,
    ):
        """Initialize BipartitePPR.

        Args:
            graph: BipartiteGraph instance.
            damping: Damping factor d. Higher values spread more to neighbors.
                     Default 0.85 (standard PageRank value).
            max_iter: Maximum PPR iterations.
            tol: Convergence tolerance.
        """
        self._graph = graph
        self._damping = damping
        self._max_iter = max_iter
        self._tol = tol

        # Lazy build normalized adjacency
        self._normalized_adj: csr_matrix | None = None

    def _ensure_normalized_adj(self) -> csr_matrix:
        """Get row-normalized unified adjacency matrix."""
        if self._normalized_adj is not None:
            return self._normalized_adj

        unified_adj = self._graph.get_unified_adjacency()
        self._normalized_adj = row_normalize(unified_adj)
        return self._normalized_adj

    def run(
        self,
        entity_activations: np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Run Personalized PageRank.

        Args:
            entity_activations: Entity activation weights from Stage 1.
                Shape (n_entities,).
            top_k: Number of top passages to return.

        Returns:
            List of (passage_id, score) tuples, sorted by score descending.
        """
        n_entities = self._graph.entity_count
        n_passages = self._graph.passage_count
        n_total = n_entities + n_passages

        if n_passages == 0:
            return []

        if n_entities == 0:
            # No entities, return passages with zero scores
            passage_ids = self._graph.get_passage_ids()
            return [(pid, 0.0) for pid in passage_ids[:top_k]]

        # Build reset vector v: entity part gets activation, passage part is zero
        v = np.zeros(n_total)
        activation_sum = entity_activations.sum()
        if activation_sum > 0:
            v[:n_entities] = entity_activations / activation_sum
        else:
            # No activation, return passages with zero scores
            passage_ids = self._graph.get_passage_ids()
            return [(pid, 0.0) for pid in passage_ids[:top_k]]

        # Get normalized adjacency matrix
        Ã_norm = self._ensure_normalized_adj()

        # PPR iteration: r = (1-d)·v + d·Ã·r
        r = v.copy()
        d = self._damping

        for _ in range(self._max_iter):
            r_new = (1 - d) * v + d * (Ã_norm.T @ r)

            if np.allclose(r_new, r, rtol=self._tol):
                break

            r = r_new

        # Extract passage scores (second half of r)
        passage_scores = r[n_entities:]

        # Get top-k passages
        if top_k >= n_passages:
            top_indices = np.argsort(passage_scores)[::-1]
        else:
            top_indices = np.argsort(passage_scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            passage_id = self._graph.get_passage_id(idx)
            score = passage_scores[idx]
            results.append((passage_id, float(score)))

        return results

    def run_with_graph(
        self,
        entity_activations: np.ndarray,
        top_k_passages: int = 10,
        score_threshold: float = 0.0001,
    ) -> dict:
        """Run PPR and return full traversal graph for visualization.

        Args:
            entity_activations: Entity activation weights from Stage 1.
            top_k_passages: Number of top passages to return as final results.
            score_threshold: Minimum PPR score to include a node in the graph.

        Returns:
            Dict with:
            - passages: List of (passage_id, score) for top-k results
            - graph_nodes: List of {id, type, score} for all traversed nodes
            - graph_edges: List of {source, target, type} for edges between nodes
        """
        n_entities = self._graph.entity_count
        n_passages = self._graph.passage_count
        n_total = n_entities + n_passages

        if n_passages == 0 or n_entities == 0:
            return {"passages": [], "graph_nodes": [], "graph_edges": []}

        # Build reset vector
        v = np.zeros(n_total)
        activation_sum = entity_activations.sum()
        if activation_sum > 0:
            v[:n_entities] = entity_activations / activation_sum
        else:
            return {"passages": [], "graph_nodes": [], "graph_edges": []}

        # PPR iteration
        Ã_norm = self._ensure_normalized_adj()
        r = v.copy()
        d = self._damping

        for _ in range(self._max_iter):
            r_new = (1 - d) * v + d * (Ã_norm.T @ r)
            if np.allclose(r_new, r, rtol=self._tol):
                break
            r = r_new

        # Extract scores
        entity_scores = r[:n_entities]
        passage_scores = r[n_entities:]

        # Get top-k passages for final results
        top_indices = np.argsort(passage_scores)[-top_k_passages:][::-1]
        passages = [
            (self._graph.get_passage_id(idx), float(passage_scores[idx]))
            for idx in top_indices
        ]

        # Build graph nodes: all nodes above threshold
        graph_nodes = []
        included_entity_indices = set()
        included_passage_indices = set()

        for idx in range(n_entities):
            if entity_scores[idx] >= score_threshold:
                eid = self._graph.get_entity_id(idx)
                graph_nodes.append({
                    "id": f"entity:{eid}",
                    "entity_id": eid,
                    "type": "entity",
                    "score": float(entity_scores[idx]),
                })
                included_entity_indices.add(idx)

        for idx in range(n_passages):
            if passage_scores[idx] >= score_threshold:
                pid = self._graph.get_passage_id(idx)
                graph_nodes.append({
                    "id": f"passage:{pid}",
                    "passage_id": pid,
                    "type": "passage",
                    "score": float(passage_scores[idx]),
                })
                included_passage_indices.add(idx)

        # Build graph edges: edges between included nodes
        graph_edges = []
        contain_matrix = self._graph.get_contain_matrix()  # (passages, entities)
        adjacency_matrix = self._graph.get_adjacency_matrix()  # (passages, passages)

        # Entity-Passage edges (contain)
        for pidx in included_passage_indices:
            row = contain_matrix.getrow(pidx)
            for eidx in row.indices:
                if eidx in included_entity_indices:
                    pid = self._graph.get_passage_id(pidx)
                    eid = self._graph.get_entity_id(eidx)
                    graph_edges.append({
                        "source": f"entity:{eid}",
                        "target": f"passage:{pid}",
                        "type": "contain",
                    })

        # Passage-Passage edges (adjacency)
        for pidx in included_passage_indices:
            row = adjacency_matrix.getrow(pidx)
            for pidx2 in row.indices:
                if pidx2 in included_passage_indices and pidx < pidx2:
                    pid1 = self._graph.get_passage_id(pidx)
                    pid2 = self._graph.get_passage_id(pidx2)
                    graph_edges.append({
                        "source": f"passage:{pid1}",
                        "target": f"passage:{pid2}",
                        "type": "adjacency",
                    })

        return {
            "passages": passages,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
        }
