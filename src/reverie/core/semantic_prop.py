"""SemanticPropagation for MemoryLink v2.

LinearRAG Stage 1: Entity activation propagation from query.

Supports two modes:
1. Query embedding mode (legacy): Compare query embedding to entity embeddings
2. Query entities mode (recommended): Extract entities from query via NER,
   then match against graph entities by embedding similarity
"""

import logging
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SemanticPropagation:
    """Entity activation propagation (LinearRAG Stage 1).

    Propagates relevance from query to entities through passages.
    Only entities above threshold τ receive propagation.

    Algorithm:
    1. Initial entity activation based on query-entity similarity (τ threshold)
       - If query_entity_embeddings provided: match query entities to graph entities
       - Else: direct query embedding vs entity embedding comparison (legacy)
    2. Compute query-passage relevance σ
    3. Iterate: a_new = max(Cᵀσ ⊙ a, a)
    4. Dynamic pruning (δ threshold)
    """

    def __init__(
        self,
        entity_embeddings: np.ndarray,
        passage_embeddings: np.ndarray,
        contain_matrix: csr_matrix,
        tau: float = 0.5,
        delta: float = 0.1,
        max_iter: int = 2,
    ):
        """Initialize SemanticPropagation.

        Args:
            entity_embeddings: Entity embedding matrix (n_entities, dim).
            passage_embeddings: Passage embedding matrix (n_passages, dim).
            contain_matrix: Contain matrix C (n_passages, n_entities).
                C[p, e] = 1 if entity e appears in passage p.
            tau: Entity activation threshold. Entities with query similarity
                 above τ are initially activated.
            delta: Pruning threshold. Activations below δ are set to zero.
            max_iter: Maximum propagation iterations.
        """
        self._entity_embeddings = entity_embeddings
        self._passage_embeddings = passage_embeddings
        self._contain_matrix = contain_matrix
        self._tau = tau
        self._delta = delta
        self._max_iter = max_iter

    def propagate(
        self,
        query_embedding: np.ndarray,
        query_entity_embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Propagate entity activations from query.

        Args:
            query_embedding: Query embedding vector (dim,) or (1, dim).
            query_entity_embeddings: Optional entity embeddings extracted from query
                via NER. Shape (n_query_entities, dim). If provided, uses entity-to-entity
                matching instead of query-to-entity matching for seed selection.

        Returns:
            Entity activation weights (n_entities,).
            Higher values indicate more relevant entities.
        """
        n_entities = self._entity_embeddings.shape[0]

        if n_entities == 0:
            return np.array([])

        # Ensure query is 2D for cosine_similarity
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Step 1: Initial entity activation
        if query_entity_embeddings is not None and len(query_entity_embeddings) > 0:
            # Entity-to-entity matching: compare query entities to graph entities
            a = self._compute_entity_activations(query_entity_embeddings)
        else:
            # Legacy: direct query embedding vs entity embedding comparison
            entity_sims = cosine_similarity(query_embedding, self._entity_embeddings)[0]
            a = np.where(entity_sims > self._tau, entity_sims, 0.0)

        n_passages = self._passage_embeddings.shape[0]
        if n_passages == 0:
            # No passages to propagate through
            a[a < self._delta] = 0.0
            return a

        # Step 2: Compute query-passage relevance σ
        sigma = cosine_similarity(query_embedding, self._passage_embeddings)[0]

        # Step 3: Propagation iterations
        for _ in range(self._max_iter):
            # Cᵀ · σ: passage-weighted entity relevance
            # contain_matrix: (n_passages, n_entities)
            # sigma: (n_passages,)
            # Cᵀ @ sigma: (n_entities,)
            passage_weighted = self._contain_matrix.T @ sigma

            # Take max of current activation and passage-propagated scores
            # Only activated entities (a > 0) receive propagation (element-wise product)
            a_new = np.maximum(passage_weighted * a, a)

            # Dynamic pruning
            a_new[a_new < self._delta] = 0.0

            # Check convergence
            if np.allclose(a_new, a):
                break

            a = a_new

        return a

    def _compute_entity_activations(
        self, query_entity_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute initial entity activations from query entity embeddings.

        For each graph entity, find the maximum similarity to any query entity.
        This allows matching synonyms or variations (e.g., "api server" and "api-server").

        Args:
            query_entity_embeddings: Query entity embeddings (n_query_entities, dim).

        Returns:
            Entity activation weights (n_entities,).
        """
        # Ensure 2D
        if query_entity_embeddings.ndim == 1:
            query_entity_embeddings = query_entity_embeddings.reshape(1, -1)

        # Compute similarity: (n_query_entities, n_graph_entities)
        sims = cosine_similarity(query_entity_embeddings, self._entity_embeddings)

        # For each graph entity, take max similarity across all query entities
        max_sims = np.max(sims, axis=0)

        # Apply threshold
        activations = np.where(max_sims > self._tau, max_sims, 0.0)

        logger.debug(
            "Entity activations: %d query entities -> %d graph entities activated (tau=%.2f)",
            len(query_entity_embeddings),
            np.sum(activations > 0),
            self._tau,
        )

        return activations
