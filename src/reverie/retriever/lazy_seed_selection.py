"""Lazy Seed Selection for Lazy PPR.

Progressive Widening algorithm to find all entity seeds above tau threshold
without loading all entity embeddings into memory.

This is Phase 3 of the Lazy PPR implementation - mathematically equivalent
to the in-memory cosine similarity computation in SemanticPropagation.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from reverie.db.reverie_db import ReverieDB

logger = logging.getLogger(__name__)


@dataclass
class SeedSelectionResult:
    """Result of lazy seed selection."""

    seeds: dict[int, float]  # entity_id -> similarity score
    query_count: int  # Number of DB queries made
    total_candidates: int  # Total candidates examined
    # Detailed timing breakdown (ms)
    timing_details: dict[str, float] = field(default_factory=dict)


class LazySeedSelector:
    """Lazy seed selection using Progressive Widening.

    Replaces the in-memory cosine similarity computation in SemanticPropagation
    with a database-backed approach that leverages pgvector indexes.

    Supports two modes:
    1. Query embedding mode: Compare query embedding directly to entity embeddings
    2. Query entities mode: Extract entities from query via NER, then match
       against graph entities by embedding similarity (max aggregation)
    """

    def __init__(
        self,
        db: "ReverieDB",
        tau: float = 0.5,
        initial_k: int = 100,
        max_k: int = 100000,
    ):
        """Initialize LazySeedSelector.

        Args:
            db: ReverieDB database interface.
            tau: Similarity threshold. Entities with cosine similarity >= tau
                 are selected as seeds.
            initial_k: Initial batch size for Progressive Widening.
            max_k: Maximum batch size to prevent runaway queries.
        """
        self._db = db
        self._tau = tau
        self._initial_k = initial_k
        self._max_k = max_k

    def select_seeds(
        self,
        query_embedding: np.ndarray,
        query_entity_embeddings: Optional[np.ndarray] = None,
    ) -> SeedSelectionResult:
        """Select entity seeds above tau threshold.

        Args:
            query_embedding: Query embedding vector (dim,) or (1, dim).
                Used for direct query-to-entity matching when
                query_entity_embeddings is not provided.
            query_entity_embeddings: Optional entity embeddings extracted from
                query via NER. Shape (n_query_entities, dim). If provided,
                uses entity-to-entity matching with max aggregation.

        Returns:
            SeedSelectionResult containing seeds dict and metadata.
        """
        if query_entity_embeddings is not None and len(query_entity_embeddings) > 0:
            return self._select_seeds_entity_mode(query_entity_embeddings)
        else:
            return self._select_seeds_query_mode(query_embedding)

    def _select_seeds_query_mode(
        self, query_embedding: np.ndarray
    ) -> SeedSelectionResult:
        """Select seeds using direct query embedding comparison.

        Args:
            query_embedding: Query embedding vector.

        Returns:
            SeedSelectionResult with entity_id -> similarity mapping.
        """
        t0 = time.time()
        timing_details: dict[str, float] = {}

        # Ensure 1D
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()

        # Use Progressive Widening via EntityRepository
        results = self._db.find_entity_seeds(
            query_embedding=query_embedding.tolist(),
            tau=self._tau,
            initial_k=self._initial_k,
            max_k=self._max_k,
        )

        seeds = {entity_id: similarity for entity_id, similarity in results}

        elapsed_ms = (time.time() - t0) * 1000
        timing_details["mode"] = "query"
        timing_details["total_ms"] = round(elapsed_ms, 1)
        timing_details["num_seeds_found"] = len(seeds)
        timing_details["db_query_ms"] = round(elapsed_ms, 1)

        logger.info(
            "Lazy seed selection (query mode): %d seeds above tau=%.2f in %.1fms",
            len(seeds),
            self._tau,
            elapsed_ms,
        )

        return SeedSelectionResult(
            seeds=seeds,
            query_count=1,
            total_candidates=len(results),
            timing_details=timing_details,
        )

    def _select_seeds_entity_mode(
        self, query_entity_embeddings: np.ndarray
    ) -> SeedSelectionResult:
        """Select seeds using entity-to-entity matching.

        For each query entity, find matching graph entities and take
        max similarity across all query entities (same as SemanticPropagation).

        Uses parallel DB queries for better performance.

        Args:
            query_entity_embeddings: Query entity embeddings (n_query_entities, dim).

        Returns:
            SeedSelectionResult with entity_id -> max_similarity mapping.
        """
        total_start = time.time()
        timing_details: dict[str, float] = {}

        # Ensure 2D
        if query_entity_embeddings.ndim == 1:
            query_entity_embeddings = query_entity_embeddings.reshape(1, -1)

        n_entities = len(query_entity_embeddings)

        # Parallel DB queries for each query entity
        def find_seeds_for_entity(idx_and_embedding: tuple[int, np.ndarray]) -> tuple[int, float, list[tuple[int, float]]]:
            """Find seeds for a single query entity. Returns (idx, elapsed_ms, results)."""
            idx, qe_embedding = idx_and_embedding
            t0 = time.time()
            results = self._db.find_entity_seeds(
                query_embedding=qe_embedding.tolist(),
                tau=self._tau,
                initial_k=self._initial_k,
                max_k=self._max_k,
            )
            elapsed_ms = (time.time() - t0) * 1000
            return (idx, elapsed_ms, results)

        # Run parallel queries (max 8 workers to avoid connection pool exhaustion)
        entity_max_sims: dict[int, float] = {}
        total_candidates = 0
        per_entity_times: list[float] = [0.0] * n_entities

        with ThreadPoolExecutor(max_workers=min(n_entities, 8)) as executor:
            indexed_embeddings = list(enumerate(query_entity_embeddings))
            results_list = list(executor.map(find_seeds_for_entity, indexed_embeddings))

        # Aggregate results (max similarity per entity)
        for idx, elapsed_ms, results in results_list:
            per_entity_times[idx] = elapsed_ms
            total_candidates += len(results)

            for entity_id, similarity in results:
                if entity_id not in entity_max_sims:
                    entity_max_sims[entity_id] = similarity
                else:
                    entity_max_sims[entity_id] = max(
                        entity_max_sims[entity_id], similarity
                    )

        total_elapsed = (time.time() - total_start) * 1000

        # Build timing details
        timing_details["mode"] = "entity"
        timing_details["parallel"] = True
        timing_details["total_ms"] = round(total_elapsed, 1)
        timing_details["num_query_entities"] = n_entities
        timing_details["num_seeds_found"] = len(entity_max_sims)
        timing_details["total_candidates"] = total_candidates
        if per_entity_times:
            timing_details["per_entity_avg_ms"] = round(sum(per_entity_times) / len(per_entity_times), 1)
            timing_details["per_entity_max_ms"] = round(max(per_entity_times), 1)
            timing_details["per_entity_min_ms"] = round(min(per_entity_times), 1)
            # Log individual entity times for debugging
            for i, t in enumerate(per_entity_times):
                timing_details[f"entity_{i}_ms"] = round(t, 1)

        logger.info(
            "Lazy seed selection (entity mode, parallel): %d query entities -> %d seeds, "
            "total=%.1fms, max_per_entity=%.1fms, candidates=%d",
            n_entities,
            len(entity_max_sims),
            total_elapsed,
            timing_details.get("per_entity_max_ms", 0),
            total_candidates,
        )

        return SeedSelectionResult(
            seeds=entity_max_sims,
            query_count=n_entities,
            total_candidates=total_candidates,
            timing_details=timing_details,
        )

    def to_activation_array(
        self,
        seeds: dict[int, float],
        entity_id_to_index: dict[int, int],
        n_entities: int,
    ) -> np.ndarray:
        """Convert seeds dict to activation array format.

        For compatibility with existing SemanticPropagation interface.

        Args:
            seeds: Entity ID to similarity mapping from select_seeds().
            entity_id_to_index: Mapping from entity ID to array index.
            n_entities: Total number of entities.

        Returns:
            Numpy array of shape (n_entities,) with activations.
        """
        activations = np.zeros(n_entities)

        for entity_id, similarity in seeds.items():
            if entity_id in entity_id_to_index:
                idx = entity_id_to_index[entity_id]
                activations[idx] = similarity

        return activations
