"""MemoryLinkV2Pipeline for MemoryLink v2.

Fully Lazy retrieval pipeline - no full graph loading.

Uses:
- LazySeedSelector for Progressive Widening seed selection (no entity embedding load)
- LazyBipartitePPR for DB-backed PPR iteration (no graph matrix load)

Seed Entity Selection:
- Extracts named entities from query using NER (same as indexing)
- Matches query entities to graph entities by embedding similarity via pgvector
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import numpy as np

from reverie.core.lazy_bipartite_ppr import LazyBipartitePPR, LazyPPRResult
from reverie.retriever.lazy_seed_selection import LazySeedSelector
from reverie.retriever.query_ner import QueryNERClient
from reverie.retriever.timing import StageTimer, StageTiming

if TYPE_CHECKING:
    from reverie.db.reverie_db import ReverieDB

logger = logging.getLogger(__name__)

# Pipeline constants
KNN_FETCH_MULTIPLIER = 2  # Fetch 2x top_k for better ranking after merge


def calculate_time_decay(
    created_at: datetime,
    half_life_days: float,
    reference_time: Optional[datetime] = None,
) -> float:
    """Calculate exponential time decay weight.

    Uses the formula: weight = exp(-ln(2) * age_days / half_life_days)
    This gives a half-life decay where the weight is 0.5 after half_life_days.

    Args:
        created_at: Timestamp of the passage creation.
        half_life_days: Number of days for weight to decay to 0.5.
        reference_time: Reference time for age calculation. Defaults to now.

    Returns:
        Decay weight between 0 and 1.
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    # Ensure created_at is timezone-aware
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    age_seconds = (reference_time - created_at).total_seconds()
    age_days = age_seconds / 86400  # seconds per day

    # Prevent negative age (future timestamps)
    if age_days < 0:
        return 1.0

    decay_rate = math.log(2) / half_life_days
    return math.exp(-decay_rate * age_days)


@dataclass
class RetrievalResult:
    """Result from retrieval pipeline."""

    passage_id: int
    text: str
    score: float
    source: str  # "memorylink-v2" or "knn"
    session_id: Optional[str] = None
    session_seq: Optional[int] = None
    turn_index: Optional[int] = None
    # Time decay metadata
    original_score: Optional[float] = None
    decay_weight: Optional[float] = None
    created_at: Optional[datetime] = None


@dataclass
class RetrievalResponse:
    """Response from retrieval pipeline with timing information.

    Attributes:
        results: List of retrieval results.
        timing: Stage timing breakdown.
    """

    results: list[RetrievalResult]
    timing: StageTiming


class MemoryLinkV2Pipeline:
    """MemoryLink v2 retrieval pipeline - Fully Lazy.

    No full graph loading. All operations use database queries with indexes.

    Pipeline stages:
    1. Embed query
    2. Lazy Seed Selection: Find entities above tau using Progressive Widening
    3. Lazy PPR: DB-backed PPR iteration
    4. Hybrid Merge: Combine PPR and KNN results
    """

    def __init__(
        self,
        db: "ReverieDB",
        embedding_client,
        # Hyperparameters
        tau: float = 0.5,
        damping: float = 0.85,
        beta: float = 1.0,
        # PPR tuning
        ppr_max_iter: int = 10,
        ppr_min_score: float = 0.0001,
        use_server_side_ppr: bool = False,
        # Query NER
        query_ner_client: Optional[QueryNERClient] = None,
        use_query_ner: bool = True,
        # Dual embedding KNN
        use_dual_knn: bool = True,
        # Time decay
        half_life_days: Optional[float] = None,
        # Deprecated parameters (ignored, for backward compatibility)
        conn=None,  # DEPRECATED - use db
        delta: float = 0.1,
        max_iter: int = 2,
        graph_ttl_seconds: int = 300,
    ):
        """Initialize MemoryLinkV2Pipeline.

        Args:
            db: ReverieDB database interface.
            embedding_client: Client for computing embeddings.
            tau: Entity activation threshold for seed selection.
            damping: PPR damping factor.
            beta: PPR/KNN hybrid weight (0=KNN only, 1=PPR only).
            ppr_max_iter: Maximum PPR iterations.
            ppr_min_score: Minimum score threshold for PPR node pruning.
                Lower values (e.g., 1e-4) give richer results but slower PPR.
                Higher values (e.g., 1e-3) give faster PPR but less diversity.
            use_server_side_ppr: Use PostgreSQL server-side PPR function for
                better performance. Requires run_lazy_ppr function to be installed.
            query_ner_client: Optional QueryNERClient for entity extraction.
                If None and use_query_ner is True, creates one with defaults.
            use_query_ner: Whether to use Query NER for seed selection.
            use_dual_knn: Whether to use dual embedding (user/assistant) KNN.
            half_life_days: Optional time decay half-life in days.
            conn: DEPRECATED - use db parameter instead.
            delta: DEPRECATED - ignored (was for SemanticPropagation pruning).
            max_iter: DEPRECATED - ignored (was for SemanticPropagation iterations).
            graph_ttl_seconds: DEPRECATED - ignored (no graph cache needed).
        """
        self._db = db
        self._embedding_client = embedding_client
        self._tau = tau
        self._damping = damping
        self._beta = beta
        self._ppr_max_iter = ppr_max_iter
        self._ppr_min_score = ppr_min_score
        self._use_server_side_ppr = use_server_side_ppr
        self._use_query_ner = use_query_ner
        self._use_dual_knn = use_dual_knn
        self._half_life_days = half_life_days

        # Query NER (lazy init if needed)
        self._query_ner_client = query_ner_client

        # Lazy components
        self._seed_selector: Optional[LazySeedSelector] = None
        self._ppr: Optional[LazyBipartitePPR] = None

        # Last retrieval state (for debugging/visualization)
        self._last_entity_seeds: Optional[dict[int, float]] = None
        self._last_query_entities: Optional[list[str]] = None
        # Cache last PPR result to avoid re-running for visualization
        self._last_ppr_result: Optional[LazyPPRResult] = None

    @property
    def embedding_client(self):
        """Get embedding client (for external use like API routes)."""
        return self._embedding_client

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve relevant passages for query.

        Args:
            query: Query string.
            top_k: Number of passages to return.

        Returns:
            List of RetrievalResult.
        """
        response = self.retrieve_with_timing(query, top_k)
        return response.results

    def retrieve_with_timing(
        self,
        query: str,
        top_k: int = 10,
        ppr_min_score: Optional[float] = None,
        ppr_max_iter: Optional[int] = None,
    ) -> RetrievalResponse:
        """Retrieve relevant passages for query with timing breakdown.

        Args:
            query: Query string.
            top_k: Number of passages to return.
            ppr_min_score: Override PPR min_score for this query only.
            ppr_max_iter: Override PPR max_iter for this query only (10-50).

        Returns:
            RetrievalResponse with results and timing information.
        """
        timer = StageTimer()
        timer.set_cache_hit(True)  # Fully lazy - no graph cache

        # Ensure repositories are initialized
        self._ensure_components()

        # Lazy init query NER client
        if self._use_query_ner and self._query_ner_client is None:
            self._query_ner_client = QueryNERClient()

        # Run embed_query and ner_extract in parallel (no dependencies)
        query_emb = None
        query_entities_extracted: list[str] = []

        def do_embed():
            return np.array(self._embedding_client.embed_text(query))

        def do_ner():
            if self._use_query_ner:
                return self._query_ner_client.extract_entities(query)
            return []

        with timer.stage("embed_query"):
            with ThreadPoolExecutor(max_workers=2) as executor:
                embed_future = executor.submit(do_embed)
                ner_future = executor.submit(do_ner)

                query_emb = embed_future.result()
                query_entities_extracted = ner_future.result()
                self._last_query_entities = query_entities_extracted

        # Record NER time separately for timing breakdown
        # (already included in embed_query stage since parallel)

        # Get KNN results (always needed for hybrid)
        with timer.stage("knn_search"):
            if self._use_dual_knn:
                knn_results = self._passage_dual_knn(query_emb, top_k * KNN_FETCH_MULTIPLIER)
                if not knn_results:
                    knn_results = self._passage_knn(query_emb, top_k * KNN_FETCH_MULTIPLIER)
            else:
                knn_results = self._passage_knn(query_emb, top_k * KNN_FETCH_MULTIPLIER)

        # Compute embeddings for extracted entities (if any) - PARALLEL
        query_entity_embeddings = None
        if query_entities_extracted:
            with timer.stage("ner_embed"):
                # Parallel embedding for query entities
                with ThreadPoolExecutor(max_workers=min(len(query_entities_extracted), 8)) as executor:
                    futures = [
                        executor.submit(self._embedding_client.embed_text, entity)
                        for entity in query_entities_extracted
                    ]
                    embeddings = [f.result() for f in futures]
                query_entity_embeddings = np.array(embeddings)

            logger.info(
                "Query NER: %d entities extracted, embeddings computed: %s",
                len(query_entities_extracted),
                query_entities_extracted[:5],
            )

        # Stage 1: Lazy Seed Selection (Progressive Widening)
        with timer.stage("semantic_prop"):  # Keep name for timing compatibility
            seed_result = self._seed_selector.select_seeds(
                query_emb,
                query_entity_embeddings=query_entity_embeddings,
            )
            entity_seeds = seed_result.seeds
            self._last_entity_seeds = entity_seeds

        # Pass detailed timing if available
        if seed_result.timing_details:
            timer.set_seed_selection_details(seed_result.timing_details)

        # Stage 2: Lazy PPR (only if entities are activated)
        ppr_results = []
        if entity_seeds:
            # Use per-query overrides if provided
            if ppr_min_score is not None or ppr_max_iter is not None:
                ppr = LazyBipartitePPR(
                    db=self._db,
                    damping=self._damping,
                    max_iter=ppr_max_iter if ppr_max_iter is not None else self._ppr_max_iter,
                    min_score=ppr_min_score if ppr_min_score is not None else self._ppr_min_score,
                    use_server_side=self._use_server_side_ppr,
                )
            else:
                ppr = self._ppr

            with timer.stage("ppr"):
                ppr_full_result = ppr.run_full(entity_seeds)
            # Cache PPR result for get_traversal_graph() to reuse
            self._last_ppr_result = ppr_full_result

            # Record PPR timing details
            if ppr_full_result.timing_details:
                td = ppr_full_result.timing_details
                # Per-iteration stages for Python PPR
                for i, (db_ms, prop_ms) in enumerate(td.per_iter_timings):
                    timer._stages[f"ppr_i{i}_db"] = db_ms
                    timer._stages[f"ppr_i{i}_prop"] = prop_ms
                # PPR summary details (for server-side PPR display)
                timer.set_ppr_details({
                    "iterations": td.iterations,
                    "db_query_ms": td.db_query_ms,
                    "propagation_ms": td.propagation_ms,
                    "per_iter_avg_ms": td.per_iter_avg_ms,
                    "total_ms": td.total_ms,
                    "server_side": len(td.per_iter_timings) == 0,  # No per-iter = server-side
                })

            # Sort passages by score and get top-k*2
            sorted_passages = sorted(
                ppr_full_result.passage_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            ppr_results = sorted_passages[:top_k * KNN_FETCH_MULTIPLIER]

        # Hybrid merge
        if ppr_results:
            with timer.stage("hybrid_merge"):
                merged = self._hybrid_merge(ppr_results, knn_results, self._beta)
            source = "memorylink-v2"
        else:
            merged = knn_results
            source = "knn"

        with timer.stage("load_passages"):
            results = self._load_passages(merged[:top_k], source=source)

        return RetrievalResponse(results=results, timing=timer.get_timing())

    def _ensure_components(self) -> None:
        """Ensure lazy components are initialized."""
        if self._seed_selector is None:
            self._seed_selector = LazySeedSelector(
                db=self._db,
                tau=self._tau,
            )

        if self._ppr is None:
            self._ppr = LazyBipartitePPR(
                db=self._db,
                damping=self._damping,
                max_iter=self._ppr_max_iter,
                min_score=self._ppr_min_score,
                use_server_side=self._use_server_side_ppr,
            )

    def _extract_query_entity_embeddings(
        self, query: str
    ) -> Optional[np.ndarray]:
        """Extract entities from query and compute their embeddings.

        Args:
            query: Query string.

        Returns:
            Entity embeddings (n_entities, dim) or None if no entities found.
        """
        # Lazy init query NER client
        if self._query_ner_client is None:
            self._query_ner_client = QueryNERClient()

        # Extract entities from query
        query_entities = self._query_ner_client.extract_entities(query)
        self._last_query_entities = query_entities

        if not query_entities:
            logger.debug("No entities extracted from query: %.50s...", query)
            return None

        logger.info(
            "Query NER extracted %d entities: %s",
            len(query_entities),
            query_entities[:5],  # Log first 5
        )

        # Compute embeddings for query entities
        embeddings = []
        for entity in query_entities:
            emb = self._embedding_client.embed_text(entity)
            embeddings.append(emb)

        return np.array(embeddings)

    def _passage_knn(
        self, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[int, float]]:
        """Get passages by KNN similarity.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results.

        Returns:
            List of (passage_id, similarity_score).
        """
        results = self._db.search_passages_by_embedding(
            query_embedding.tolist(), top_k
        )

        # Convert distance to similarity (cosine distance: 0=identical, 2=opposite)
        return [(p.id, 1.0 - d / 2.0) for p, d in results]

    def _passage_dual_knn(
        self, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[int, float]]:
        """Get passages using dual embedding (user/assistant) harmonic mean.

        Uses the search_passages_dual_embedding() database function which
        computes harmonic mean of user and assistant embedding similarities.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results.

        Returns:
            List of (passage_id, combined_score).
        """
        results = self._db.search_passages_dual_embedding(
            query_embedding.tolist(), top_k
        )

        return [(r.passage_id, r.combined_score) for r in results]

    def _hybrid_merge(
        self,
        ppr_results: list[tuple[int, float]],
        knn_results: list[tuple[int, float]],
        beta: float,
    ) -> list[tuple[int, float]]:
        """Merge PPR and KNN results.

        Args:
            ppr_results: List of (passage_id, ppr_score).
            knn_results: List of (passage_id, knn_score).
            beta: PPR weight (0=KNN only, 1=PPR only).

        Returns:
            List of (passage_id, merged_score), sorted descending.
        """
        if not ppr_results:
            return knn_results

        if not knn_results:
            return ppr_results

        # Build score dicts
        ppr_scores = {pid: score for pid, score in ppr_results}
        knn_scores = {pid: score for pid, score in knn_results}

        all_pids = set(ppr_scores.keys()) | set(knn_scores.keys())

        # Normalize scores to 0-1 range
        ppr_max = max(ppr_scores.values()) if ppr_scores else 1.0
        knn_max = max(knn_scores.values()) if knn_scores else 1.0

        merged = []
        for pid in all_pids:
            ppr_norm = ppr_scores.get(pid, 0) / ppr_max if ppr_max > 0 else 0
            knn_norm = knn_scores.get(pid, 0) / knn_max if knn_max > 0 else 0
            final = beta * ppr_norm + (1 - beta) * knn_norm
            merged.append((pid, final))

        # Sort by score descending
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    def _load_passages(
        self, results: list[tuple[int, float]], source: str
    ) -> list[RetrievalResult]:
        """Load passage text for results and apply time decay if configured.

        Args:
            results: List of (passage_id, score).
            source: Source algorithm name.

        Returns:
            List of RetrievalResult, re-sorted by final score if decay is applied.
        """
        if not results:
            return []

        loaded = []
        reference_time = datetime.now(timezone.utc)

        # Batch load all passages in a single query (N+1 -> 1 optimization)
        passage_ids = [pid for pid, _ in results]
        passages_map = {p.id: p for p in self._db.get_passages_by_ids(passage_ids)}

        for passage_id, original_score in results:
            passage = passages_map.get(passage_id)
            if passage:
                # Apply time decay if configured
                if self._half_life_days is not None and passage.created_at:
                    decay_weight = calculate_time_decay(
                        passage.created_at,
                        self._half_life_days,
                        reference_time,
                    )
                    final_score = original_score * decay_weight
                else:
                    decay_weight = None
                    final_score = original_score

                loaded.append(RetrievalResult(
                    passage_id=passage_id,
                    text=passage.text,
                    score=final_score,
                    source=source,
                    session_id=passage.session_id,
                    session_seq=passage.session_seq,
                    turn_index=passage.turn_index,
                    original_score=original_score if decay_weight is not None else None,
                    decay_weight=decay_weight,
                    created_at=passage.created_at,
                ))

        # Re-sort by final score if time decay was applied
        if self._half_life_days is not None:
            loaded.sort(key=lambda r: r.score, reverse=True)

        return loaded

    def get_seed_entities(self, top_k: int = 20) -> list[tuple[int, float]]:
        """Get the seed entities from the last retrieval.

        Returns entities that were selected via Lazy Seed Selection.
        Call this after retrieve() to get the seed entities.

        Args:
            top_k: Maximum number of seed entities to return.

        Returns:
            List of (entity_id, activation_score), sorted by score descending.
        """
        if self._last_entity_seeds is None:
            return []

        # Sort by score descending
        sorted_seeds = sorted(
            self._last_entity_seeds.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_seeds[:top_k]

    def get_query_entities(self) -> Optional[list[str]]:
        """Get the query entities extracted from the last retrieval.

        Returns the named entities that were extracted from the query via NER.
        Call this after retrieve() to see what entities were found.

        Returns:
            List of entity names, or None if NER was not used.
        """
        return self._last_query_entities

    def get_traversal_graph(self, score_threshold: float = 0.0001) -> dict | None:
        """Get the PPR traversal graph from the last retrieval.

        Uses cached PPR result from retrieve() - no re-computation needed.
        Returns nodes and edges with scores above the threshold.

        Args:
            score_threshold: Minimum PPR score to include a node.

        Returns:
            Dict with graph_nodes and graph_edges, or None if not available.
        """
        if self._last_ppr_result is None or self._ppr is None:
            return None

        # Build graph from cached PPR result (no re-running PPR)
        result = self._last_ppr_result

        # Sort and get top-k passages
        sorted_passages = sorted(
            result.passage_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        passages = sorted_passages[:10]

        # Build graph nodes (all nodes above threshold)
        graph_nodes = []
        included_entities = set()
        included_passages = set()

        for eid, score in result.entity_scores.items():
            if score >= score_threshold:
                graph_nodes.append({
                    "id": f"entity:{eid}",
                    "entity_id": eid,
                    "type": "entity",
                    "score": float(score),
                })
                included_entities.add(eid)

        for pid, score in result.passage_scores.items():
            if score >= score_threshold:
                graph_nodes.append({
                    "id": f"passage:{pid}",
                    "passage_id": pid,
                    "type": "passage",
                    "score": float(score),
                })
                included_passages.add(pid)

        # Build graph edges between included nodes using DB query
        graph_edges = []
        if included_entities or included_passages:
            edges = self._db.get_ppr_neighbors(
                list(included_entities), list(included_passages)
            )

            seen_edges = set()
            for edge in edges:
                if edge.edge_type == "e2p":
                    if edge.source_id in included_entities and edge.target_id in included_passages:
                        edge_key = (f"entity:{edge.source_id}", f"passage:{edge.target_id}")
                        if edge_key not in seen_edges:
                            graph_edges.append({
                                "source": f"entity:{edge.source_id}",
                                "target": f"passage:{edge.target_id}",
                                "type": "contain",
                            })
                            seen_edges.add(edge_key)

                elif edge.edge_type == "p2e":
                    # p2e creates same edge as e2p, skip to avoid duplicates
                    pass

                elif edge.edge_type in ("p2p_a", "p2p_b"):
                    if edge.source_id in included_passages and edge.target_id in included_passages:
                        # Normalize edge direction to avoid duplicates
                        pid_a, pid_b = min(edge.source_id, edge.target_id), max(edge.source_id, edge.target_id)
                        edge_key = (f"passage:{pid_a}", f"passage:{pid_b}")
                        if edge_key not in seen_edges:
                            graph_edges.append({
                                "source": f"passage:{pid_a}",
                                "target": f"passage:{pid_b}",
                                "type": "adjacency",
                            })
                            seen_edges.add(edge_key)

        return {
            "passages": passages,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "iterations": result.iterations,
            "converged": result.converged,
        }

    def invalidate_graph(self) -> None:
        """Invalidate any cached state.

        In fully lazy mode, this just clears the last retrieval state.
        """
        self._last_entity_seeds = None
        self._last_query_entities = None
        self._last_ppr_result = None
        logger.info("Retrieval state cleared")

    # Backward compatibility properties
    @property
    def _graph(self):
        """Deprecated: Graph is no longer used."""
        return None

    @property
    def _propagation(self):
        """Deprecated: SemanticPropagation is replaced by LazySeedSelector."""
        return None
