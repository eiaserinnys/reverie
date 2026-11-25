"""LazyBipartitePPR for MemoryLink v2.

Lazy PPR implementation that doesn't require loading the full graph into memory.
Instead, it queries neighbors from the database on each iteration.

This is mathematically equivalent to the matrix-based BipartitePPR but:
1. Uses pre-computed out_degree from DB for normalization
2. Queries only active node neighbors per iteration
3. Uses UNION ALL batch query for efficiency (1 query per iteration)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from reverie.db.reverie_db import ReverieDB

logger = logging.getLogger(__name__)


@dataclass
class PPRTimingDetails:
    """Detailed timing breakdown for PPR execution."""

    total_ms: float = 0.0
    iterations: int = 0
    db_query_ms: float = 0.0  # Total time spent in DB queries
    propagation_ms: float = 0.0  # Total time spent in propagation calculation
    per_iter_avg_ms: float = 0.0  # Average time per iteration
    # Per-iteration breakdown: list of (db_ms, prop_ms) tuples
    per_iter_timings: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class LazyPPRResult:
    """Result from Lazy PPR computation."""

    entity_scores: dict[int, float]  # entity_id → score
    passage_scores: dict[int, float]  # passage_id → score
    iterations: int  # Number of iterations until convergence
    converged: bool  # Whether PPR converged before max_iter
    timing_details: Optional[PPRTimingDetails] = None  # Detailed timing breakdown


class LazyBipartitePPR:
    """Lazy Personalized PageRank on bipartite graph.

    Runs PPR without loading the full graph matrix.
    Uses database queries to fetch neighbors on each iteration.

    Convention: r = (1-d)·v + d·Ã·r
    - d = damping factor (default 0.85)
    - d가 클수록 이웃으로 더 많이 전파 (85% to neighbors, 15% reset)
    - d가 작을수록 seed(활성화된 entity)에 집중

    The graph structure is:
        Entity ─contain─ Passage ─adjacency─ Passage
               ←───────→        ←──────────→
    """

    def __init__(
        self,
        db: "ReverieDB",
        damping: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-6,
        min_score: float = 1e-4,
        use_server_side: bool = False,
    ):
        """Initialize LazyBipartitePPR.

        Args:
            db: ReverieDB database interface.
            damping: Damping factor d. Higher values spread more to neighbors.
                     Default 0.85 (standard PageRank value).
            max_iter: Maximum PPR iterations.
            tol: Convergence tolerance.
            min_score: Minimum score to keep a node active. Nodes below this
                       threshold are pruned to prevent explosion.
            use_server_side: If True, use PostgreSQL server-side PPR function
                            for better performance. Requires run_lazy_ppr function
                            to be installed in the database.
        """
        self._db = db
        self._damping = damping
        self._max_iter = max_iter
        self._tol = tol
        self._min_score = min_score
        self._use_server_side = use_server_side

    def run(
        self,
        entity_seeds: dict[int, float],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Run Lazy Personalized PageRank.

        Args:
            entity_seeds: Entity activation weights from seed selection.
                Dict of {entity_id: similarity_score}.
            top_k: Number of top passages to return.

        Returns:
            List of (passage_id, score) tuples, sorted by score descending.
        """
        result = self.run_full(entity_seeds)

        # Sort passages by score and return top-k
        sorted_passages = sorted(
            result.passage_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_passages[:top_k]

    def run_full(
        self,
        entity_seeds: dict[int, float],
    ) -> LazyPPRResult:
        """Run Lazy PPR and return full result including metadata.

        Args:
            entity_seeds: Entity activation weights from seed selection.
                Dict of {entity_id: similarity_score}.

        Returns:
            LazyPPRResult with entity_scores, passage_scores, iterations, converged.
        """
        # Use server-side SQL function if enabled
        if self._use_server_side:
            return self.run_full_sql(entity_seeds)

        total_start = time.time()
        total_db_time = 0.0
        total_prop_time = 0.0
        per_iter_timings: list[tuple[float, float]] = []

        if not entity_seeds:
            return LazyPPRResult(
                entity_scores={},
                passage_scores={},
                iterations=0,
                converged=True,
            )

        # Normalize seeds to sum to 1
        seed_sum = sum(entity_seeds.values())
        if seed_sum == 0:
            return LazyPPRResult(
                entity_scores={},
                passage_scores={},
                iterations=0,
                converged=True,
            )

        normalized_seeds = {
            eid: score / seed_sum for eid, score in entity_seeds.items()
        }

        # Initialize scores: only seed entities have initial activation
        # Use (type, id) tuples internally to distinguish entity vs passage
        scores: dict[tuple[Literal["e", "p"], int], float] = {
            ("e", eid): activation for eid, activation in normalized_seeds.items()
        }

        d = self._damping
        converged = False
        iteration = 0

        for iteration in range(self._max_iter):
            # Collect active nodes (only those above min_score threshold)
            active_entities = [
                id for (t, id), score in scores.items()
                if t == "e" and score >= self._min_score
            ]
            active_passages = [
                id for (t, id), score in scores.items()
                if t == "p" and score >= self._min_score
            ]

            logger.debug(
                "PPR iter %d: %d entities, %d passages active (min_score=%.1e, total_nodes=%d)",
                iteration, len(active_entities), len(active_passages),
                self._min_score, len(scores)
            )

            # Fetch all neighbors in one query
            db_start = time.time()
            edges = self._db.get_ppr_neighbors(
                active_entities, active_passages
            )
            iter_db_time = time.time() - db_start
            total_db_time += iter_db_time

            logger.debug(
                "PPR iter %d: fetched %d edges in %.1fms",
                iteration, len(edges), iter_db_time * 1000
            )

            # Compute propagation
            prop_start = time.time()
            new_scores: dict[tuple[Literal["e", "p"], int], float] = defaultdict(float)

            for edge in edges:
                if edge.out_degree == 0:
                    # Skip nodes with zero out_degree - likely data integrity issue
                    logger.warning(
                        "Zero out_degree for edge %s (type=%s, src=%d, dst=%d). "
                        "This may indicate stale out_degree values.",
                        edge.edge_type, edge.edge_type, edge.source_id, edge.target_id
                    )
                    continue

                weight = 1.0 / edge.out_degree

                if edge.edge_type == "e2p":
                    # Entity → Passage
                    src_score = scores.get(("e", edge.source_id), 0.0)
                    new_scores[("p", edge.target_id)] += src_score * weight

                elif edge.edge_type == "p2e":
                    # Passage → Entity
                    src_score = scores.get(("p", edge.source_id), 0.0)
                    new_scores[("e", edge.target_id)] += src_score * weight

                elif edge.edge_type in ("p2p_a", "p2p_b"):
                    # Passage → Passage (both directions handled by separate edge types)
                    src_score = scores.get(("p", edge.source_id), 0.0)
                    new_scores[("p", edge.target_id)] += src_score * weight

            # Apply teleport (reset to seed)
            final_scores: dict[tuple[Literal["e", "p"], int], float] = {}
            all_nodes = set(scores.keys()) | set(new_scores.keys())

            for node in all_nodes:
                propagated = new_scores.get(node, 0.0)
                # Seed value: only entities that were seeds get teleport
                if node[0] == "e" and node[1] in normalized_seeds:
                    seed_val = normalized_seeds[node[1]]
                else:
                    seed_val = 0.0

                final_scores[node] = (1 - d) * seed_val + d * propagated
            iter_prop_time = time.time() - prop_start
            total_prop_time += iter_prop_time

            # Record per-iteration timing
            per_iter_timings.append((
                round(iter_db_time * 1000, 1),
                round(iter_prop_time * 1000, 1),
            ))

            logger.debug(
                "PPR iter %d: db=%.1fms, prop=%.1fms, new_nodes=%d",
                iteration, iter_db_time * 1000, iter_prop_time * 1000, len(new_scores)
            )

            # Check convergence
            if self._check_convergence(scores, final_scores):
                converged = True
                scores = final_scores
                break

            scores = final_scores

        # Convert internal representation to separate dicts
        entity_scores = {id: score for (t, id), score in scores.items() if t == "e"}
        passage_scores = {id: score for (t, id), score in scores.items() if t == "p"}

        total_elapsed = (time.time() - total_start) * 1000
        actual_iterations = iteration + 1

        timing_details = PPRTimingDetails(
            total_ms=round(total_elapsed, 1),
            iterations=actual_iterations,
            db_query_ms=round(total_db_time * 1000, 1),
            propagation_ms=round(total_prop_time * 1000, 1),
            per_iter_avg_ms=round(total_elapsed / actual_iterations, 1) if actual_iterations > 0 else 0.0,
            per_iter_timings=per_iter_timings,
        )

        logger.info(
            "PPR complete: %d iters, %s, total=%.1fms (db=%.1fms, prop=%.1fms), "
            "entities=%d, passages=%d",
            actual_iterations,
            "converged" if converged else "max_iter",
            total_elapsed,
            total_db_time * 1000,
            total_prop_time * 1000,
            len(entity_scores),
            len(passage_scores),
        )

        return LazyPPRResult(
            entity_scores=entity_scores,
            passage_scores=passage_scores,
            iterations=actual_iterations,
            converged=converged,
            timing_details=timing_details,
        )

    def run_full_sql(
        self,
        entity_seeds: dict[int, float],
    ) -> LazyPPRResult:
        """Run PPR using server-side PostgreSQL function.

        This eliminates network round-trips by running all PPR iterations
        inside the database. Requires the run_lazy_ppr function to be
        installed (see migrations/018_run_lazy_ppr_function.sql).

        Args:
            entity_seeds: Entity activation weights from seed selection.
                Dict of {entity_id: similarity_score}.

        Returns:
            LazyPPRResult with entity_scores, passage_scores, iterations, converged.
        """
        total_start = time.time()

        if not entity_seeds:
            return LazyPPRResult(
                entity_scores={},
                passage_scores={},
                iterations=0,
                converged=True,
            )

        # Normalize seeds to sum to 1
        seed_sum = sum(entity_seeds.values())
        if seed_sum == 0:
            return LazyPPRResult(
                entity_scores={},
                passage_scores={},
                iterations=0,
                converged=True,
            )

        seed_ids = list(entity_seeds.keys())
        seed_scores = [v / seed_sum for v in entity_seeds.values()]

        # Execute server-side PPR function via ReverieDB
        results = self._db.run_ppr_server_side(
            seed_ids=seed_ids,
            seed_scores=seed_scores,
            damping=self._damping,
            max_iter=self._max_iter,
            min_score=self._min_score,
            tol=self._tol,
        )

        # Parse results into entity and passage scores
        entity_scores: dict[int, float] = {}
        passage_scores: dict[int, float] = {}

        for node_type, node_id, node_score in results:
            if node_type == 'e':
                entity_scores[node_id] = node_score
            elif node_type == 'p':
                passage_scores[node_id] = node_score

        total_elapsed = (time.time() - total_start) * 1000

        timing_details = PPRTimingDetails(
            total_ms=round(total_elapsed, 1),
            iterations=self._max_iter,  # Server-side runs max_iter
            db_query_ms=round(total_elapsed, 1),  # All time is DB time
            propagation_ms=0.0,  # No Python propagation
            per_iter_avg_ms=round(total_elapsed / self._max_iter, 1) if self._max_iter > 0 else 0.0,
            per_iter_timings=[],  # Not available for server-side
        )

        logger.info(
            "PPR (server-side) complete: max_iter=%d, total=%.1fms, "
            "entities=%d, passages=%d",
            self._max_iter,
            total_elapsed,
            len(entity_scores),
            len(passage_scores),
        )

        return LazyPPRResult(
            entity_scores=entity_scores,
            passage_scores=passage_scores,
            iterations=self._max_iter,
            converged=True,  # Server-side always runs to max_iter
            timing_details=timing_details,
        )

    def run_with_graph(
        self,
        entity_seeds: dict[int, float],
        top_k_passages: int = 10,
        score_threshold: float = 0.0001,
    ) -> dict:
        """Run PPR and return full traversal graph for visualization.

        Args:
            entity_seeds: Entity activation weights from seed selection.
            top_k_passages: Number of top passages to return as final results.
            score_threshold: Minimum PPR score to include a node in the graph.

        Returns:
            Dict with:
            - passages: List of (passage_id, score) for top-k results
            - graph_nodes: List of {id, type, score} for all traversed nodes
            - graph_edges: List of {source, target, type} for edges between nodes
            - iterations: Number of iterations until convergence
            - converged: Whether PPR converged
        """
        result = self.run_full(entity_seeds)

        # Sort and get top-k passages
        sorted_passages = sorted(
            result.passage_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        passages = sorted_passages[:top_k_passages]

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

        # Build graph edges between included nodes
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

    def _check_convergence(
        self,
        old_scores: dict[tuple[Literal["e", "p"], int], float],
        new_scores: dict[tuple[Literal["e", "p"], int], float],
    ) -> bool:
        """Check if PPR has converged.

        Uses relative tolerance comparison similar to np.allclose.

        Args:
            old_scores: Previous iteration scores.
            new_scores: Current iteration scores.

        Returns:
            True if converged, False otherwise.
        """
        all_keys = set(old_scores.keys()) | set(new_scores.keys())

        if not all_keys:
            return True

        for key in all_keys:
            old_val = old_scores.get(key, 0.0)
            new_val = new_scores.get(key, 0.0)

            # Relative tolerance check: |new - old| <= tol * max(|old|, |new|)
            diff = abs(new_val - old_val)
            scale = max(abs(old_val), abs(new_val))

            if scale > 0 and diff > self._tol * scale:
                return False
            elif scale == 0 and diff > self._tol:
                return False

        return True
