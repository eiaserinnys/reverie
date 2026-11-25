"""Timing utilities for profiling retrieval pipeline stages.

This module provides StageTimer for measuring execution time of
individual pipeline stages, and StageTiming for storing results.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageTiming:
    """Timing data for pipeline stages.

    Attributes:
        total_ms: Total time across all measured stages.
        stages: Dict mapping stage name to duration in ms.
        graph_build_ms: Time spent building graph (None if cache hit).
        cache_hit: Whether graph cache was hit.
        seed_selection_details: Detailed timing for seed selection (optional).
        ppr_details: Detailed timing for PPR execution (optional).
    """

    total_ms: float
    stages: dict[str, float]
    graph_build_ms: Optional[float] = None
    cache_hit: Optional[bool] = None
    seed_selection_details: Optional[dict[str, float]] = None
    ppr_details: Optional[dict[str, float]] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        result = {
            "total_ms": self.total_ms,
            "stages": self.stages,
            "graph_build_ms": self.graph_build_ms,
            "cache_hit": self.cache_hit,
        }
        if self.seed_selection_details:
            result["seed_selection_details"] = self.seed_selection_details
        if self.ppr_details:
            result["ppr_details"] = self.ppr_details
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "StageTiming":
        """Create from dict."""
        return cls(
            total_ms=data["total_ms"],
            stages=data["stages"],
            graph_build_ms=data.get("graph_build_ms"),
            cache_hit=data.get("cache_hit"),
            seed_selection_details=data.get("seed_selection_details"),
            ppr_details=data.get("ppr_details"),
        )


class StageTimer:
    """Timer for measuring pipeline stage durations.

    Usage:
        timer = StageTimer()

        with timer.stage("embed_query"):
            query_emb = embed(query)

        with timer.stage("knn_search"):
            results = knn(query_emb)

        timing = timer.get_timing()
        print(timing.stages)  # {"embed_query": 12.3, "knn_search": 8.5}
    """

    def __init__(self):
        self._stages: dict[str, float] = {}
        self._graph_build_ms: Optional[float] = None
        self._cache_hit: Optional[bool] = None
        self._current_stage: Optional[str] = None
        self._seed_selection_details: Optional[dict[str, float]] = None
        self._ppr_details: Optional[dict[str, float]] = None

    @contextmanager
    def stage(self, name: str):
        """Context manager for timing a stage.

        Args:
            name: Stage name (e.g., "embed_query", "knn_search").

        Raises:
            RuntimeError: If nested stages are attempted.
        """
        if self._current_stage is not None:
            raise RuntimeError(
                f"Cannot start stage '{name}' while '{self._current_stage}' is running. "
                "Nested stages are not allowed."
            )

        self._current_stage = name
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._stages[name] = elapsed_ms
            self._current_stage = None

    @contextmanager
    def graph_build(self):
        """Context manager for timing graph build (separate from stages).

        Use this for the graph build operation, which is tracked
        separately to distinguish cache hits from misses.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self._graph_build_ms = (time.perf_counter() - start) * 1000

    def set_cache_hit(self, hit: bool) -> None:
        """Set whether graph cache was hit.

        Args:
            hit: True if cache hit, False if cache miss.
        """
        self._cache_hit = hit

    def set_seed_selection_details(self, details: dict[str, float]) -> None:
        """Set detailed timing for seed selection.

        Args:
            details: Dict with per-entity timing breakdown.
        """
        self._seed_selection_details = details

    def set_ppr_details(self, details: dict[str, float]) -> None:
        """Set detailed timing for PPR execution.

        Args:
            details: Dict with iterations, db_query_ms, propagation_ms, per_iter_avg_ms.
        """
        self._ppr_details = details

    def get_timing(self) -> StageTiming:
        """Get timing results.

        Returns:
            StageTiming with all measured durations.
        """
        stages_ms = sum(self._stages.values())
        # Include graph_build_ms in total if present
        total_ms = stages_ms + (self._graph_build_ms or 0)

        return StageTiming(
            total_ms=total_ms,
            stages=dict(self._stages),
            graph_build_ms=self._graph_build_ms,
            cache_hit=self._cache_hit,
            seed_selection_details=self._seed_selection_details,
            ppr_details=self._ppr_details,
        )
