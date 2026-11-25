"""Graph cache manager with TTL-based invalidation.

Provides cached access to BipartiteGraph with automatic
refresh on TTL expiration or manual invalidation.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

from reverie.config import get_settings
from reverie.db.connection import get_connection
from reverie.core.bipartite import BipartiteGraph
from reverie.core.bipartite_builder import BipartiteGraphBuilder

logger = logging.getLogger(__name__)


class GraphCacheManager:
    """TTL-based cache manager for BipartiteGraph.

    Features:
    - Lazy loading on first access
    - TTL-based automatic refresh
    - Manual invalidation after indexing
    - Thread-safe access
    - Profile isolation

    Usage:
        # Create manager (usually singleton per profile)
        cache = GraphCacheManager(profile_id="my_profile", ttl_seconds=300)

        # Get graph (auto-loads if needed)
        graph = cache.get_graph()

        # Use graph
        scores = graph.compute_ppr(...)

        # Invalidate after indexing new data
        cache.invalidate()
    """

    def __init__(
        self,
        profile_id: str,
        ttl_seconds: int | None = None,
    ):
        """Initialize the cache manager.

        Args:
            profile_id: Profile to filter data by.
            ttl_seconds: Cache TTL in seconds (default from config).
        """
        settings = get_settings()
        self._profile_id = profile_id
        self._ttl = ttl_seconds or settings.graph_ttl_seconds
        self._lock = threading.RLock()

        # Cached data
        self._graph: Optional[BipartiteGraph] = None
        self._entity_embeddings: Optional[np.ndarray] = None
        self._passage_embeddings: Optional[np.ndarray] = None
        self._entity_list: list = []
        self._passage_list: list = []
        self._cache_time: Optional[float] = None

        # Stats
        self._stats = {
            "hits": 0,
            "misses": 0,
            "refreshes": 0,
            "invalidations": 0,
        }

    @property
    def profile_id(self) -> str:
        """Get profile ID."""
        return self._profile_id

    @property
    def ttl(self) -> int:
        """Get TTL in seconds."""
        return self._ttl

    @property
    def is_cached(self) -> bool:
        """Check if graph is currently cached."""
        return self._graph is not None

    @property
    def cache_age(self) -> float | None:
        """Get age of cached data in seconds, or None if not cached."""
        if self._cache_time is None:
            return None
        return time.time() - self._cache_time

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            **self._stats,
            "is_cached": self.is_cached,
            "cache_age_seconds": self.cache_age,
            "ttl_seconds": self._ttl,
            "profile_id": self._profile_id,
        }

    def get_graph(self) -> BipartiteGraph:
        """Get the cached graph, rebuilding if needed.

        Returns:
            BipartiteGraph instance.
        """
        with self._lock:
            if self._should_refresh():
                self._refresh()
                self._stats["misses"] += 1
            else:
                self._stats["hits"] += 1

            return self._graph

    def get_entity_embeddings(self, embedding_dim: int = 1536) -> np.ndarray:
        """Get entity embeddings matrix.

        Args:
            embedding_dim: Dimension of embeddings.

        Returns:
            numpy array of shape (n_entities, embedding_dim).
        """
        with self._lock:
            if self._should_refresh():
                self._refresh()

            if self._entity_embeddings is None:
                self._entity_embeddings = self._build_entity_embeddings(embedding_dim)

            return self._entity_embeddings

    def get_passage_embeddings(self, embedding_dim: int = 1536) -> np.ndarray:
        """Get passage embeddings matrix.

        Args:
            embedding_dim: Dimension of embeddings.

        Returns:
            numpy array of shape (n_passages, embedding_dim).
        """
        with self._lock:
            if self._should_refresh():
                self._refresh()

            if self._passage_embeddings is None:
                self._passage_embeddings = self._build_passage_embeddings(embedding_dim)

            return self._passage_embeddings

    @property
    def entity_list(self) -> list:
        """Get list of entities (matching graph indices)."""
        with self._lock:
            if self._should_refresh():
                self._refresh()
            return self._entity_list

    @property
    def passage_list(self) -> list:
        """Get list of passages (matching graph indices)."""
        with self._lock:
            if self._should_refresh():
                self._refresh()
            return self._passage_list

    def invalidate(self) -> None:
        """Invalidate the cache, forcing rebuild on next access.

        Call this after indexing new data.
        """
        with self._lock:
            self._graph = None
            self._entity_embeddings = None
            self._passage_embeddings = None
            self._entity_list = []
            self._passage_list = []
            self._cache_time = None
            self._stats["invalidations"] += 1
            logger.debug(f"[{self._profile_id}] Graph cache invalidated")

    def refresh(self) -> BipartiteGraph:
        """Force refresh the cache and return new graph.

        Returns:
            Freshly built BipartiteGraph.
        """
        with self._lock:
            self._refresh()
            return self._graph

    def _should_refresh(self) -> bool:
        """Check if cache should be refreshed."""
        if self._graph is None:
            return True
        if self._cache_time is None:
            return True
        return (time.time() - self._cache_time) >= self._ttl

    def _refresh(self) -> None:
        """Rebuild the graph from database.

        Must be called with lock held.
        """
        logger.debug(f"[{self._profile_id}] Refreshing graph cache")
        start_time = time.time()

        with get_connection() as conn:
            builder = BipartiteGraphBuilder.from_connection(conn)
            self._graph = builder.build()
            self._entity_list = builder._entity_list
            self._passage_list = builder._passage_list

        # Clear embedding caches (will be rebuilt on demand)
        self._entity_embeddings = None
        self._passage_embeddings = None

        self._cache_time = time.time()
        self._stats["refreshes"] += 1

        elapsed = (time.time() - start_time) * 1000
        logger.debug(
            f"[{self._profile_id}] Graph cache refreshed: "
            f"{len(self._entity_list)} entities, "
            f"{len(self._passage_list)} passages "
            f"({elapsed:.0f}ms)"
        )

    def _build_entity_embeddings(self, embedding_dim: int) -> np.ndarray:
        """Build entity embeddings matrix from entity list.

        Must be called with lock held and graph loaded.
        """
        n_entities = len(self._entity_list)
        embeddings = np.zeros((n_entities, embedding_dim), dtype=np.float64)

        for i, entity in enumerate(self._entity_list):
            if entity.embedding is not None:
                embeddings[i] = np.array(entity.embedding)

        return embeddings

    def _build_passage_embeddings(self, embedding_dim: int) -> np.ndarray:
        """Build passage embeddings matrix from passage list.

        Must be called with lock held and graph loaded.
        """
        n_passages = len(self._passage_list)
        embeddings = np.zeros((n_passages, embedding_dim), dtype=np.float64)

        for i, passage in enumerate(self._passage_list):
            if passage.embedding is not None:
                embeddings[i] = np.array(passage.embedding)

        return embeddings


# Global cache instances per profile
_cache_instances: dict[str, GraphCacheManager] = {}
_cache_lock = threading.Lock()


def get_graph_cache(profile_id: str) -> GraphCacheManager:
    """Get or create a GraphCacheManager for the given profile.

    This is a factory function that maintains singleton instances
    per profile.

    Args:
        profile_id: Profile identifier (required).

    Returns:
        GraphCacheManager instance.
    """
    with _cache_lock:
        if profile_id not in _cache_instances:
            _cache_instances[profile_id] = GraphCacheManager(profile_id)
        return _cache_instances[profile_id]


def invalidate_graph_cache(profile_id: str) -> None:
    """Invalidate the graph cache for a profile.

    Convenience function for use after indexing.

    Args:
        profile_id: Profile identifier (required).
    """
    with _cache_lock:
        if profile_id in _cache_instances:
            _cache_instances[profile_id].invalidate()


def clear_all_caches() -> None:
    """Clear all graph cache instances.

    Useful for testing or complete reset.
    """
    global _cache_instances
    with _cache_lock:
        for cache in _cache_instances.values():
            cache.invalidate()
        _cache_instances.clear()
