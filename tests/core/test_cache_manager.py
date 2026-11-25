"""Tests for GraphCacheManager with TTL-based caching."""

import time
from unittest.mock import MagicMock, patch

import pytest

from reverie.core.cache_manager import (
    GraphCacheManager,
    clear_all_caches,
    get_graph_cache,
    invalidate_graph_cache,
)


@pytest.fixture
def cache():
    """Create a fresh GraphCacheManager for each test."""
    # Clear global instances
    clear_all_caches()
    return GraphCacheManager(profile_id="test", ttl_seconds=1)


@pytest.fixture
def mock_builder():
    """Mock BipartiteGraphBuilder for tests."""
    with patch("reverie.core.cache_manager.get_connection") as mock_conn, \
         patch("reverie.core.cache_manager.BipartiteGraphBuilder") as mock_builder:

        # Configure mock
        mock_graph = MagicMock()
        mock_builder_instance = MagicMock()
        mock_builder_instance.build.return_value = mock_graph
        mock_builder_instance._entity_list = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]
        mock_builder_instance._passage_list = [
            MagicMock(embedding=[0.3] * 1536),
        ]
        mock_builder.from_connection.return_value = mock_builder_instance

        yield mock_builder_instance, mock_graph


class TestGraphCacheManager:
    """Tests for GraphCacheManager functionality."""

    def test_initialization(self, cache):
        """Test cache manager initialization."""
        assert cache.profile_id == "test"
        assert cache.ttl == 1
        assert cache.is_cached is False
        assert cache.cache_age is None

    def test_get_graph_loads_on_first_access(self, cache, mock_builder):
        """Test that first access loads the graph."""
        mock_builder_instance, mock_graph = mock_builder

        graph = cache.get_graph()

        assert graph is mock_graph
        assert cache.is_cached is True
        assert cache.cache_age is not None
        assert cache.stats["misses"] == 1

    def test_get_graph_uses_cache(self, cache, mock_builder):
        """Test that subsequent accesses use cached graph."""
        mock_builder_instance, mock_graph = mock_builder

        # First access
        graph1 = cache.get_graph()
        # Second access
        graph2 = cache.get_graph()

        assert graph1 is graph2
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1
        # Builder should only be called once
        assert mock_builder_instance.build.call_count == 1

    def test_cache_expires_after_ttl(self, cache, mock_builder):
        """Test that cache expires after TTL."""
        mock_builder_instance, mock_graph = mock_builder

        # First access
        cache.get_graph()
        assert cache.stats["misses"] == 1

        # Wait for TTL to expire
        time.sleep(1.1)

        # Second access should trigger refresh
        cache.get_graph()
        assert cache.stats["misses"] == 2
        assert mock_builder_instance.build.call_count == 2

    def test_invalidate_clears_cache(self, cache, mock_builder):
        """Test that invalidate clears the cache."""
        mock_builder_instance, mock_graph = mock_builder

        # Load cache
        cache.get_graph()
        assert cache.is_cached is True

        # Invalidate
        cache.invalidate()

        assert cache.is_cached is False
        assert cache.cache_age is None
        assert cache.stats["invalidations"] == 1

    def test_refresh_forces_rebuild(self, cache, mock_builder):
        """Test that refresh forces cache rebuild."""
        mock_builder_instance, mock_graph = mock_builder

        # First access
        cache.get_graph()
        assert mock_builder_instance.build.call_count == 1

        # Force refresh (before TTL expires)
        cache.refresh()

        assert mock_builder_instance.build.call_count == 2
        assert cache.stats["refreshes"] == 2

    def test_stats_tracking(self, cache, mock_builder):
        """Test that statistics are tracked correctly."""
        mock_builder_instance, _ = mock_builder

        cache.get_graph()  # Miss
        cache.get_graph()  # Hit
        cache.get_graph()  # Hit
        cache.invalidate()
        cache.get_graph()  # Miss

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["invalidations"] == 1
        assert stats["is_cached"] is True


class TestEmbeddingCaching:
    """Tests for embedding matrix caching."""

    def test_entity_embeddings_cached(self, cache, mock_builder):
        """Test that entity embeddings are cached."""
        mock_builder_instance, _ = mock_builder

        emb1 = cache.get_entity_embeddings()
        emb2 = cache.get_entity_embeddings()

        # Should return same array (cached)
        assert emb1 is emb2
        assert emb1.shape == (2, 1536)

    def test_passage_embeddings_cached(self, cache, mock_builder):
        """Test that passage embeddings are cached."""
        mock_builder_instance, _ = mock_builder

        emb1 = cache.get_passage_embeddings()
        emb2 = cache.get_passage_embeddings()

        # Should return same array (cached)
        assert emb1 is emb2
        assert emb1.shape == (1, 1536)

    def test_embeddings_cleared_on_invalidate(self, cache, mock_builder):
        """Test that embedding caches are cleared on invalidate."""
        mock_builder_instance, _ = mock_builder

        cache.get_entity_embeddings()
        cache.invalidate()

        # Should rebuild
        cache.get_entity_embeddings()

        # Graph should be built twice
        assert mock_builder_instance.build.call_count == 2


class TestGlobalCacheFactory:
    """Tests for global cache factory functions."""

    def test_get_graph_cache_singleton(self):
        """Test that get_graph_cache returns singleton per profile."""
        clear_all_caches()

        cache1 = get_graph_cache("profile1")
        cache2 = get_graph_cache("profile1")
        cache3 = get_graph_cache("profile2")

        assert cache1 is cache2
        assert cache1 is not cache3
        assert cache1.profile_id == "profile1"
        assert cache3.profile_id == "profile2"

    def test_invalidate_graph_cache(self, mock_builder):
        """Test global cache invalidation."""
        clear_all_caches()

        cache = get_graph_cache("test_inv")
        cache.get_graph()  # Load cache
        assert cache.is_cached is True

        invalidate_graph_cache("test_inv")

        assert cache.is_cached is False

    def test_clear_all_caches(self, mock_builder):
        """Test clearing all cache instances."""
        clear_all_caches()

        cache1 = get_graph_cache("profile1")
        cache2 = get_graph_cache("profile2")
        cache1.get_graph()
        cache2.get_graph()

        clear_all_caches()

        # New instances should be created
        new_cache1 = get_graph_cache("profile1")
        assert new_cache1 is not cache1
        assert new_cache1.is_cached is False


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_access(self, cache, mock_builder):
        """Test that concurrent access is handled safely."""
        import threading

        mock_builder_instance, _ = mock_builder
        results = []
        errors = []

        def access_cache():
            try:
                graph = cache.get_graph()
                results.append(graph is not None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_cache) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(results)

    def test_concurrent_invalidation(self, cache, mock_builder):
        """Test that concurrent invalidation is safe."""
        import threading

        mock_builder_instance, _ = mock_builder
        errors = []

        def access_and_invalidate():
            try:
                for _ in range(5):
                    cache.get_graph()
                    cache.invalidate()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_and_invalidate) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
