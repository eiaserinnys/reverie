"""Tests for API response cache."""

import json
import tempfile
from pathlib import Path

import pytest

from reverie.cache.api_cache import ApiCache


class TestApiCache:
    """Test ApiCache file-based caching."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Provide a temporary cache directory."""
        return tmp_path / "api_cache"

    @pytest.fixture
    def cache(self, cache_dir):
        """Provide an ApiCache instance."""
        return ApiCache(cache_dir=str(cache_dir))

    def test_cache_dir_created(self, cache):
        """Cache directory is created on init."""
        assert Path(cache.cache_dir).exists()

    def test_miss_returns_none(self, cache):
        """Cache miss returns None."""
        result = cache.get("embedding", {"input": "hello"})
        assert result is None

    def test_put_and_get(self, cache):
        """Put a value and retrieve it."""
        request = {"model": "text-embedding-3-small", "input": "hello"}
        response = {"data": [{"embedding": [0.1, 0.2], "index": 0}]}

        cache.put("embedding", request, response)
        result = cache.get("embedding", request)

        assert result == response

    def test_same_request_different_namespace(self, cache):
        """Different namespaces don't collide."""
        request = {"input": "hello"}
        resp_a = {"result": "a"}
        resp_b = {"result": "b"}

        cache.put("embedding", request, resp_a)
        cache.put("chat", request, resp_b)

        assert cache.get("embedding", request) == resp_a
        assert cache.get("chat", request) == resp_b

    def test_deterministic_hash(self, cache):
        """Same request always produces the same hash."""
        request = {"model": "m", "input": ["a", "b"]}
        response = {"ok": True}

        cache.put("ns", request, response)

        # Create a new cache with the same directory
        cache2 = ApiCache(cache_dir=cache.cache_dir)
        assert cache2.get("ns", request) == response

    def test_key_order_invariant(self, cache):
        """Dict key order doesn't affect the hash."""
        req1 = {"model": "m", "input": "x"}
        req2 = {"input": "x", "model": "m"}
        response = {"ok": True}

        cache.put("ns", req1, response)
        assert cache.get("ns", req2) == response

    def test_disabled_cache(self, tmp_path):
        """Cache with enabled=False always returns None."""
        cache = ApiCache(cache_dir=str(tmp_path / "disabled"), enabled=False)
        request = {"input": "hello"}
        response = {"ok": True}

        cache.put("ns", request, response)
        assert cache.get("ns", request) is None

    def test_stats_tracking(self, cache):
        """Cache tracks hit/miss stats."""
        request = {"input": "hello"}
        response = {"ok": True}

        cache.get("ns", request)  # miss
        cache.put("ns", request, response)
        cache.get("ns", request)  # hit
        cache.get("ns", request)  # hit
        cache.get("ns", {"input": "world"})  # miss

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 2

    def test_corrupted_cache_file(self, cache):
        """Corrupted cache file is handled gracefully."""
        request = {"input": "hello"}
        response = {"ok": True}

        cache.put("ns", request, response)

        # Corrupt the file
        cache_file = cache._get_cache_path("ns", request)
        cache_file.write_text("not json{{{")

        # Should return None, not crash
        assert cache.get("ns", request) is None

    def test_clear(self, cache):
        """Clear removes all cached entries."""
        cache.put("ns", {"a": 1}, {"r": 1})
        cache.put("ns", {"a": 2}, {"r": 2})

        cache.clear()

        assert cache.get("ns", {"a": 1}) is None
        assert cache.get("ns", {"a": 2}) is None
