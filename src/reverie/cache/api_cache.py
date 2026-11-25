"""File-based API response cache.

Caches API responses (embeddings, chat completions) to avoid
redundant API calls during development and testing.

Cache key = sha256(namespace + canonical JSON of request).
Storage = one JSON file per entry in cache_dir/namespace/.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ApiCache:
    """File-based cache for API request/response pairs.

    Args:
        cache_dir: Directory for cache files. Created if not exists.
        enabled: If False, get() always returns None (bypass mode).
    """

    def __init__(self, cache_dir: str, enabled: bool = True):
        self._cache_dir = Path(cache_dir)
        self._enabled = enabled
        self._hits = 0
        self._misses = 0

        if self._enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> str:
        """Return cache directory path."""
        return str(self._cache_dir)

    @property
    def stats(self) -> dict[str, int]:
        """Return cache hit/miss statistics."""
        return {"hits": self._hits, "misses": self._misses}

    def get(self, namespace: str, request: dict) -> Any | None:
        """Retrieve cached response for a request.

        Args:
            namespace: Cache namespace (e.g. "embedding", "chat").
            request: Request parameters dict.

        Returns:
            Cached response dict, or None on miss/disabled/error.
        """
        if not self._enabled:
            return None

        cache_path = self._get_cache_path(namespace, request)
        if not cache_path.exists():
            self._misses += 1
            logger.debug("Cache MISS (%s): %s", namespace, cache_path.name)
            return None

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            self._hits += 1
            logger.debug("Cache HIT (%s): %s", namespace, cache_path.name)
            return data
        except (json.JSONDecodeError, OSError):
            self._misses += 1
            logger.debug("Cache MISS (read error) (%s): %s", namespace, cache_path.name)
            return None

    def put(self, namespace: str, request: dict, response: Any) -> None:
        """Store a response in the cache.

        Args:
            namespace: Cache namespace.
            request: Request parameters dict (used as cache key).
            response: Response data to cache (must be JSON-serializable).
        """
        if not self._enabled:
            return

        cache_path = self._get_cache_path(namespace, request)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cache_path.write_text(
                json.dumps(response, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Failed to write cache: %s", e)

    def clear(self) -> None:
        """Remove all cached entries."""
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, namespace: str, request: dict) -> Path:
        """Compute the file path for a cache entry."""
        key = self._compute_key(namespace, request)
        return self._cache_dir / namespace / f"{key}.json"

    @staticmethod
    def _compute_key(namespace: str, request: dict) -> str:
        """Compute a deterministic hash for a request.

        Uses sorted JSON serialization to ensure key-order invariance.
        """
        canonical = json.dumps(request, sort_keys=True, ensure_ascii=False)
        raw = f"{namespace}:{canonical}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
