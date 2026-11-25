"""OpenAI embedding client.

Generates embeddings using OpenAI text-embedding-3-small API.
Supports single and batch embedding with rate-limit retry.
"""

import logging
import time
from typing import Optional

from openai import OpenAI, RateLimitError

from reverie.cache.api_cache import ApiCache
from reverie.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """OpenAI embedding client with batch support.

    Args:
        api_key: OpenAI API key. Defaults to config value.
        model: Embedding model name. Defaults to config value.
        dimensions: Embedding dimensions. Defaults to config value.
        batch_size: Max texts per API call. Defaults to config value.
        max_retries: Max retries on rate limit. Defaults to 3.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_retries: int = 3,
        cache: Optional[ApiCache] = None,
    ):
        settings = get_settings()
        self._api_key = api_key or settings.openai_api_key
        self._model = model or settings.embedding_model
        self._dimensions = dimensions or settings.embedding_dimensions
        self._batch_size = batch_size or settings.embedding_batch_size
        self._max_retries = max_retries
        self._client = OpenAI(api_key=self._api_key)
        self._cache = cache or ApiCache(
            cache_dir=settings.cache_dir,
            enabled=settings.cache_enabled,
        )

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text. Empty/whitespace-only text is replaced
                  with a single space to avoid API errors.

        Returns:
            List of floats (embedding vector).
        """
        text = self._preprocess(text)
        response = self._call_api([text])
        return response[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Splits into chunks of batch_size and processes sequentially.
        Retries on rate limit with exponential backoff.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors, same order as input.
        """
        if not texts:
            return []

        processed = [self._preprocess(t) for t in texts]
        all_embeddings: list[list[float]] = []

        for i in range(0, len(processed), self._batch_size):
            batch = processed[i : i + self._batch_size]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _preprocess(self, text: str) -> str:
        """Preprocess text for embedding API.

        Replaces empty/whitespace-only strings with a single space.
        Strips leading/trailing whitespace and collapses newlines.
        """
        text = text.strip()
        if not text:
            return " "
        # Replace multiple newlines with single newline
        lines = [line for line in text.split("\n") if line.strip()]
        return "\n".join(lines) if lines else " "

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embedding API with cache and retry on rate limit.

        Args:
            texts: Preprocessed texts (max batch_size).

        Returns:
            List of embedding vectors.
        """
        cache_request = {
            "model": self._model,
            "input": texts,
            "dimensions": self._dimensions,
        }
        cached = self._cache.get("embedding", cache_request)
        if cached is not None:
            return cached

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                    dimensions=self._dimensions,
                )
                # Sort by index to preserve order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                result = [item.embedding for item in sorted_data]

                self._cache.put("embedding", cache_request, result)
                return result
            except RateLimitError as e:
                if attempt < self._max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Rate limited, retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, self._max_retries,
                    )
                    time.sleep(wait)
                else:
                    raise
