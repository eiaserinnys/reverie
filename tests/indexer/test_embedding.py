"""Tests for OpenAI embedding client."""

from unittest.mock import MagicMock, patch

import pytest

from reverie.indexer.embedding import EmbeddingClient


@pytest.fixture
def mock_openai():
    """Create a mock OpenAI client."""
    with patch("reverie.indexer.embedding.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


def _make_embedding_response(embeddings: list[list[float]]):
    """Create a mock embedding response."""
    mock_response = MagicMock()
    mock_data = []
    for i, emb in enumerate(embeddings):
        item = MagicMock()
        item.index = i
        item.embedding = emb
        mock_data.append(item)
    mock_response.data = mock_data
    return mock_response


class TestEmbeddingClient:
    """Test EmbeddingClient."""

    def test_embed_text_single(self, mock_openai):
        """Embed a single text."""
        expected = [0.1, 0.2, 0.3]
        mock_openai.embeddings.create.return_value = _make_embedding_response(
            [expected]
        )

        client = EmbeddingClient(api_key="test-key")
        result = client.embed_text("Hello world")

        assert result == expected
        mock_openai.embeddings.create.assert_called_once()
        call_kwargs = mock_openai.embeddings.create.call_args[1]
        assert call_kwargs["input"] == ["Hello world"]
        assert call_kwargs["model"] == "text-embedding-3-small"
        assert call_kwargs["dimensions"] == 1536

    def test_embed_text_empty_string(self, mock_openai):
        """Empty string should be preprocessed to single space."""
        mock_openai.embeddings.create.return_value = _make_embedding_response(
            [[0.0, 0.0, 0.0]]
        )

        client = EmbeddingClient(api_key="test-key")
        client.embed_text("")

        call_kwargs = mock_openai.embeddings.create.call_args[1]
        assert call_kwargs["input"] == [" "]

    def test_embed_text_whitespace(self, mock_openai):
        """Whitespace-only string should be preprocessed."""
        mock_openai.embeddings.create.return_value = _make_embedding_response(
            [[0.0]]
        )

        client = EmbeddingClient(api_key="test-key")
        client.embed_text("   \n\n  ")

        call_kwargs = mock_openai.embeddings.create.call_args[1]
        assert call_kwargs["input"] == [" "]

    def test_embed_batch_simple(self, mock_openai):
        """Batch embed multiple texts."""
        expected = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_openai.embeddings.create.return_value = _make_embedding_response(
            expected
        )

        client = EmbeddingClient(api_key="test-key", batch_size=16)
        result = client.embed_batch(["a", "b", "c"])

        assert result == expected
        assert mock_openai.embeddings.create.call_count == 1

    def test_embed_batch_splits_by_batch_size(self, mock_openai):
        """Batch should be split when exceeding batch_size."""
        batch1 = [[0.1], [0.2]]
        batch2 = [[0.3]]

        mock_openai.embeddings.create.side_effect = [
            _make_embedding_response(batch1),
            _make_embedding_response(batch2),
        ]

        client = EmbeddingClient(api_key="test-key", batch_size=2)
        result = client.embed_batch(["a", "b", "c"])

        assert result == [[0.1], [0.2], [0.3]]
        assert mock_openai.embeddings.create.call_count == 2

    def test_embed_batch_empty(self, mock_openai):
        """Empty list should return empty list."""
        client = EmbeddingClient(api_key="test-key")
        result = client.embed_batch([])

        assert result == []
        mock_openai.embeddings.create.assert_not_called()

    def test_rate_limit_retry(self, mock_openai):
        """Should retry on rate limit error."""
        from openai import RateLimitError

        error_response = MagicMock()
        error_response.status_code = 429
        error_response.headers = {}

        mock_openai.embeddings.create.side_effect = [
            RateLimitError(
                message="Rate limit exceeded",
                response=error_response,
                body=None,
            ),
            _make_embedding_response([[0.1, 0.2]]),
        ]

        client = EmbeddingClient(api_key="test-key", max_retries=3)

        with patch("reverie.indexer.embedding.time.sleep"):
            result = client.embed_text("test")

        assert result == [0.1, 0.2]
        assert mock_openai.embeddings.create.call_count == 2

    def test_rate_limit_exhausted(self, mock_openai):
        """Should raise after max retries exhausted."""
        from openai import RateLimitError

        error_response = MagicMock()
        error_response.status_code = 429
        error_response.headers = {}

        mock_openai.embeddings.create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=error_response,
            body=None,
        )

        client = EmbeddingClient(api_key="test-key", max_retries=2)

        with patch("reverie.indexer.embedding.time.sleep"):
            with pytest.raises(RateLimitError):
                client.embed_text("test")

        # 1 initial + 2 retries = 3 calls
        assert mock_openai.embeddings.create.call_count == 3

    def test_preprocess_newlines(self, mock_openai):
        """Text with excessive newlines should be cleaned."""
        mock_openai.embeddings.create.return_value = _make_embedding_response(
            [[0.1]]
        )

        client = EmbeddingClient(api_key="test-key")
        client.embed_text("line1\n\n\n\nline2\n\n")

        call_kwargs = mock_openai.embeddings.create.call_args[1]
        assert call_kwargs["input"] == ["line1\nline2"]

    def test_preserves_response_order(self, mock_openai):
        """Embeddings should be returned in input order."""
        # Simulate out-of-order response
        mock_response = MagicMock()
        items = []
        for idx, emb in [(1, [0.2]), (0, [0.1]), (2, [0.3])]:
            item = MagicMock()
            item.index = idx
            item.embedding = emb
            items.append(item)
        mock_response.data = items
        mock_openai.embeddings.create.return_value = mock_response

        client = EmbeddingClient(api_key="test-key")
        result = client.embed_batch(["a", "b", "c"])

        assert result == [[0.1], [0.2], [0.3]]
