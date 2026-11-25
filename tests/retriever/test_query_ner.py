"""Tests for QueryNERClient.

Tests for query-time entity extraction using NER.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestQueryNERClient:
    """Test QueryNERClient."""

    def test_extract_entities_basic(self):
        """Should extract entities from query using OpenIEClient.extract_query_entities."""
        from reverie.retriever.query_ner import QueryNERClient

        # Mock OpenIEClient
        mock_openie = MagicMock()
        mock_openie.extract_query_entities.return_value = ["agent-soul", "memory"]

        client = QueryNERClient(openie_client=mock_openie)
        entities = client.extract_entities("에이전트 봇의 메모리 시스템은 어떻게 작동해?")

        assert entities == ["agent-soul", "memory"]
        mock_openie.extract_query_entities.assert_called_once()

    def test_extract_entities_empty_query(self):
        """Should return empty list for empty query."""
        from reverie.retriever.query_ner import QueryNERClient

        mock_openie = MagicMock()
        client = QueryNERClient(openie_client=mock_openie)

        assert client.extract_entities("") == []
        assert client.extract_entities("   ") == []
        mock_openie.extract_entities.assert_not_called()

    def test_extract_entities_no_entities(self):
        """Should return empty list when no entities found."""
        from reverie.retriever.query_ner import QueryNERClient

        mock_openie = MagicMock()
        mock_openie.extract_query_entities.return_value = []

        client = QueryNERClient(openie_client=mock_openie)
        entities = client.extract_entities("안녕하세요")

        assert entities == []

    def test_default_openie_client(self):
        """Should create default OpenIEClient if not provided."""
        from reverie.retriever.query_ner import QueryNERClient

        with patch('reverie.retriever.query_ner.OpenIEClient') as MockOpenIE:
            mock_instance = MagicMock()
            mock_instance.extract_query_entities.return_value = ["test"]
            MockOpenIE.return_value = mock_instance

            client = QueryNERClient()
            entities = client.extract_entities("test query")

            MockOpenIE.assert_called_once()
            assert entities == ["test"]
