"""Tests for MCP server tools (semantic_search, memory_status)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reverie.mcp.server import create_server, _get_pipeline, _get_stats
from reverie.retriever.pipeline_v2 import RetrievalResult


class TestSemanticSearch:
    """Test semantic_search tool."""

    @pytest.fixture
    def mock_pipeline(self):
        """Mock MemoryLinkV2Pipeline."""
        mock = MagicMock()
        mock.retrieve.return_value = [
            RetrievalResult(
                passage_id=1,
                text="systemctl restarted agent-bot.service",
                score=0.85,
                source="memorylink-v2",
            ),
            RetrievalResult(
                passage_id=2,
                text="pgvector extension installed successfully",
                score=0.62,
                source="memorylink-v2",
            ),
        ]
        return mock

    @pytest.mark.asyncio
    async def test_semantic_search_returns_results(self, mock_pipeline):
        """semantic_search returns formatted results."""
        with patch("reverie.mcp.server._get_pipeline", return_value=mock_pipeline):
            server = create_server()
            # Get the tool function directly
            tools = server._tool_manager._tools
            tool_fn = tools["semantic_search"].fn

            result = await tool_fn(query="agent restart", top_k=5)

            assert "systemctl restarted agent-bot.service" in result
            assert "0.85" in result

    @pytest.mark.asyncio
    async def test_semantic_search_empty_results(self, mock_pipeline):
        """semantic_search handles empty results."""
        mock_pipeline.retrieve.return_value = []

        with patch("reverie.mcp.server._get_pipeline", return_value=mock_pipeline):
            server = create_server()
            tools = server._tool_manager._tools
            tool_fn = tools["semantic_search"].fn

            result = await tool_fn(query="nonexistent topic", top_k=5)

            assert "No results" in result or "0 results" in result

    @pytest.mark.asyncio
    async def test_semantic_search_pipeline_error(self):
        """semantic_search handles pipeline errors gracefully."""
        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.side_effect = Exception("DB connection failed")

        with patch("reverie.mcp.server._get_pipeline", return_value=mock_pipeline):
            server = create_server()
            tools = server._tool_manager._tools
            tool_fn = tools["semantic_search"].fn

            result = await tool_fn(query="test", top_k=5)

            assert "error" in result.lower() or "Error" in result


class TestMemoryStatus:
    """Test memory_status tool."""

    @pytest.mark.asyncio
    async def test_memory_status_returns_stats(self):
        """memory_status returns database statistics."""
        mock_stats = {
            "entities": 150,
            "passages": 50,
            "entity_passages": 300,
            "passage_adjacency": 45,
            "sessions": 4,
            "last_indexed": "2026-02-05T03:00:00",
            "process_memory_mb": 128.5,
        }

        with patch("reverie.mcp.server._get_stats", return_value=mock_stats):
            server = create_server()
            tools = server._tool_manager._tools
            tool_fn = tools["memory_status"].fn

            result = await tool_fn()

            assert "150" in result  # entities
            assert "50" in result   # passages
            assert "128.5" in result  # memory usage

    @pytest.mark.asyncio
    async def test_memory_status_db_error(self):
        """memory_status handles DB errors."""
        with patch("reverie.mcp.server._get_stats", side_effect=Exception("DB error")):
            server = create_server()
            tools = server._tool_manager._tools
            tool_fn = tools["memory_status"].fn

            result = await tool_fn()

            assert "error" in result.lower() or "Error" in result


class TestRecallEnrich:
    """Test recall_enrich tool."""

    @pytest.fixture
    def mock_pipeline(self):
        """Mock MemoryLinkV2Pipeline."""
        mock = MagicMock()
        mock.retrieve.return_value = [
            RetrievalResult(
                passage_id=1,
                text="Agent service was restarted.",
                score=0.90,
                source="memorylink-v2",
            ),
        ]
        return mock

    @pytest.mark.asyncio
    async def test_recall_enrich_returns_enriched_prompt(self, mock_pipeline):
        """recall_enrich enriches the prompt with memory context."""
        with patch("reverie.mcp.server._get_pipeline", return_value=mock_pipeline):
            server = create_server()
            tools = server._tool_manager._tools
            tool_fn = tools["recall_enrich"].fn

            result = await tool_fn(prompt="What happened?")

            assert "<long-term-memory>" in result
            assert "Agent service was restarted." in result
            assert "What happened?" in result

    @pytest.mark.asyncio
    async def test_recall_enrich_returns_original_on_error(self):
        """recall_enrich returns original prompt when pipeline fails."""
        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.side_effect = Exception("DB error")

        with patch("reverie.mcp.server._get_pipeline", return_value=mock_pipeline):
            server = create_server()
            tools = server._tool_manager._tools
            tool_fn = tools["recall_enrich"].fn

            result = await tool_fn(prompt="My prompt")

            # Should return original prompt on error
            assert result == "My prompt"

    @pytest.mark.asyncio
    async def test_recall_enrich_empty_results(self):
        """recall_enrich returns original prompt when no results found."""
        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.return_value = []

        with patch("reverie.mcp.server._get_pipeline", return_value=mock_pipeline):
            server = create_server()
            tools = server._tool_manager._tools
            tool_fn = tools["recall_enrich"].fn

            result = await tool_fn(prompt="My prompt")

            assert result == "My prompt"


class TestInvalidateGraph:
    """Test invalidate_graph tool."""

    @pytest.mark.asyncio
    async def test_invalidate_graph_clears_cache(self):
        """invalidate_graph clears graph cache."""
        mock_pipeline = MagicMock()

        with patch("reverie.mcp.server._get_pipeline", return_value=mock_pipeline):
            server = create_server()
            tools = server._tool_manager._tools
            tool_fn = tools["invalidate_graph"].fn

            result = await tool_fn()

            mock_pipeline.invalidate_graph.assert_called_once()
            assert "invalidated" in result.lower()

    @pytest.mark.asyncio
    async def test_invalidate_graph_handles_error(self):
        """invalidate_graph handles errors gracefully."""
        mock_pipeline = MagicMock()
        mock_pipeline.invalidate_graph.side_effect = Exception("Test error")

        with patch("reverie.mcp.server._get_pipeline", return_value=mock_pipeline):
            server = create_server()
            tools = server._tool_manager._tools
            tool_fn = tools["invalidate_graph"].fn

            result = await tool_fn()

            assert "error" in result.lower()


class TestCreateServer:
    """Test server creation."""

    def test_server_has_tools(self):
        """Server registers all tools."""
        server = create_server()
        tools = server._tool_manager._tools
        assert "semantic_search" in tools
        assert "recall_enrich" in tools
        assert "memory_status" in tools
        assert "invalidate_graph" in tools
