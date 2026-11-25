"""Integration tests for the full recall flow.

Tests the complete path: query → pipeline → formatter → RecallService → enriched prompt.
All external dependencies (DB, OpenAI) are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from reverie.recall.service import RecallService
from reverie.retriever.formatter import format_recall_context
from reverie.retriever.pipeline_v2 import RetrievalResult


def _make_result(passage_id, text, score=0.8, source="ppr"):
    return RetrievalResult(
        passage_id=passage_id,
        text=text,
        score=score,
        source=source,
    )


class TestFullRecallFlow:
    """End-to-end recall flow with mocked pipeline."""

    @pytest.fixture
    def pipeline_with_results(self):
        """Pipeline returning realistic results."""
        pipeline = MagicMock()
        pipeline.retrieve.return_value = [
            _make_result(
                1,
                "2026-02-03 세션에서 에이전트 서비스를 systemctl로 재시작했습니다. "
                "서비스 재시작 후 PM2 프로세스가 정상적으로 올라온 것을 확인했습니다.",
                0.92,
                "ppr",
            ),
            _make_result(
                2,
                "pgvector 확장이 PostgreSQL에 성공적으로 설치되었습니다. "
                "CREATE EXTENSION vector; 명령으로 활성화했습니다.",
                0.78,
                "ppr",
            ),
            _make_result(
                3,
                "reverie 프로젝트의 Phase 1에서 JSONL 파서를 구현했습니다. "
                "Claude Agent SDK의 세션 로그를 파싱합니다.",
                0.65,
                "knn",
            ),
        ]
        return pipeline

    def test_full_flow_enriches_prompt(self, pipeline_with_results):
        """Full recall flow produces enriched prompt with all passages."""
        service = RecallService(
            pipeline=pipeline_with_results,
            max_tokens=4000,
            max_passages=5,
        )

        prompt = "에이전트 서비스가 재시작된 적 있어?"
        result = service.enrich_prompt(prompt)

        # Memory context block
        assert "<long-term-memory>" in result
        assert "</long-term-memory>" in result

        # All three passages present
        assert "systemctl로 재시작했습니다" in result
        assert "pgvector" in result
        assert "JSONL 파서" in result

        # Original prompt preserved
        assert prompt in result

        # Context comes before prompt
        assert result.index("</long-term-memory>") < result.index(prompt)

    def test_flow_preserves_score_ordering(self, pipeline_with_results):
        """Passages appear in score order (highest first)."""
        service = RecallService(pipeline=pipeline_with_results)
        result = service.enrich_prompt("test query")

        pos_systemctl = result.index("systemctl")
        pos_pgvector = result.index("pgvector")
        pos_jsonl = result.index("JSONL")

        assert pos_systemctl < pos_pgvector < pos_jsonl

    def test_flow_includes_metadata(self, pipeline_with_results):
        """Scores and source labels are included in output."""
        service = RecallService(pipeline=pipeline_with_results)
        result = service.enrich_prompt("test")

        assert "0.92" in result
        assert "0.78" in result
        assert "0.65" in result
        assert "ppr" in result
        assert "knn" in result

    def test_flow_with_token_budget(self):
        """Token budget limits the number of passages included."""
        pipeline = MagicMock()
        pipeline.retrieve.return_value = [
            _make_result(i, f"UniquePassage{i} " * 50, 0.9 - i * 0.05)
            for i in range(10)
        ]

        service = RecallService(
            pipeline=pipeline,
            max_tokens=200,  # Tight budget
            max_passages=10,
        )
        result = service.enrich_prompt("query")

        # Should include at least the first passage (guaranteed)
        assert "UniquePassage0" in result
        # Count how many unique passage markers are present
        included = sum(1 for i in range(10) if f"UniquePassage{i}" in result)
        # Budget is tight, should not include all 10
        assert included < 10


class TestRecallServiceWithFormatter:
    """Test RecallService correctly uses format_recall_context."""

    def test_formatter_output_matches_service_output(self):
        """RecallService output matches direct formatter + concatenation."""
        results = [
            _make_result(1, "Test passage A.", 0.9, "ppr"),
            _make_result(2, "Test passage B.", 0.7, "knn"),
        ]

        pipeline = MagicMock()
        pipeline.retrieve.return_value = results

        service = RecallService(
            pipeline=pipeline,
            max_tokens=4000,
            max_passages=5,
        )

        prompt = "My original prompt."
        enriched = service.enrich_prompt(prompt)

        # Expected: formatter output + \n\n + prompt
        expected_context = format_recall_context(
            results, max_tokens=4000, max_passages=5
        )
        expected = f"{expected_context}\n\n{prompt}"

        assert enriched == expected


class TestMCPServerToolIntegration:
    """Test MCP server tools can be called correctly."""

    @pytest.mark.asyncio
    async def test_semantic_search_with_mocked_pipeline(self):
        """semantic_search tool returns formatted results."""
        from reverie.mcp.server import create_server

        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.return_value = [
            _make_result(1, "Found passage via MCP.", 0.88, "ppr"),
        ]

        with patch("reverie.mcp.server._get_pipeline", return_value=mock_pipeline):
            server = create_server()
            tools = server._tool_manager._tools
            search_fn = tools["semantic_search"].fn

            result = await search_fn(query="test query", top_k=3)

            assert "Found passage via MCP." in result
            assert "0.88" in result

    @pytest.mark.asyncio
    async def test_memory_status_returns_stats(self):
        """memory_status tool returns formatted statistics."""
        from reverie.mcp.server import create_server

        mock_stats = {
            "entities": 100,
            "passages": 50,
            "entity_passages": 200,
            "passage_adjacency": 45,
            "sessions": 4,
            "last_indexed": "2026-02-05T04:00:00",
            "process_memory_mb": 256.5,
        }

        with patch("reverie.mcp.server._get_stats", return_value=mock_stats):
            server = create_server()
            tools = server._tool_manager._tools
            status_fn = tools["memory_status"].fn

            result = await status_fn()

            assert "100" in result  # entities
            assert "50" in result   # passages


class TestRecallDisabledFlow:
    """Test recall when disabled (e.g., in test environments)."""

    def test_disabled_recall_passes_through(self):
        """When recall is disabled, pipeline is never called."""
        pipeline = MagicMock()
        service = RecallService(pipeline=pipeline, enabled=False)

        prompt = "This should pass through unchanged."
        result = service.enrich_prompt(prompt)

        assert result == prompt
        pipeline.retrieve.assert_not_called()

    def test_enabled_recall_calls_pipeline(self):
        """When recall is enabled, pipeline IS called."""
        pipeline = MagicMock()
        pipeline.retrieve.return_value = []
        service = RecallService(pipeline=pipeline, enabled=True)

        service.enrich_prompt("test")

        pipeline.retrieve.assert_called_once()
