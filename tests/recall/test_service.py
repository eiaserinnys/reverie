"""Tests for RecallService — prompt enrichment with long-term memory."""

from unittest.mock import MagicMock, patch

import pytest

from reverie.recall.service import RecallService
from reverie.retriever.pipeline_v2 import RetrievalResult


def _make_result(passage_id, text, score=0.8, source="ppr"):
    return RetrievalResult(
        passage_id=passage_id,
        text=text,
        score=score,
        source=source,
    )


class TestRecallServiceEnrich:
    """Test RecallService.enrich_prompt()."""

    @pytest.fixture
    def mock_pipeline(self):
        """Mock RetrievalPipeline that returns two results."""
        pipeline = MagicMock()
        pipeline.retrieve.return_value = [
            _make_result(1, "Agent was restarted via systemctl.", 0.85, "ppr"),
            _make_result(2, "pgvector extension installed.", 0.72, "knn"),
        ]
        return pipeline

    @pytest.fixture
    def service(self, mock_pipeline):
        """RecallService with mocked pipeline."""
        return RecallService(pipeline=mock_pipeline)

    def test_enrich_adds_context_before_prompt(self, service):
        """Enriched prompt has memory context before original prompt."""
        prompt = "What happened with Agent recently?"
        result = service.enrich_prompt(prompt)

        assert "<long-term-memory>" in result
        assert "</long-term-memory>" in result
        # Original prompt comes after the context block
        assert result.index("</long-term-memory>") < result.index(prompt)

    def test_enrich_contains_passage_text(self, service):
        """Enriched prompt contains retrieved passage text."""
        result = service.enrich_prompt("test query")

        assert "Agent was restarted via systemctl." in result
        assert "pgvector extension installed." in result

    def test_enrich_contains_original_prompt(self, service):
        """Original prompt is preserved in the enriched output."""
        prompt = "Tell me about the deployment."
        result = service.enrich_prompt(prompt)

        assert prompt in result

    def test_enrich_calls_pipeline_with_query(self, service, mock_pipeline):
        """Pipeline.retrieve() is called with the original prompt."""
        service.enrich_prompt("my query")

        mock_pipeline.retrieve.assert_called_once_with("my query", top_k=5)

    def test_enrich_respects_top_k_config(self, mock_pipeline):
        """RecallService passes configured top_k to pipeline."""
        service = RecallService(pipeline=mock_pipeline, max_passages=3)
        service.enrich_prompt("query")

        mock_pipeline.retrieve.assert_called_once_with("query", top_k=3)


class TestRecallServiceEmpty:
    """Test RecallService when no results are found."""

    def test_enrich_returns_original_when_no_results(self):
        """Returns original prompt when pipeline returns empty results."""
        pipeline = MagicMock()
        pipeline.retrieve.return_value = []
        service = RecallService(pipeline=pipeline)

        prompt = "My original prompt."
        result = service.enrich_prompt(prompt)

        assert result == prompt

    def test_enrich_no_xml_tags_when_empty(self):
        """No XML tags when there are no results."""
        pipeline = MagicMock()
        pipeline.retrieve.return_value = []
        service = RecallService(pipeline=pipeline)

        result = service.enrich_prompt("test")

        assert "<long-term-memory>" not in result


class TestRecallServiceGracefulDegradation:
    """Test RecallService error handling."""

    def test_pipeline_exception_returns_original(self):
        """If pipeline raises, original prompt is returned."""
        pipeline = MagicMock()
        pipeline.retrieve.side_effect = Exception("DB connection failed")
        service = RecallService(pipeline=pipeline)

        prompt = "My important prompt."
        result = service.enrich_prompt(prompt)

        assert result == prompt

    def test_pipeline_exception_is_logged(self):
        """Pipeline exceptions are logged."""
        pipeline = MagicMock()
        pipeline.retrieve.side_effect = Exception("DB connection failed")
        service = RecallService(pipeline=pipeline)

        with patch("reverie.recall.service.logger") as mock_logger:
            service.enrich_prompt("test")
            mock_logger.warning.assert_called_once()

    def test_formatter_exception_returns_original(self):
        """If formatter raises, original prompt is returned."""
        pipeline = MagicMock()
        pipeline.retrieve.return_value = [
            _make_result(1, "Some text.", 0.9),
        ]
        service = RecallService(pipeline=pipeline)

        with patch("reverie.recall.service.format_recall_context", side_effect=Exception("format error")):
            prompt = "My prompt."
            result = service.enrich_prompt(prompt)
            assert result == prompt


class TestRecallServiceDisabled:
    """Test RecallService on/off toggle."""

    def test_disabled_returns_original(self):
        """When disabled, returns original prompt without calling pipeline."""
        pipeline = MagicMock()
        service = RecallService(pipeline=pipeline, enabled=False)

        prompt = "My prompt."
        result = service.enrich_prompt(prompt)

        assert result == prompt
        pipeline.retrieve.assert_not_called()

    def test_enabled_by_default(self):
        """Service is enabled by default."""
        pipeline = MagicMock()
        pipeline.retrieve.return_value = []
        service = RecallService(pipeline=pipeline)

        assert service.enabled is True
