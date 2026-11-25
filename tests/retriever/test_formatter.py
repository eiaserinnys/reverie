"""Tests for recall context formatter."""

import pytest

from reverie.retriever.formatter import format_recall_context
from reverie.retriever.pipeline_v2 import RetrievalResult


def _make_result(passage_id, text, score=0.8, source="ppr"):
    return RetrievalResult(
        passage_id=passage_id,
        text=text,
        score=score,
        source=source,
    )


class TestFormatRecallContext:
    """Test format_recall_context function."""

    def test_basic_formatting(self):
        """Formats results with XML tags and metadata."""
        results = [
            _make_result(1, "Agent was restarted via systemctl.", 0.85, "ppr"),
        ]

        output = format_recall_context(results, max_tokens=4000)

        assert "<long-term-memory>" in output
        assert "</long-term-memory>" in output
        assert "Agent was restarted via systemctl." in output
        assert "0.85" in output
        assert "ppr" in output

    def test_empty_results(self):
        """Empty results returns empty string."""
        output = format_recall_context([], max_tokens=4000)
        assert output == ""

    def test_max_passages_limit(self):
        """Respects max_passages parameter."""
        results = [_make_result(i, f"Passage {i}", 0.9 - i * 0.1) for i in range(10)]

        output = format_recall_context(results, max_tokens=100000, max_passages=3)

        # Should only include 3 passages
        assert "Passage 0" in output
        assert "Passage 1" in output
        assert "Passage 2" in output
        assert "Passage 3" not in output

    def test_token_budget_truncation(self):
        """Stops adding passages when token budget is exceeded."""
        # Each passage ~10 tokens, budget allows ~2
        results = [
            _make_result(1, "Short passage one.", 0.9),
            _make_result(2, "Short passage two.", 0.8),
            _make_result(3, "Short passage three.", 0.7),
            _make_result(4, "Short passage four.", 0.6),
            _make_result(5, "Short passage five.", 0.5),
        ]

        # Very tight budget — should include at least 1 but not all
        output = format_recall_context(results, max_tokens=50, max_passages=10)

        # At least one passage should be included
        assert "Short passage one." in output
        # Should not include all 5
        assert output.count("Short passage") < 5

    def test_results_ordered_by_score(self):
        """Results appear in score order."""
        results = [
            _make_result(1, "First (highest)", 0.95),
            _make_result(2, "Second", 0.80),
            _make_result(3, "Third (lowest)", 0.65),
        ]

        output = format_recall_context(results, max_tokens=4000)

        pos1 = output.index("First (highest)")
        pos2 = output.index("Second")
        pos3 = output.index("Third (lowest)")
        assert pos1 < pos2 < pos3

    def test_mixed_sources(self):
        """Handles mixed ppr and knn sources."""
        results = [
            _make_result(1, "PPR result", 0.85, "ppr"),
            _make_result(2, "KNN result", 0.72, "knn"),
        ]

        output = format_recall_context(results, max_tokens=4000)

        assert "ppr" in output
        assert "knn" in output

    def test_single_passage_always_included(self):
        """Even with tiny budget, at least one passage is included."""
        results = [
            _make_result(1, "The only passage here.", 0.9),
        ]

        output = format_recall_context(results, max_tokens=10)

        assert "The only passage here." in output
