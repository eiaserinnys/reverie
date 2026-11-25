"""Recall service for prompt enrichment.

Retrieves relevant long-term memory passages and injects them
as context before the original prompt. Fails gracefully - if
retrieval fails for any reason, the original prompt is returned
unmodified.
"""

import logging
from typing import Any, Protocol

from reverie.retriever.formatter import format_recall_context

logger = logging.getLogger(__name__)


class RetrievalPipelineProtocol(Protocol):
    """Protocol for retrieval pipelines (v1 and v2 compatible)."""

    def retrieve(self, query: str, top_k: int = 10) -> list[Any]:
        """Retrieve passages relevant to query."""
        ...


class RecallService:
    """Enriches prompts with relevant long-term memory context.

    Usage:
        service = RecallService(pipeline=pipeline)
        enriched = service.enrich_prompt("What happened yesterday?")
        # enriched = "<long-term-memory>...</long-term-memory>\n\nWhat happened yesterday?"
    """

    def __init__(
        self,
        pipeline: RetrievalPipelineProtocol,
        max_tokens: int = 4000,
        max_passages: int = 5,
        enabled: bool = True,
    ):
        """Initialize RecallService.

        Args:
            pipeline: Retrieval pipeline instance.
            max_tokens: Token budget for the context block.
            max_passages: Maximum number of passages to include.
            enabled: Whether recall is active. When False, enrich_prompt
                     returns the original prompt without calling the pipeline.
        """
        self._pipeline = pipeline
        self._max_tokens = max_tokens
        self._max_passages = max_passages
        self.enabled = enabled

    def enrich_prompt(self, prompt: str) -> str:
        """Enrich a prompt with relevant long-term memory context.

        Retrieves passages from the knowledge graph, formats them
        into a context block, and prepends it to the prompt.

        If disabled, retrieval fails, or no results are found,
        returns the original prompt unmodified.

        Args:
            prompt: The original user/task prompt.

        Returns:
            Enriched prompt with memory context, or original prompt.
        """
        if not self.enabled:
            return prompt

        try:
            results = self._pipeline.retrieve(prompt, top_k=self._max_passages)
        except Exception:
            logger.warning("Recall retrieval failed, returning original prompt")
            return prompt

        if not results:
            return prompt

        try:
            context = format_recall_context(
                results,
                max_tokens=self._max_tokens,
                max_passages=self._max_passages,
            )
        except Exception:
            logger.warning("Recall formatting failed, returning original prompt")
            return prompt

        if not context:
            return prompt

        return f"{context}\n\n{prompt}"
