"""Recall context formatter.

Formats retrieval results into a context block for prompt enrichment,
with token budget management and deduplication.
"""

import logging
from typing import Any

from reverie.config import get_settings
from reverie.parser.token_counter import count_tokens

logger = logging.getLogger(__name__)

_HEADER = (
    "<long-term-memory>\n"
    "The following passages were retrieved from long-term memory\n"
    "based on relevance to your current prompt.\n"
)
_FOOTER = "</long-term-memory>"


def format_recall_context(
    results: list[Any],  # RetrievalResult from v1 or v2 pipeline
    max_tokens: int | None = None,
    max_passages: int | None = None,
) -> str:
    """Format retrieval results into a context block for prompt injection.

    Adds passages in score order until token budget or passage limit is reached.
    Always includes at least one passage if results are non-empty.

    Args:
        results: Retrieval results sorted by score descending.
                 Each result should have .score, .source, and .text attributes.
        max_tokens: Maximum tokens for the entire context block. Uses config default if None.
        max_passages: Maximum number of passages to include. Uses config default if None.

    Returns:
        Formatted context string, or empty string if no results.
    """
    if not results:
        return ""

    # Get defaults from config if not provided
    settings = get_settings()
    if max_tokens is None:
        max_tokens = settings.recall_max_tokens
    if max_passages is None:
        max_passages = settings.recall_max_passages

    header_tokens = count_tokens(_HEADER)
    footer_tokens = count_tokens(_FOOTER)
    overhead = header_tokens + footer_tokens

    budget = max(0, max_tokens - overhead)
    used_tokens = 0
    blocks: list[str] = []

    for i, r in enumerate(results):
        if len(blocks) >= max_passages:
            break

        entry = (
            f"[{i + 1}] (score: {r.score:.2f}, source: {r.source})\n"
            f"{r.text}\n"
        )
        entry_tokens = count_tokens(entry)

        # Always include at least the first passage
        if len(blocks) > 0 and used_tokens + entry_tokens > budget:
            break

        blocks.append(entry)
        used_tokens += entry_tokens

    if not blocks:
        return ""

    return _HEADER + "\n" + "\n".join(blocks) + "\n" + _FOOTER
