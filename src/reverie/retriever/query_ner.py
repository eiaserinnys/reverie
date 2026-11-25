"""Query NER extraction for seed entity matching.

Extracts named entities from user queries to find seed entities in the graph.
Reuses OpenIE NER functionality for consistency.
"""

import logging
from typing import Optional

from reverie.indexer.openie import OpenIEClient

logger = logging.getLogger(__name__)


class QueryNERClient:
    """Extract named entities from user queries.

    Lightweight wrapper around OpenIEClient for query-time entity extraction.
    Uses a lightweight prompt (~400 tokens) optimized for search queries,
    ~10x smaller than the full indexing prompt (~4000 tokens).
    """

    def __init__(self, openie_client: Optional[OpenIEClient] = None):
        """Initialize QueryNERClient.

        Args:
            openie_client: Optional OpenIEClient instance.
                If None, creates a new one with default settings.
        """
        self._client = openie_client or OpenIEClient()

    def extract_entities(self, query: str) -> list[str]:
        """Extract named entities from a query.

        Uses lightweight query NER prompt for faster extraction.

        Args:
            query: User query string.

        Returns:
            List of extracted entity names.
        """
        if not query or not query.strip():
            return []

        entities = self._client.extract_query_entities(query)
        logger.debug("Query NER extracted %d entities from: %.50s...", len(entities), query)
        return entities
