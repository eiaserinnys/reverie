"""MemoryLink v2 retrieval pipeline.

Implements LinearRAG-based retrieval with:
- Semantic Propagation: Query → Entity activation via embedding similarity
- BipartitePPR: Random walk on Entity↔Passage bipartite graph
"""

from reverie.core.bipartite_ppr import BipartitePPR
from reverie.retriever.formatter import format_recall_context
from reverie.retriever.pipeline_v2 import MemoryLinkV2Pipeline, RetrievalResult
from reverie.retriever.query_ner import QueryNERClient
from reverie.core.semantic_prop import SemanticPropagation

__all__ = [
    "MemoryLinkV2Pipeline",
    "RetrievalResult",
    "BipartitePPR",
    "SemanticPropagation",
    "QueryNERClient",
    "format_recall_context",
]
