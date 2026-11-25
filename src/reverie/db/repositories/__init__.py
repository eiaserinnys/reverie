"""Database repository modules for reverie.

Provides CRUD operations for passages, entities, and their mappings.
Uses MemoryLink v2 schema with entity_passages and passage_adjacency tables.
"""

from reverie.db.repositories.adjacency_repo import AdjacencyRepository
from reverie.db.repositories.entity_passage_repo import EntityPassageRepository
from reverie.db.repositories.entity_repo import EntityRepository
from reverie.db.repositories.passage_history_repo import PassageProcessingHistoryRepository
from reverie.db.repositories.passage_repo import PassageRepository
from reverie.db.repositories.ppr_neighbor_repo import PPRNeighborRepository
from reverie.db.repositories.session_index_state_repo import SessionIndexStateRepository

__all__ = [
    "PassageRepository",
    "EntityRepository",
    "EntityPassageRepository",
    "AdjacencyRepository",
    "PPRNeighborRepository",
    "SessionIndexStateRepository",
    "PassageProcessingHistoryRepository",
]
