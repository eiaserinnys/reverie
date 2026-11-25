"""ReverieDB Protocol - Abstract database interface for Reverie.

Defines the contract for all database backends (PostgreSQL, SQLite).
Business logic code should only interact with this interface.
"""

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass
class EntityData:
    """Entity data transfer object."""

    id: int
    name: str
    embedding: list[float] | None
    created_at: datetime
    out_degree: int = 0


@dataclass
class PassageData:
    """Passage data transfer object."""

    id: int
    session_id: str
    turn_index: int
    text: str
    metadata: dict
    embedding: list[float] | None
    created_at: datetime
    user_text: str | None = None
    user_embedding: list[float] | None = None
    assistant_text: str | None = None
    assistant_embedding: list[float] | None = None
    profile_id: str = "default"
    session_seq: int | None = None
    out_degree: int = 0


@dataclass
class DualSearchResult:
    """Result from dual embedding search."""

    passage_id: int
    user_similarity: float
    assistant_similarity: float
    combined_score: float
    user_text: str | None
    assistant_text: str | None


@dataclass
class PPRNeighborEdge:
    """A single edge in the PPR graph."""

    edge_type: str  # "e2p", "p2e", "p2p_a", "p2p_b"
    source_id: int
    target_id: int
    out_degree: int


@dataclass
class SessionIndexState:
    """Session indexing state."""

    profile_id: str
    session_id: str
    session_seq: int
    file_path: str | None
    file_mtime: datetime | None
    file_size_bytes: int
    file_lines_count: int
    last_indexed_turn: int
    last_indexed_line: int
    last_passage_id: int | None
    passages_indexed: int
    passages_dropped: int
    status: str  # "pending", "in_progress", "completed", "failed"
    error_message: str | None
    created_at: datetime
    updated_at: datetime


@runtime_checkable
class ReverieDB(Protocol):
    """Abstract database interface for Reverie.

    All database operations should go through this interface.
    Implementations: PostgresDB, SqliteDB (future).
    """

    # ========== Connection Management ==========

    @abstractmethod
    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    @abstractmethod
    def rollback(self) -> None:
        """Rollback the current transaction."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        ...

    @property
    @abstractmethod
    def closed(self) -> bool:
        """Check if the connection is closed."""
        ...

    # ========== Entity Operations ==========

    @abstractmethod
    def upsert_entity(
        self,
        name: str,
        embedding: list[float] | None = None,
    ) -> int:
        """Insert or update an entity. Returns entity ID."""
        ...

    @abstractmethod
    def upsert_entity_with_similarity(
        self,
        name: str,
        embedding: list[float] | None = None,
        similarity_threshold: float | None = None,
    ) -> int:
        """Insert or update entity with embedding similarity matching."""
        ...

    @abstractmethod
    def get_entity_by_id(self, entity_id: int) -> EntityData | None:
        """Get entity by ID."""
        ...

    @abstractmethod
    def get_entity_by_name(self, name: str) -> EntityData | None:
        """Get entity by name (exact match)."""
        ...

    @abstractmethod
    def resolve_entity_alias(self, name: str) -> EntityData | None:
        """Resolve an alias to its canonical entity."""
        ...

    @abstractmethod
    def resolve_entity_aliases_batch(
        self, names: list[str]
    ) -> dict[str, EntityData]:
        """Resolve multiple aliases to canonical entities."""
        ...

    @abstractmethod
    def get_entities_by_ids(self, entity_ids: list[int]) -> list[EntityData]:
        """Get multiple entities by IDs."""
        ...

    @abstractmethod
    def search_entities_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[EntityData, float]]:
        """Search entities by embedding similarity.

        Returns list of (entity, distance) tuples.
        """
        ...

    @abstractmethod
    def find_entity_seeds(
        self,
        query_embedding: list[float],
        tau: float,
        initial_k: int = 100,
        max_k: int = 100000,
    ) -> list[tuple[int, float]]:
        """Find entity seeds with similarity >= tau.

        Returns list of (entity_id, similarity) tuples.
        """
        ...

    @abstractmethod
    def update_entity_out_degree(self, entity_id: int, delta: int) -> None:
        """Update entity out_degree by delta."""
        ...

    @abstractmethod
    def count_entities(self) -> int:
        """Count total entities."""
        ...

    @abstractmethod
    def list_entities(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_field: str = "id",
        sort_order: str = "desc",
    ) -> tuple[list[dict], int]:
        """List entities with pagination.

        Returns (entities, total_count).
        """
        ...

    # ========== Passage Operations ==========

    @abstractmethod
    def insert_passage(
        self,
        session_id: str,
        turn_index: int,
        text: str,
        metadata: dict | None = None,
        embedding: list[float] | None = None,
        user_text: str | None = None,
        user_embedding: list[float] | None = None,
        assistant_text: str | None = None,
        assistant_embedding: list[float] | None = None,
        profile_id: str = "default",
        session_seq: int | None = None,
        out_degree: int = 0,
    ) -> int:
        """Insert a passage. Returns passage ID."""
        ...

    @abstractmethod
    def get_passage_by_id(self, passage_id: int) -> PassageData | None:
        """Get passage by ID."""
        ...

    @abstractmethod
    def get_passages_by_ids(self, passage_ids: list[int]) -> list[PassageData]:
        """Get multiple passages by IDs."""
        ...

    @abstractmethod
    def get_passages_by_session(
        self,
        session_id: str,
        profile_id: str | None = None,
    ) -> list[PassageData]:
        """Get passages for a session."""
        ...

    @abstractmethod
    def search_passages_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        profile_id: str | None = None,
    ) -> list[tuple[PassageData, float]]:
        """Search passages by embedding.

        Returns list of (passage, distance) tuples.
        """
        ...

    @abstractmethod
    def search_passages_dual_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[DualSearchResult]:
        """Search passages using dual embeddings."""
        ...

    @abstractmethod
    def delete_passages_by_session(self, session_id: str) -> int:
        """Delete passages for a session. Returns deleted count."""
        ...

    @abstractmethod
    def update_passage_out_degree(self, passage_id: int, delta: int) -> None:
        """Update passage out_degree by delta."""
        ...

    @abstractmethod
    def count_passages(self) -> int:
        """Count total passages."""
        ...

    @abstractmethod
    def list_passages(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_field: str = "id",
        sort_order: str = "desc",
    ) -> tuple[list[dict], int]:
        """List passages with pagination.

        Returns (passages, total_count).
        """
        ...

    # ========== Entity-Passage Link Operations ==========

    @abstractmethod
    def link_entity_passage(self, entity_id: int, passage_id: int) -> bool:
        """Create entity-passage link.

        Returns True if link was created, False if already exists.
        """
        ...

    @abstractmethod
    def get_entity_passages(
        self,
        entity_id: int,
        limit: int = 50,
    ) -> list[dict]:
        """Get passages containing an entity."""
        ...

    @abstractmethod
    def get_passage_entities(self, passage_id: int) -> list[dict]:
        """Get entities in a passage."""
        ...

    @abstractmethod
    def get_passage_entity_ids(
        self,
        passage_ids: list[int],
    ) -> dict[int, set[int]]:
        """Get entity IDs for multiple passages.

        Returns dict mapping passage_id to set of entity_ids.
        """
        ...

    @abstractmethod
    def count_entity_passages(self) -> int:
        """Count total entity-passage links."""
        ...

    # ========== Adjacency Operations ==========

    @abstractmethod
    def create_adjacency(self, passage_id_a: int, passage_id_b: int) -> None:
        """Create passage adjacency (bidirectional)."""
        ...

    @abstractmethod
    def get_adjacent_passages(self, passage_id: int) -> list[dict]:
        """Get passages adjacent to a passage."""
        ...

    @abstractmethod
    def count_adjacencies(self) -> int:
        """Count total adjacencies."""
        ...

    # ========== PPR Graph Traversal ==========

    @abstractmethod
    def get_ppr_neighbors(
        self,
        active_entity_ids: list[int],
        active_passage_ids: list[int],
    ) -> list[PPRNeighborEdge]:
        """Get all PPR neighbor edges for active nodes."""
        ...

    @abstractmethod
    def run_ppr_server_side(
        self,
        seed_ids: list[int],
        seed_scores: list[float],
        damping: float,
        max_iter: int,
        min_score: float,
        tol: float,
    ) -> list[tuple[str, int, float]]:
        """Run PPR on server side (if supported).

        Returns list of (node_type, node_id, score) tuples.
        """
        ...

    # ========== Session State Operations ==========

    @abstractmethod
    def get_session_state(
        self,
        profile_id: str,
        session_id: str,
    ) -> SessionIndexState | None:
        """Get session indexing state."""
        ...

    @abstractmethod
    def upsert_session_state(
        self,
        profile_id: str,
        session_id: str,
        file_path: str,
        file_mtime: datetime,
        file_size_bytes: int,
        file_lines_count: int,
        session_seq: int | None = None,
    ) -> SessionIndexState:
        """Create or update session state."""
        ...

    @abstractmethod
    def update_session_progress(
        self,
        profile_id: str,
        session_id: str,
        last_indexed_turn: int,
        last_indexed_line: int,
        last_passage_id: int | None,
        passages_indexed_delta: int = 0,
        passages_dropped_delta: int = 0,
    ) -> None:
        """Update session indexing progress."""
        ...

    @abstractmethod
    def mark_session_completed(self, profile_id: str, session_id: str) -> None:
        """Mark session as completed."""
        ...

    @abstractmethod
    def mark_session_failed(
        self,
        profile_id: str,
        session_id: str,
        error_message: str,
    ) -> None:
        """Mark session as failed."""
        ...

    @abstractmethod
    def reset_session_for_reindex(
        self,
        profile_id: str,
        session_id: str,
    ) -> None:
        """Reset session state for full reindex."""
        ...

    @abstractmethod
    def get_session_stats(self, profile_id: str) -> dict:
        """Get session indexing statistics."""
        ...

    # ========== Processing History ==========

    @abstractmethod
    def record_processing_history(
        self,
        profile_id: str,
        session_id: str,
        turn_index: int,
        action: str,
        drop_reason: str | None = None,
        raw_text: str | None = None,
        preprocessed_text: str | None = None,
        entities_extracted: list[str] | None = None,
        passage_id: int | None = None,
    ) -> None:
        """Record passage processing history."""
        ...

    @abstractmethod
    def delete_processing_history(
        self,
        profile_id: str,
        session_id: str,
    ) -> int:
        """Delete processing history for a session."""
        ...

    # ========== Statistics ==========

    @abstractmethod
    def get_stats(self) -> dict:
        """Get database statistics."""
        ...

    @abstractmethod
    def count_distinct_sessions(self) -> int:
        """Count distinct sessions."""
        ...

    @abstractmethod
    def get_last_indexed_time(self) -> datetime | None:
        """Get last indexing timestamp."""
        ...
