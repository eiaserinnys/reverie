"""Repository for passages table.

CRUD operations for session passage chunks with pgvector embedding search.
Supports dual embedding (user/assistant) for harmonic mean scoring.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PassageRow:
    """Data class for a passages table row.

    Supports both legacy (combined text/embedding) and new (user/assistant split)
    storage formats.
    """

    id: int
    session_id: str
    turn_index: int
    text: str  # Legacy: combined turn text (deprecated)
    metadata: dict
    embedding: list[float] | None  # Legacy: combined embedding (deprecated)
    created_at: datetime
    # New dual embedding fields
    user_text: str | None = None
    user_embedding: list[float] | None = None
    assistant_text: str | None = None
    assistant_embedding: list[float] | None = None
    # Profile support (incremental indexing)
    profile_id: str = "default"
    session_seq: int | None = None
    # Lazy PPR support: number of edges from this passage (entities + adjacencies)
    out_degree: int = 0

    @property
    def has_dual_embeddings(self) -> bool:
        """Check if this passage has the new dual embedding format."""
        return self.user_embedding is not None and self.assistant_embedding is not None


@dataclass
class DualSearchResult:
    """Result from dual embedding search."""

    passage_id: int
    user_similarity: float
    assistant_similarity: float
    combined_score: float
    user_text: str | None
    assistant_text: str | None


class PassageRepository:
    """CRUD operations for the passages table."""

    EMBEDDING_DIM = 1536

    def __init__(self, conn):
        """Initialize with a psycopg2 connection."""
        self._conn = conn

    def _validate_embedding(
        self, embedding: list[float] | None, name: str
    ) -> None:
        """Validate embedding dimension.

        Args:
            embedding: Embedding vector to validate.
            name: Name of the embedding for error messages.

        Raises:
            ValueError: If embedding has wrong dimension.
        """
        if embedding is not None and len(embedding) != self.EMBEDDING_DIM:
            raise ValueError(
                f"{name} must have {self.EMBEDDING_DIM} dimensions, "
                f"got {len(embedding)}"
            )

    def insert(
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
        """Insert a passage and return its ID.

        Supports both legacy (text/embedding) and new (user_text/assistant_text
        with separate embeddings) formats.

        Args:
            session_id: Session identifier.
            turn_index: Turn index within session.
            text: Combined turn text (deprecated, for backward compatibility).
            metadata: Optional metadata dict.
            embedding: Combined embedding (deprecated).
            user_text: User message text.
            user_embedding: User message embedding.
            assistant_text: Assistant response text.
            assistant_embedding: Assistant response embedding.
            profile_id: Profile identifier (default: "default").
            session_seq: Session sequence number for display (S{seq}-T{turn}).
            out_degree: Initial out_degree for Lazy PPR (entities + adjacencies).

        Returns:
            Inserted passage ID.

        Raises:
            ValueError: If any embedding has wrong dimension.
        """
        # Validate embedding dimensions
        self._validate_embedding(embedding, "embedding")
        self._validate_embedding(user_embedding, "user_embedding")
        self._validate_embedding(assistant_embedding, "assistant_embedding")

        meta_json = json.dumps(metadata or {})
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO passages (
                    session_id, turn_index, text, metadata, embedding,
                    user_text, user_embedding, assistant_text, assistant_embedding,
                    profile_id, session_seq, out_degree
                )
                VALUES (
                    %s, %s, %s, %s,
                    %s::vector,
                    %s, %s::vector,
                    %s, %s::vector,
                    %s, %s, %s
                )
                RETURNING id
                """,
                (
                    session_id,
                    turn_index,
                    text,
                    meta_json,
                    str(embedding) if embedding else None,
                    user_text,
                    str(user_embedding) if user_embedding else None,
                    assistant_text,
                    str(assistant_embedding) if assistant_embedding else None,
                    profile_id,
                    session_seq,
                    out_degree,
                ),
            )
            return cur.fetchone()[0]

    def get_by_id(self, passage_id: int) -> PassageRow | None:
        """Get a passage by its ID."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, session_id, turn_index, text, metadata, embedding,
                       created_at,
                       user_text, user_embedding, assistant_text, assistant_embedding,
                       profile_id, session_seq, out_degree
                FROM passages WHERE id = %s
                """,
                (passage_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_passage(row)

    def get_by_session(
        self, session_id: str, profile_id: str | None = None
    ) -> list[PassageRow]:
        """Get all passages for a session, ordered by turn_index.

        Args:
            session_id: Session identifier.
            profile_id: Optional profile filter. If None, returns all profiles.

        Returns:
            List of PassageRow ordered by turn_index.
        """
        with self._conn.cursor() as cur:
            if profile_id:
                cur.execute(
                    """
                    SELECT id, session_id, turn_index, text, metadata, embedding,
                           created_at,
                           user_text, user_embedding, assistant_text, assistant_embedding,
                           profile_id, session_seq, out_degree
                    FROM passages
                    WHERE session_id = %s AND profile_id = %s
                    ORDER BY turn_index
                    """,
                    (session_id, profile_id),
                )
            else:
                cur.execute(
                    """
                    SELECT id, session_id, turn_index, text, metadata, embedding,
                           created_at,
                           user_text, user_embedding, assistant_text, assistant_embedding,
                           profile_id, session_seq, out_degree
                    FROM passages WHERE session_id = %s
                    ORDER BY turn_index
                    """,
                    (session_id,),
                )
            return [self._row_to_passage(row) for row in cur.fetchall()]

    def search_by_embedding(
        self, query_embedding: list[float], top_k: int = 10, profile_id: str | None = None
    ) -> list[tuple[PassageRow, float]]:
        """Search passages by cosine similarity using pgvector HNSW index.

        Uses legacy combined embedding for backward compatibility.
        For dual embedding search, use search_by_dual_embedding().

        Args:
            query_embedding: Query embedding vector.
            top_k: Maximum number of results.
            profile_id: Optional profile filter.

        Returns:
            List of (PassageRow, cosine_distance) sorted by distance (ascending).
            Lower distance = more similar.
        """
        with self._conn.cursor() as cur:
            if profile_id:
                cur.execute(
                    """
                    SELECT id, session_id, turn_index, text, metadata, embedding,
                           created_at,
                           user_text, user_embedding, assistant_text, assistant_embedding,
                           profile_id, session_seq, out_degree,
                           embedding <=> %s::vector AS distance
                    FROM passages
                    WHERE embedding IS NOT NULL AND profile_id = %s
                    ORDER BY distance
                    LIMIT %s
                    """,
                    (str(query_embedding), profile_id, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT id, session_id, turn_index, text, metadata, embedding,
                           created_at,
                           user_text, user_embedding, assistant_text, assistant_embedding,
                           profile_id, session_seq, out_degree,
                           embedding <=> %s::vector AS distance
                    FROM passages
                    WHERE embedding IS NOT NULL
                    ORDER BY distance
                    LIMIT %s
                    """,
                    (str(query_embedding), top_k),
                )
            results = []
            for row in cur.fetchall():
                passage = self._row_to_passage(row[:14])
                distance = row[14]
                results.append((passage, distance))
            return results

    def search_by_dual_embedding(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[DualSearchResult]:
        """Search passages using harmonic mean of user/assistant embeddings.

        Requires passages to have both user_embedding and assistant_embedding.
        Uses the search_passages_dual_embedding() database function.

        Args:
            query_embedding: Query embedding vector.
            top_k: Maximum number of results.

        Returns:
            List of DualSearchResult sorted by combined_score descending.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT passage_id, user_similarity, assistant_similarity,
                       combined_score, user_text, assistant_text
                FROM search_passages_dual_embedding(%s::vector, %s)
                """,
                (str(query_embedding), top_k),
            )
            return [
                DualSearchResult(
                    passage_id=row[0],
                    user_similarity=row[1],
                    assistant_similarity=row[2],
                    combined_score=row[3],
                    user_text=row[4],
                    assistant_text=row[5],
                )
                for row in cur.fetchall()
            ]

    def get_by_ids(self, passage_ids: list[int]) -> list[PassageRow]:
        """Get multiple passages by their IDs in a single query."""
        if not passage_ids:
            return []
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, session_id, turn_index, text, metadata, embedding,
                       created_at,
                       user_text, user_embedding, assistant_text, assistant_embedding,
                       profile_id, session_seq, out_degree
                FROM passages WHERE id = ANY(%s)
                ORDER BY id
                """,
                (list(passage_ids),),
            )
            return [self._row_to_passage(row) for row in cur.fetchall()]

    def get_all(self, profile_id: str | None = None) -> list[PassageRow]:
        """Get all passages, ordered by session_id and turn_index.

        Args:
            profile_id: Optional profile filter. If None, returns all profiles.

        Returns:
            List of PassageRow ordered by session_id and turn_index.
        """
        with self._conn.cursor() as cur:
            if profile_id:
                cur.execute(
                    """
                    SELECT id, session_id, turn_index, text, metadata, embedding,
                           created_at,
                           user_text, user_embedding, assistant_text, assistant_embedding,
                           profile_id, session_seq, out_degree
                    FROM passages
                    WHERE profile_id = %s
                    ORDER BY session_id, turn_index
                    """,
                    (profile_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, session_id, turn_index, text, metadata, embedding,
                           created_at,
                           user_text, user_embedding, assistant_text, assistant_embedding,
                           profile_id, session_seq, out_degree
                    FROM passages
                    ORDER BY session_id, turn_index
                    """
                )
            return [self._row_to_passage(row) for row in cur.fetchall()]

    def get_migration_status(self) -> dict:
        """Get passage migration status for dual embeddings.

        Returns:
            Dict with total, migrated, pending counts and percentage.
        """
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM get_passage_migration_status()")
            row = cur.fetchone()
            return {
                "total_passages": row[0],
                "migrated_passages": row[1],
                "pending_passages": row[2],
                "migration_percentage": row[3],
            }

    def delete_by_session(self, session_id: str) -> int:
        """Delete all passages for a session and update related out_degree values.

        Also decreases out_degree of entities that were linked to deleted passages,
        and handles adjacency cleanup (adjacencies are cascade-deleted by FK,
        but we need to update out_degree of remaining adjacent passages first).

        Args:
            session_id: Session identifier.

        Returns:
            Number of deleted passages.
        """
        with self._conn.cursor() as cur:
            # Get passage IDs to be deleted
            cur.execute(
                "SELECT id FROM passages WHERE session_id = %s",
                (session_id,),
            )
            passage_ids = [row[0] for row in cur.fetchall()]

            if not passage_ids:
                return 0

            # Decrease out_degree of entities linked to these passages
            cur.execute(
                """
                UPDATE entities e
                SET out_degree = out_degree - counts.cnt
                FROM (
                    SELECT entity_id, COUNT(*) as cnt
                    FROM entity_passages
                    WHERE passage_id = ANY(%s)
                    GROUP BY entity_id
                ) counts
                WHERE e.id = counts.entity_id
                """,
                (passage_ids,),
            )

            # Decrease out_degree of passages adjacent to these passages
            # (for passages that are NOT being deleted)
            cur.execute(
                """
                UPDATE passages p
                SET out_degree = out_degree - counts.cnt
                FROM (
                    SELECT
                        CASE
                            WHEN passage_id_a = ANY(%s) THEN passage_id_b
                            ELSE passage_id_a
                        END as adjacent_pid,
                        COUNT(*) as cnt
                    FROM passage_adjacency
                    WHERE (passage_id_a = ANY(%s) OR passage_id_b = ANY(%s))
                      AND NOT (passage_id_a = ANY(%s) AND passage_id_b = ANY(%s))
                    GROUP BY adjacent_pid
                ) counts
                WHERE p.id = counts.adjacent_pid
                """,
                (passage_ids, passage_ids, passage_ids, passage_ids, passage_ids),
            )

            # Delete entity_passages links (cascade will handle, but explicit for clarity)
            cur.execute(
                "DELETE FROM entity_passages WHERE passage_id = ANY(%s)",
                (passage_ids,),
            )

            # Delete adjacencies (cascade will handle, but explicit for clarity)
            cur.execute(
                """
                DELETE FROM passage_adjacency
                WHERE passage_id_a = ANY(%s) OR passage_id_b = ANY(%s)
                """,
                (passage_ids, passage_ids),
            )

            # Delete passages
            cur.execute(
                "DELETE FROM passages WHERE session_id = %s",
                (session_id,),
            )
            deleted = cur.rowcount

            return deleted

    def update_out_degree(self, passage_id: int, delta: int) -> None:
        """Update out_degree of a passage by a delta value.

        Args:
            passage_id: Passage ID.
            delta: Value to add to out_degree (can be negative).
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE passages SET out_degree = out_degree + %s WHERE id = %s",
                (delta, passage_id),
            )

    def _row_to_passage(self, row: tuple) -> PassageRow:
        """Convert a database row tuple to PassageRow.

        Expected row format (14 columns):
        0: id, 1: session_id, 2: turn_index, 3: text, 4: metadata, 5: embedding,
        6: created_at, 7: user_text, 8: user_embedding, 9: assistant_text, 10: assistant_embedding,
        11: profile_id, 12: session_seq, 13: out_degree
        """
        meta = row[4]
        if isinstance(meta, str):
            meta = json.loads(meta)
        embedding = row[5]
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        user_embedding = row[8] if len(row) > 8 else None
        if isinstance(user_embedding, str):
            user_embedding = json.loads(user_embedding)
        assistant_embedding = row[10] if len(row) > 10 else None
        if isinstance(assistant_embedding, str):
            assistant_embedding = json.loads(assistant_embedding)

        return PassageRow(
            id=row[0],
            session_id=row[1],
            turn_index=row[2],
            text=row[3],
            metadata=meta or {},
            embedding=embedding,
            created_at=row[6],
            user_text=row[7] if len(row) > 7 else None,
            user_embedding=user_embedding,
            assistant_text=row[9] if len(row) > 9 else None,
            assistant_embedding=assistant_embedding,
            profile_id=row[11] if len(row) > 11 and row[11] else "default",
            session_seq=row[12] if len(row) > 12 else None,
            out_degree=row[13] if len(row) > 13 and row[13] is not None else 0,
        )
