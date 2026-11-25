"""Repository for passage_processing_history table.

Records the processing history of each turn, including drop reasons
and extracted entities for debugging and analytics.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PassageHistoryRow:
    """Data class for a passage_processing_history table row."""

    id: int
    profile_id: str
    session_id: str
    turn_index: int
    indexing_path: str
    action: str  # 'indexed', 'dropped', 'skipped', 'failed'
    drop_reason: str | None
    raw_text: str | None
    preprocessed_text: str | None
    entities_extracted: list[str] | None
    passage_id: int | None
    processed_at: datetime


class PassageProcessingHistoryRepository:
    """CRUD operations for the passage_processing_history table."""

    def __init__(self, conn):
        """Initialize with a psycopg2 connection."""
        self._conn = conn

    def record(
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
        indexing_path: str = "",
    ) -> int:
        """Record a turn processing event.

        Uses ON CONFLICT to update existing records for the same turn.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            turn_index: Turn index within session.
            action: Processing action ('indexed', 'dropped', 'skipped', 'failed').
            drop_reason: Reason for dropping (if action='dropped').
            raw_text: Original turn text (truncated).
            preprocessed_text: Preprocessed text (truncated).
            entities_extracted: List of extracted entity names.
            passage_id: ID of created passage (if action='indexed').
            indexing_path: Folder path being indexed (default: empty string).

        Returns:
            Record ID.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO passage_processing_history (
                    profile_id, session_id, turn_index, indexing_path, action,
                    drop_reason, raw_text, preprocessed_text,
                    entities_extracted, passage_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (profile_id, session_id, turn_index, indexing_path) DO UPDATE SET
                    action = EXCLUDED.action,
                    drop_reason = EXCLUDED.drop_reason,
                    raw_text = EXCLUDED.raw_text,
                    preprocessed_text = EXCLUDED.preprocessed_text,
                    entities_extracted = EXCLUDED.entities_extracted,
                    passage_id = EXCLUDED.passage_id,
                    processed_at = NOW()
                RETURNING id
                """,
                (
                    profile_id,
                    session_id,
                    turn_index,
                    indexing_path,
                    action,
                    drop_reason,
                    raw_text,
                    preprocessed_text,
                    entities_extracted,
                    passage_id,
                ),
            )
            return cur.fetchone()[0]

    def get_by_session(
        self,
        profile_id: str,
        session_id: str,
        limit: int | None = None,
        indexing_path: str | None = None,
    ) -> list[PassageHistoryRow]:
        """Get all processing history for a session.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            limit: Maximum number of records (None = no limit).
            indexing_path: Optional indexing path filter.

        Returns:
            List of PassageHistoryRow ordered by turn_index.
        """
        with self._conn.cursor() as cur:
            base_query = """
                SELECT id, profile_id, session_id, turn_index, indexing_path, action,
                       drop_reason, raw_text, preprocessed_text,
                       entities_extracted, passage_id, processed_at
                FROM passage_processing_history
                WHERE profile_id = %s AND session_id = %s
            """
            params: list = [profile_id, session_id]

            if indexing_path is not None:
                base_query += " AND indexing_path = %s"
                params.append(indexing_path)

            if limit:
                base_query += " ORDER BY turn_index DESC LIMIT %s"
                params.append(limit)
            else:
                base_query += " ORDER BY turn_index"

            cur.execute(base_query, params)
            return [self._row_to_history(row) for row in cur.fetchall()]

    def get_by_turn(
        self,
        profile_id: str,
        session_id: str,
        turn_index: int,
        indexing_path: str = "",
    ) -> PassageHistoryRow | None:
        """Get processing history for a specific turn.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            turn_index: Turn index.
            indexing_path: Folder path being indexed (default: empty string).

        Returns:
            PassageHistoryRow if found, None otherwise.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, profile_id, session_id, turn_index, indexing_path, action,
                       drop_reason, raw_text, preprocessed_text,
                       entities_extracted, passage_id, processed_at
                FROM passage_processing_history
                WHERE profile_id = %s AND session_id = %s AND turn_index = %s
                      AND indexing_path = %s
                """,
                (profile_id, session_id, turn_index, indexing_path),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_history(row)

    def get_dropped(
        self, profile_id: str, limit: int = 100, indexing_path: str | None = None
    ) -> list[PassageHistoryRow]:
        """Get recently dropped turns for a profile.

        Args:
            profile_id: Profile identifier.
            limit: Maximum number of records.
            indexing_path: Optional indexing path filter.

        Returns:
            List of dropped PassageHistoryRow.
        """
        with self._conn.cursor() as cur:
            base_query = """
                SELECT id, profile_id, session_id, turn_index, indexing_path, action,
                       drop_reason, raw_text, preprocessed_text,
                       entities_extracted, passage_id, processed_at
                FROM passage_processing_history
                WHERE profile_id = %s AND action = 'dropped'
            """
            params: list = [profile_id]

            if indexing_path is not None:
                base_query += " AND indexing_path = %s"
                params.append(indexing_path)

            base_query += " ORDER BY processed_at DESC LIMIT %s"
            params.append(limit)

            cur.execute(base_query, params)
            return [self._row_to_history(row) for row in cur.fetchall()]

    def get_failed(
        self, profile_id: str, limit: int = 100, indexing_path: str | None = None
    ) -> list[PassageHistoryRow]:
        """Get recently failed turns for a profile.

        Args:
            profile_id: Profile identifier.
            limit: Maximum number of records.
            indexing_path: Optional indexing path filter.

        Returns:
            List of failed PassageHistoryRow.
        """
        with self._conn.cursor() as cur:
            base_query = """
                SELECT id, profile_id, session_id, turn_index, indexing_path, action,
                       drop_reason, raw_text, preprocessed_text,
                       entities_extracted, passage_id, processed_at
                FROM passage_processing_history
                WHERE profile_id = %s AND action = 'failed'
            """
            params: list = [profile_id]

            if indexing_path is not None:
                base_query += " AND indexing_path = %s"
                params.append(indexing_path)

            base_query += " ORDER BY processed_at DESC LIMIT %s"
            params.append(limit)

            cur.execute(base_query, params)
            return [self._row_to_history(row) for row in cur.fetchall()]

    def get_stats(self, profile_id: str, indexing_path: str | None = None) -> dict:
        """Get processing statistics for a profile.

        Args:
            profile_id: Profile identifier.
            indexing_path: Optional indexing path filter.

        Returns:
            Dict with counts by action type.
        """
        with self._conn.cursor() as cur:
            base_query = """
                SELECT
                    COUNT(*) FILTER (WHERE action = 'indexed') as indexed,
                    COUNT(*) FILTER (WHERE action = 'dropped') as dropped,
                    COUNT(*) FILTER (WHERE action = 'skipped') as skipped,
                    COUNT(*) FILTER (WHERE action = 'failed') as failed
                FROM passage_processing_history
                WHERE profile_id = %s
            """
            params: list = [profile_id]

            if indexing_path is not None:
                base_query += " AND indexing_path = %s"
                params.append(indexing_path)

            cur.execute(base_query, params)
            row = cur.fetchone()
            return {
                "indexed": row[0],
                "dropped": row[1],
                "skipped": row[2],
                "failed": row[3],
            }

    def get_drop_reasons(
        self, profile_id: str, indexing_path: str | None = None
    ) -> list[tuple[str, int]]:
        """Get drop reason distribution for a profile.

        Args:
            profile_id: Profile identifier.
            indexing_path: Optional indexing path filter.

        Returns:
            List of (drop_reason, count) tuples ordered by count desc.
        """
        with self._conn.cursor() as cur:
            base_query = """
                SELECT drop_reason, COUNT(*) as count
                FROM passage_processing_history
                WHERE profile_id = %s AND action = 'dropped'
            """
            params: list = [profile_id]

            if indexing_path is not None:
                base_query += " AND indexing_path = %s"
                params.append(indexing_path)

            base_query += " GROUP BY drop_reason ORDER BY count DESC"

            cur.execute(base_query, params)
            return [(row[0], row[1]) for row in cur.fetchall()]

    def delete_by_session(
        self, profile_id: str, session_id: str, indexing_path: str | None = None
    ) -> int:
        """Delete all history for a session.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            indexing_path: Optional indexing path filter (None = delete all paths).

        Returns:
            Number of deleted records.
        """
        with self._conn.cursor() as cur:
            if indexing_path is not None:
                cur.execute(
                    """
                    DELETE FROM passage_processing_history
                    WHERE profile_id = %s AND session_id = %s AND indexing_path = %s
                    """,
                    (profile_id, session_id, indexing_path),
                )
            else:
                cur.execute(
                    """
                    DELETE FROM passage_processing_history
                    WHERE profile_id = %s AND session_id = %s
                    """,
                    (profile_id, session_id),
                )
            return cur.rowcount

    def _row_to_history(self, row: tuple) -> PassageHistoryRow:
        """Convert a database row tuple to PassageHistoryRow."""
        return PassageHistoryRow(
            id=row[0],
            profile_id=row[1],
            session_id=row[2],
            turn_index=row[3],
            indexing_path=row[4],
            action=row[5],
            drop_reason=row[6],
            raw_text=row[7],
            preprocessed_text=row[8],
            entities_extracted=row[9],
            passage_id=row[10],
            processed_at=row[11],
        )
