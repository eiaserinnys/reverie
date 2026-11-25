"""Repository for session_index_state table.

Tracks incremental indexing state for each session, enabling
atomic resume from any point of failure.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SessionIndexStateRow:
    """Data class for a session_index_state table row."""

    id: int
    profile_id: str
    session_id: str
    session_seq: int
    file_path: str
    indexing_path: str
    file_mtime: datetime
    file_size_bytes: int
    file_lines_count: int
    last_indexed_turn: int
    last_indexed_line: int
    last_passage_id: int | None
    total_passages_indexed: int
    total_passages_dropped: int
    status: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    last_indexed_at: datetime | None


class SessionIndexStateRepository:
    """CRUD operations for the session_index_state table."""

    def __init__(self, conn):
        """Initialize with a psycopg2 connection."""
        self._conn = conn

    def get_by_session(
        self, profile_id: str, session_id: str, indexing_path: str = ""
    ) -> SessionIndexStateRow | None:
        """Get indexing state for a specific session.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            indexing_path: Folder path being indexed (default: empty string).

        Returns:
            SessionIndexStateRow if found, None otherwise.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, profile_id, session_id, session_seq, file_path,
                       indexing_path, file_mtime, file_size_bytes, file_lines_count,
                       last_indexed_turn, last_indexed_line, last_passage_id,
                       total_passages_indexed, total_passages_dropped,
                       status, error_message,
                       created_at, updated_at, last_indexed_at
                FROM session_index_state
                WHERE profile_id = %s AND session_id = %s AND indexing_path = %s
                """,
                (profile_id, session_id, indexing_path),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_state(row)

    def upsert(
        self,
        profile_id: str,
        session_id: str,
        file_path: str,
        file_mtime: datetime,
        file_size_bytes: int,
        file_lines_count: int = 0,
        session_seq: int | None = None,
        indexing_path: str = "",
    ) -> SessionIndexStateRow:
        """Insert or update session indexing state.

        If session_seq is not provided, generates one using get_next_session_seq().

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            file_path: Path to the session JSONL file.
            file_mtime: Last modified time of the file.
            file_size_bytes: Size of the file in bytes.
            file_lines_count: Number of lines in the file.
            session_seq: Session sequence number (auto-generated if None).
            indexing_path: Folder path being indexed (default: empty string).

        Returns:
            Created or updated SessionIndexStateRow.
        """
        with self._conn.cursor() as cur:
            # Get next session_seq if not provided
            if session_seq is None:
                cur.execute("SELECT get_next_session_seq(%s)", (profile_id,))
                session_seq = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO session_index_state (
                    profile_id, session_id, session_seq, file_path,
                    indexing_path, file_mtime, file_size_bytes, file_lines_count
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (profile_id, session_id, indexing_path) DO UPDATE SET
                    file_path = EXCLUDED.file_path,
                    file_mtime = EXCLUDED.file_mtime,
                    file_size_bytes = EXCLUDED.file_size_bytes,
                    file_lines_count = EXCLUDED.file_lines_count
                RETURNING id, profile_id, session_id, session_seq, file_path,
                          indexing_path, file_mtime, file_size_bytes, file_lines_count,
                          last_indexed_turn, last_indexed_line, last_passage_id,
                          total_passages_indexed, total_passages_dropped,
                          status, error_message,
                          created_at, updated_at, last_indexed_at
                """,
                (
                    profile_id,
                    session_id,
                    session_seq,
                    file_path,
                    indexing_path,
                    file_mtime,
                    file_size_bytes,
                    file_lines_count,
                ),
            )
            return self._row_to_state(cur.fetchone())

    def update_progress(
        self,
        profile_id: str,
        session_id: str,
        last_indexed_turn: int,
        last_indexed_line: int,
        last_passage_id: int | None,
        passages_indexed_delta: int = 0,
        passages_dropped_delta: int = 0,
        indexing_path: str = "",
    ) -> None:
        """Update indexing progress for a session.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            last_indexed_turn: Last successfully indexed turn index.
            last_indexed_line: Last successfully indexed line number.
            last_passage_id: ID of last created passage (for adjacency).
            passages_indexed_delta: Number of passages indexed in this update.
            passages_dropped_delta: Number of passages dropped in this update.
            indexing_path: Folder path being indexed (default: empty string).
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE session_index_state SET
                    last_indexed_turn = %s,
                    last_indexed_line = %s,
                    last_passage_id = %s,
                    total_passages_indexed = total_passages_indexed + %s,
                    total_passages_dropped = total_passages_dropped + %s,
                    last_indexed_at = NOW(),
                    status = 'in_progress'
                WHERE profile_id = %s AND session_id = %s AND indexing_path = %s
                """,
                (
                    last_indexed_turn,
                    last_indexed_line,
                    last_passage_id,
                    passages_indexed_delta,
                    passages_dropped_delta,
                    profile_id,
                    session_id,
                    indexing_path,
                ),
            )

    def mark_completed(
        self, profile_id: str, session_id: str, indexing_path: str = ""
    ) -> None:
        """Mark a session as fully indexed.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            indexing_path: Folder path being indexed (default: empty string).
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE session_index_state SET
                    status = 'completed',
                    error_message = NULL,
                    last_indexed_at = NOW()
                WHERE profile_id = %s AND session_id = %s AND indexing_path = %s
                """,
                (profile_id, session_id, indexing_path),
            )

    def mark_failed(
        self, profile_id: str, session_id: str, error_message: str, indexing_path: str = ""
    ) -> None:
        """Mark a session indexing as failed.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            error_message: Error description.
            indexing_path: Folder path being indexed (default: empty string).
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE session_index_state SET
                    status = 'failed',
                    error_message = %s
                WHERE profile_id = %s AND session_id = %s AND indexing_path = %s
                """,
                (error_message, profile_id, session_id, indexing_path),
            )

    def mark_pending(
        self, profile_id: str, session_id: str, indexing_path: str = ""
    ) -> None:
        """Reset session status to pending (for re-queuing).

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            indexing_path: Folder path being indexed (default: empty string).
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE session_index_state SET
                    status = 'pending',
                    error_message = NULL
                WHERE profile_id = %s AND session_id = %s AND indexing_path = %s
                """,
                (profile_id, session_id, indexing_path),
            )

    def get_all_by_profile(self, profile_id: str) -> list[SessionIndexStateRow]:
        """Get all session states for a profile.

        Args:
            profile_id: Profile identifier.

        Returns:
            List of all SessionIndexStateRow for the profile.
        """
        return self.get_by_profile(profile_id, status=None)

    def get_by_profile(
        self,
        profile_id: str,
        status: str | None = None,
        indexing_path: str | None = None,
    ) -> list[SessionIndexStateRow]:
        """Get all session states for a profile.

        Args:
            profile_id: Profile identifier.
            status: Optional status filter.
            indexing_path: Optional indexing path filter.

        Returns:
            List of SessionIndexStateRow.
        """
        with self._conn.cursor() as cur:
            # Build query dynamically based on filters
            base_query = """
                SELECT id, profile_id, session_id, session_seq, file_path,
                       indexing_path, file_mtime, file_size_bytes, file_lines_count,
                       last_indexed_turn, last_indexed_line, last_passage_id,
                       total_passages_indexed, total_passages_dropped,
                       status, error_message,
                       created_at, updated_at, last_indexed_at
                FROM session_index_state
                WHERE profile_id = %s
            """
            params: list = [profile_id]

            if status is not None:
                base_query += " AND status = %s"
                params.append(status)

            if indexing_path is not None:
                base_query += " AND indexing_path = %s"
                params.append(indexing_path)

            base_query += " ORDER BY session_seq"

            cur.execute(base_query, params)
            return [self._row_to_state(row) for row in cur.fetchall()]

    def get_stats(self, profile_id: str, indexing_path: str | None = None) -> dict:
        """Get indexing statistics for a profile.

        Args:
            profile_id: Profile identifier.
            indexing_path: Optional indexing path filter.

        Returns:
            Dict with counts by status and totals.
        """
        with self._conn.cursor() as cur:
            base_query = """
                SELECT
                    COUNT(*) FILTER (WHERE status = 'pending') as pending,
                    COUNT(*) FILTER (WHERE status = 'in_progress') as in_progress,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed,
                    COALESCE(SUM(total_passages_indexed), 0) as total_indexed,
                    COALESCE(SUM(total_passages_dropped), 0) as total_dropped
                FROM session_index_state
                WHERE profile_id = %s
            """
            params: list = [profile_id]

            if indexing_path is not None:
                base_query += " AND indexing_path = %s"
                params.append(indexing_path)

            cur.execute(base_query, params)
            row = cur.fetchone()
            return {
                "pending": row[0],
                "in_progress": row[1],
                "completed": row[2],
                "failed": row[3],
                "total_indexed": row[4],
                "total_dropped": row[5],
            }

    def delete_by_session(
        self, profile_id: str, session_id: str, indexing_path: str | None = None
    ) -> bool:
        """Delete indexing state for a session.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            indexing_path: Optional indexing path filter (None = delete all paths).

        Returns:
            True if deleted, False if not found.
        """
        with self._conn.cursor() as cur:
            if indexing_path is not None:
                cur.execute(
                    """
                    DELETE FROM session_index_state
                    WHERE profile_id = %s AND session_id = %s AND indexing_path = %s
                    """,
                    (profile_id, session_id, indexing_path),
                )
            else:
                cur.execute(
                    """
                    DELETE FROM session_index_state
                    WHERE profile_id = %s AND session_id = %s
                    """,
                    (profile_id, session_id),
                )
            return cur.rowcount > 0

    def reset_for_full_reindex(
        self, profile_id: str, session_id: str, indexing_path: str = ""
    ) -> None:
        """Reset session state for full reindexing.

        Clears all progress counters while keeping session_seq.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            indexing_path: Folder path being indexed (default: empty string).
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE session_index_state SET
                    last_indexed_turn = -1,
                    last_indexed_line = 0,
                    last_passage_id = NULL,
                    total_passages_indexed = 0,
                    total_passages_dropped = 0,
                    status = 'pending',
                    error_message = NULL
                WHERE profile_id = %s AND session_id = %s AND indexing_path = %s
                """,
                (profile_id, session_id, indexing_path),
            )

    def _row_to_state(self, row: tuple) -> SessionIndexStateRow:
        """Convert a database row tuple to SessionIndexStateRow."""
        return SessionIndexStateRow(
            id=row[0],
            profile_id=row[1],
            session_id=row[2],
            session_seq=row[3],
            file_path=row[4],
            indexing_path=row[5],
            file_mtime=row[6],
            file_size_bytes=row[7],
            file_lines_count=row[8],
            last_indexed_turn=row[9],
            last_indexed_line=row[10],
            last_passage_id=row[11],
            total_passages_indexed=row[12],
            total_passages_dropped=row[13],
            status=row[14],
            error_message=row[15],
            created_at=row[16],
            updated_at=row[17],
            last_indexed_at=row[18],
        )
