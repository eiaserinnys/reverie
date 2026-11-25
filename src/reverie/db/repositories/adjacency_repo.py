"""Repository for passage adjacency edges.

Manages the passage_adjacency table for MemoryLink v2.
Stores Passage↔Passage temporal edges within sessions.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PassageAdjacency:
    """Passage adjacency record (Passage↔Passage edge)."""

    passage_id_a: int
    passage_id_b: int


class AdjacencyRepository:
    """Repository for passage adjacency operations.

    Passage adjacency represents temporal connections between passages
    in the same session within a certain turn window.
    """

    def __init__(self, conn):
        self._conn = conn

    def create(
        self, passage_id_a: int, passage_id_b: int, update_out_degree: bool = True
    ) -> PassageAdjacency:
        """Create a passage adjacency edge.

        Auto-orders passage IDs so that passage_id_a < passage_id_b
        to maintain the CHECK constraint and avoid duplicates.

        Args:
            passage_id_a: First passage ID.
            passage_id_b: Second passage ID.
            update_out_degree: If True, increment out_degree of both passages.

        Returns:
            PassageAdjacency record.
        """
        # Ensure a < b for uniqueness constraint
        pid_a, pid_b = min(passage_id_a, passage_id_b), max(passage_id_a, passage_id_b)

        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO passage_adjacency (passage_id_a, passage_id_b)
                VALUES (%s, %s)
                ON CONFLICT (passage_id_a, passage_id_b) DO NOTHING
                RETURNING passage_id_a
                """,
                (pid_a, pid_b),
            )
            result = cur.fetchone()

            # Only update out_degree if the insert actually happened (not a conflict)
            if result is not None and update_out_degree:
                # Both passages get +1 out_degree (bidirectional edge)
                cur.execute(
                    "UPDATE passages SET out_degree = out_degree + 1 WHERE id = %s",
                    (pid_a,),
                )
                cur.execute(
                    "UPDATE passages SET out_degree = out_degree + 1 WHERE id = %s",
                    (pid_b,),
                )

            self._conn.commit()
            return PassageAdjacency(passage_id_a=pid_a, passage_id_b=pid_b)

    def get_adjacent_passages(self, passage_id: int) -> list[int]:
        """Get all passage IDs adjacent to a given passage.

        Args:
            passage_id: The passage ID to find neighbors for.

        Returns:
            List of adjacent passage IDs.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT passage_id_b FROM passage_adjacency WHERE passage_id_a = %s
                UNION
                SELECT passage_id_a FROM passage_adjacency WHERE passage_id_b = %s
                ORDER BY 1
                """,
                (passage_id, passage_id),
            )
            return [row[0] for row in cur.fetchall()]

    def get_all(self) -> list[PassageAdjacency]:
        """Get all adjacency records.

        Returns:
            List of all PassageAdjacency records.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT passage_id_a, passage_id_b
                FROM passage_adjacency
                ORDER BY passage_id_a, passage_id_b
                """
            )
            return [
                PassageAdjacency(passage_id_a=row[0], passage_id_b=row[1])
                for row in cur.fetchall()
            ]

    def delete_by_passage(
        self, passage_id: int, update_out_degree: bool = True
    ) -> int:
        """Delete all adjacencies involving a specific passage and update out_degree.

        Args:
            passage_id: The passage ID to delete adjacencies for.
            update_out_degree: If True, decrement out_degree of adjacent passages.

        Returns:
            Number of deleted records.
        """
        with self._conn.cursor() as cur:
            if update_out_degree:
                # Decrement out_degree of all adjacent passages (excluding the one being deleted)
                cur.execute(
                    """
                    UPDATE passages p
                    SET out_degree = out_degree - 1
                    FROM passage_adjacency a
                    WHERE (a.passage_id_a = %s AND p.id = a.passage_id_b)
                       OR (a.passage_id_b = %s AND p.id = a.passage_id_a)
                    """,
                    (passage_id, passage_id),
                )

            cur.execute(
                """
                DELETE FROM passage_adjacency
                WHERE passage_id_a = %s OR passage_id_b = %s
                """,
                (passage_id, passage_id),
            )
            count = cur.rowcount
            self._conn.commit()
            return count

    def delete_all(self) -> int:
        """Delete all adjacency records.

        Returns:
            Number of deleted records.
        """
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM passage_adjacency")
            count = cur.rowcount
            self._conn.commit()
            return count

    def bulk_create(self, pairs: list[tuple[int, int]]) -> int:
        """Create multiple adjacencies in a single operation.

        Args:
            pairs: List of (passage_id_a, passage_id_b) tuples.

        Returns:
            Number of created records.
        """
        if not pairs:
            return 0

        # Normalize pairs: ensure a < b and deduplicate
        normalized = set()
        for a, b in pairs:
            pid_a, pid_b = min(a, b), max(a, b)
            normalized.add((pid_a, pid_b))

        with self._conn.cursor() as cur:
            # Use executemany for bulk insert
            cur.executemany(
                """
                INSERT INTO passage_adjacency (passage_id_a, passage_id_b)
                VALUES (%s, %s)
                ON CONFLICT (passage_id_a, passage_id_b) DO NOTHING
                """,
                list(normalized),
            )
            self._conn.commit()
            return len(normalized)

    def create_for_session(self, session_id: str, window: int = 1) -> int:
        """Create adjacencies for all passages in a session within the given window.

        Connects each passage to passages within `window` positions in the sequence.
        Note: Uses passage sequence order (not turn_index) because preprocessor
        may drop some turns, creating gaps in turn_index.

        Args:
            session_id: The session ID to create adjacencies for.
            window: Maximum position difference for adjacency (default 1 = adjacent only).

        Returns:
            Number of created adjacency records.
        """
        # Get all passages for the session ordered by turn_index
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, turn_index
                FROM passages
                WHERE session_id = %s
                ORDER BY turn_index
                """,
                (session_id,),
            )
            passages = cur.fetchall()  # [(id, turn_index), ...]

        if len(passages) < 2:
            return 0

        # Generate all pairs within the window (by sequence position, not turn_index)
        pairs = []
        for i in range(len(passages)):
            pid_i = passages[i][0]
            for j in range(i + 1, min(i + 1 + window, len(passages))):
                pid_j = passages[j][0]
                pairs.append((pid_i, pid_j))

        return self.bulk_create(pairs)
