"""Repository for entity-passage links.

Manages the entity_passages table for MemoryLink algorithm.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EntityPassage:
    """Entity-passage link record."""

    entity_id: int
    passage_id: int


class EntityPassageRepository:
    """Repository for entity-passage link operations."""

    def __init__(self, conn):
        self._conn = conn

    def upsert(self, entity_id: int, passage_id: int) -> EntityPassage:
        """Create or update entity-passage link.

        Args:
            entity_id: Entity ID.
            passage_id: Passage ID.

        Returns:
            EntityPassage record.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entity_passages (entity_id, passage_id)
                VALUES (%s, %s)
                ON CONFLICT (entity_id, passage_id) DO NOTHING
                RETURNING entity_id, passage_id
                """,
                (entity_id, passage_id),
            )
            # If conflict, just return the existing record
            self._conn.commit()
            return EntityPassage(entity_id=entity_id, passage_id=passage_id)

    def delete(
        self, entity_id: int, passage_id: int, update_out_degree: bool = True
    ) -> bool:
        """Delete entity-passage link and update out_degree values.

        Args:
            entity_id: Entity ID.
            passage_id: Passage ID.
            update_out_degree: If True, decrement out_degree of entity and passage.

        Returns:
            True if deleted, False if not found.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM entity_passages
                WHERE entity_id = %s AND passage_id = %s
                RETURNING entity_id
                """,
                (entity_id, passage_id),
            )
            result = cur.fetchone()
            deleted = result is not None

            if deleted and update_out_degree:
                # Decrement entity out_degree
                cur.execute(
                    "UPDATE entities SET out_degree = out_degree - 1 WHERE id = %s",
                    (entity_id,),
                )
                # Decrement passage out_degree
                cur.execute(
                    "UPDATE passages SET out_degree = out_degree - 1 WHERE id = %s",
                    (passage_id,),
                )

            self._conn.commit()
            return deleted

    def get_passages_by_entity(self, entity_id: int) -> list[int]:
        """Get all passage IDs linked to an entity.

        Args:
            entity_id: Entity ID.

        Returns:
            List of passage IDs.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT passage_id
                FROM entity_passages
                WHERE entity_id = %s
                ORDER BY passage_id
                """,
                (entity_id,),
            )
            return [row[0] for row in cur.fetchall()]

    def get_entities_by_passage(self, passage_id: int) -> list[int]:
        """Get all entity IDs linked to a passage.

        Args:
            passage_id: Passage ID.

        Returns:
            List of entity IDs.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT entity_id
                FROM entity_passages
                WHERE passage_id = %s
                ORDER BY entity_id
                """,
                (passage_id,),
            )
            return [row[0] for row in cur.fetchall()]

    def get_all(self) -> list[EntityPassage]:
        """Get all entity-passage links.

        Returns:
            List of all EntityPassage records.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT entity_id, passage_id
                FROM entity_passages
                ORDER BY entity_id, passage_id
                """
            )
            return [
                EntityPassage(entity_id=row[0], passage_id=row[1])
                for row in cur.fetchall()
            ]

    def delete_by_passage(
        self, passage_id: int, update_out_degree: bool = True
    ) -> int:
        """Delete all entity links for a passage and update out_degree values.

        Args:
            passage_id: Passage ID.
            update_out_degree: If True, decrement out_degree of affected entities.

        Returns:
            Number of deleted records.
        """
        with self._conn.cursor() as cur:
            if update_out_degree:
                # Decrement out_degree of all affected entities first
                cur.execute(
                    """
                    UPDATE entities e
                    SET out_degree = out_degree - 1
                    FROM entity_passages ep
                    WHERE ep.entity_id = e.id AND ep.passage_id = %s
                    """,
                    (passage_id,),
                )

            cur.execute(
                """
                DELETE FROM entity_passages
                WHERE passage_id = %s
                """,
                (passage_id,),
            )
            count = cur.rowcount
            self._conn.commit()
            return count

    def delete_all(self) -> int:
        """Delete all entity-passage links.

        Returns:
            Number of deleted records.
        """
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM entity_passages")
            count = cur.rowcount
            self._conn.commit()
            return count
