"""PostgreSQL implementation of ReverieDB.

Wraps existing Repository classes to implement the unified ReverieDB interface.
This allows gradual migration from direct repository usage to the abstract interface.
"""

import logging
from datetime import datetime

from reverie.db.reverie_db import (
    DualSearchResult,
    EntityData,
    PassageData,
    PPRNeighborEdge,
    ReverieDB,
    SessionIndexState,
)
from reverie.db.repositories.adjacency_repo import AdjacencyRepository
from reverie.db.repositories.entity_passage_repo import EntityPassageRepository
from reverie.db.repositories.entity_repo import EntityRepository, EntityRow
from reverie.db.repositories.passage_history_repo import PassageProcessingHistoryRepository
from reverie.db.repositories.passage_repo import PassageRepository, PassageRow
from reverie.db.repositories.ppr_neighbor_repo import (
    PPRNeighborRepository,
    PPRNeighborEdge as RepoPPRNeighborEdge,
)
from reverie.db.repositories.session_index_state_repo import SessionIndexStateRepository

logger = logging.getLogger(__name__)


def _entity_row_to_data(row: EntityRow) -> EntityData:
    """Convert EntityRow to EntityData."""
    return EntityData(
        id=row.id,
        name=row.name,
        embedding=row.embedding,
        created_at=row.created_at,
        out_degree=row.out_degree,
    )


def _passage_row_to_data(row: PassageRow) -> PassageData:
    """Convert PassageRow to PassageData."""
    return PassageData(
        id=row.id,
        session_id=row.session_id,
        turn_index=row.turn_index,
        text=row.text,
        metadata=row.metadata,
        embedding=row.embedding,
        created_at=row.created_at,
        user_text=row.user_text,
        user_embedding=row.user_embedding,
        assistant_text=row.assistant_text,
        assistant_embedding=row.assistant_embedding,
        profile_id=row.profile_id,
        session_seq=row.session_seq,
        out_degree=row.out_degree,
    )


class PostgresDB(ReverieDB):
    """PostgreSQL implementation of ReverieDB.

    Wraps existing repositories and provides unified interface.
    Uses psycopg2 connection for PostgreSQL-specific features.
    """

    def __init__(self, conn):
        """Initialize with a psycopg2 connection.

        Args:
            conn: psycopg2 connection object.
        """
        self._conn = conn

        # Initialize repositories
        self._entity_repo = EntityRepository(conn)
        self._passage_repo = PassageRepository(conn)
        self._entity_passage_repo = EntityPassageRepository(conn)
        self._adjacency_repo = AdjacencyRepository(conn)
        self._ppr_neighbor_repo = PPRNeighborRepository(conn)
        self._session_state_repo = SessionIndexStateRepository(conn)
        self._history_repo = PassageProcessingHistoryRepository(conn)

    # ========== Connection Management ==========

    def commit(self) -> None:
        """Commit the current transaction."""
        self._conn.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self._conn.rollback()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @property
    def closed(self) -> bool:
        """Check if the connection is closed."""
        return self._conn.closed

    @property
    def raw_connection(self):
        """Get the underlying psycopg2 connection.

        This is for backward compatibility during migration.
        New code should use ReverieDB methods instead.
        """
        return self._conn

    # ========== Entity Operations ==========

    def upsert_entity(
        self,
        name: str,
        embedding: list[float] | None = None,
    ) -> int:
        """Insert or update an entity. Returns entity ID."""
        return self._entity_repo.upsert(name, embedding)

    def upsert_entity_with_similarity(
        self,
        name: str,
        embedding: list[float] | None = None,
        similarity_threshold: float | None = None,
    ) -> int:
        """Insert or update entity with embedding similarity matching."""
        return self._entity_repo.upsert_with_similarity(
            name, embedding, similarity_threshold
        )

    def get_entity_by_id(self, entity_id: int) -> EntityData | None:
        """Get entity by ID."""
        row = self._entity_repo.get_by_id(entity_id)
        return _entity_row_to_data(row) if row else None

    def get_entity_by_name(self, name: str) -> EntityData | None:
        """Get entity by name (exact match)."""
        row = self._entity_repo.get_by_name(name)
        return _entity_row_to_data(row) if row else None

    def resolve_entity_alias(self, name: str) -> EntityData | None:
        """Resolve an alias to its canonical entity."""
        row = self._entity_repo.resolve_alias(name)
        return _entity_row_to_data(row) if row else None

    def resolve_entity_aliases_batch(
        self, names: list[str]
    ) -> dict[str, EntityData]:
        """Resolve multiple aliases to canonical entities."""
        result = self._entity_repo.resolve_aliases_batch(names)
        return {k: _entity_row_to_data(v) for k, v in result.items()}

    def get_entities_by_ids(self, entity_ids: list[int]) -> list[EntityData]:
        """Get multiple entities by IDs."""
        rows = self._entity_repo.get_by_ids(entity_ids)
        return [_entity_row_to_data(row) for row in rows]

    def search_entities_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[EntityData, float]]:
        """Search entities by embedding similarity."""
        results = self._entity_repo.search_by_embedding(query_embedding, top_k)
        return [(entity_row_to_data(e), d) for e, d in results]

    def find_entity_seeds(
        self,
        query_embedding: list[float],
        tau: float,
        initial_k: int = 100,
        max_k: int = 100000,
    ) -> list[tuple[int, float]]:
        """Find entity seeds with similarity >= tau."""
        return self._entity_repo.find_seeds_above_tau(
            query_embedding, tau, initial_k, max_k
        )

    def update_entity_out_degree(self, entity_id: int, delta: int) -> None:
        """Update entity out_degree by delta."""
        self._entity_repo.update_out_degree(entity_id, delta)

    def count_entities(self) -> int:
        """Count total entities."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM entities")
            return cur.fetchone()[0]

    def list_entities(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_field: str = "id",
        sort_order: str = "desc",
    ) -> tuple[list[dict], int]:
        """List entities with pagination."""
        # Field mapping for safety
        field_map = {"id": "e.id", "name": "e.name", "created_at": "e.created_at"}
        order_map = {"asc": "ASC", "desc": "DESC"}

        sql_sort = field_map.get(sort_field, "e.id")
        sql_order = order_map.get(sort_order, "DESC")

        with self._conn.cursor() as cur:
            # Total count
            cur.execute("SELECT COUNT(*) FROM entities")
            total = cur.fetchone()[0]

            # Paginated results
            cur.execute(
                f"""
                SELECT e.id, e.name, e.created_at,
                       (SELECT COUNT(*) FROM entity_passages ep WHERE ep.entity_id = e.id) as passage_count
                FROM entities e
                ORDER BY {sql_sort} {sql_order}
                OFFSET %s LIMIT %s
                """,
                (offset, limit),
            )
            entities = []
            for row in cur.fetchall():
                entities.append({
                    "id": row[0],
                    "name": row[1],
                    "created_at": row[2].isoformat() if row[2] else None,
                    "passage_count": row[3],
                })

        return entities, total

    # ========== Passage Operations ==========

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
        return self._passage_repo.insert(
            session_id=session_id,
            turn_index=turn_index,
            text=text,
            metadata=metadata,
            embedding=embedding,
            user_text=user_text,
            user_embedding=user_embedding,
            assistant_text=assistant_text,
            assistant_embedding=assistant_embedding,
            profile_id=profile_id,
            session_seq=session_seq,
            out_degree=out_degree,
        )

    def get_passage_by_id(self, passage_id: int) -> PassageData | None:
        """Get passage by ID."""
        row = self._passage_repo.get_by_id(passage_id)
        return _passage_row_to_data(row) if row else None

    def get_passages_by_ids(self, passage_ids: list[int]) -> list[PassageData]:
        """Get multiple passages by IDs."""
        rows = self._passage_repo.get_by_ids(passage_ids)
        return [_passage_row_to_data(row) for row in rows]

    def get_passages_by_session(
        self,
        session_id: str,
        profile_id: str | None = None,
    ) -> list[PassageData]:
        """Get passages for a session."""
        rows = self._passage_repo.get_by_session(session_id, profile_id)
        return [_passage_row_to_data(row) for row in rows]

    def search_passages_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        profile_id: str | None = None,
    ) -> list[tuple[PassageData, float]]:
        """Search passages by embedding."""
        results = self._passage_repo.search_by_embedding(
            query_embedding, top_k, profile_id
        )
        return [(_passage_row_to_data(p), d) for p, d in results]

    def search_passages_dual_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[DualSearchResult]:
        """Search passages using dual embeddings."""
        results = self._passage_repo.search_by_dual_embedding(query_embedding, top_k)
        return [
            DualSearchResult(
                passage_id=r.passage_id,
                user_similarity=r.user_similarity,
                assistant_similarity=r.assistant_similarity,
                combined_score=r.combined_score,
                user_text=r.user_text,
                assistant_text=r.assistant_text,
            )
            for r in results
        ]

    def delete_passages_by_session(self, session_id: str) -> int:
        """Delete passages for a session. Returns deleted count."""
        return self._passage_repo.delete_by_session(session_id)

    def update_passage_out_degree(self, passage_id: int, delta: int) -> None:
        """Update passage out_degree by delta."""
        self._passage_repo.update_out_degree(passage_id, delta)

    def count_passages(self) -> int:
        """Count total passages."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM passages")
            return cur.fetchone()[0]

    def list_passages(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_field: str = "id",
        sort_order: str = "desc",
    ) -> tuple[list[dict], int]:
        """List passages with pagination."""
        # Field mapping for safety
        field_map = {
            "id": "p.id",
            "session_seq": "p.session_seq",
            "turn_index": "p.turn_index",
        }
        order_map = {"asc": "ASC", "desc": "DESC"}

        sql_sort = field_map.get(sort_field, "p.id")
        sql_order = order_map.get(sort_order, "DESC")

        with self._conn.cursor() as cur:
            # Total count
            cur.execute("SELECT COUNT(*) FROM passages")
            total = cur.fetchone()[0]

            # Paginated results
            cur.execute(
                f"""
                SELECT p.id, p.session_id, p.session_seq, p.turn_index,
                       LEFT(p.text, 200) as text_preview,
                       (SELECT COUNT(*) FROM entity_passages ep WHERE ep.passage_id = p.id) as entity_count
                FROM passages p
                ORDER BY {sql_sort} {sql_order}
                OFFSET %s LIMIT %s
                """,
                (offset, limit),
            )
            passages = []
            for row in cur.fetchall():
                passages.append({
                    "id": row[0],
                    "session_id": row[1],
                    "session_seq": row[2],
                    "turn_index": row[3],
                    "text_preview": row[4],
                    "entity_count": row[5],
                })

        return passages, total

    # ========== Entity-Passage Link Operations ==========

    def link_entity_passage(self, entity_id: int, passage_id: int) -> bool:
        """Create entity-passage link.

        Returns True if link was created, False if already exists.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entity_passages (entity_id, passage_id)
                VALUES (%s, %s)
                ON CONFLICT (entity_id, passage_id) DO NOTHING
                RETURNING entity_id
                """,
                (entity_id, passage_id),
            )
            result = cur.fetchone()
            if result is not None:
                # Update out_degree since we created a new link
                cur.execute(
                    "UPDATE entities SET out_degree = out_degree + 1 WHERE id = %s",
                    (entity_id,),
                )
                return True
            return False

    def get_entity_passages(
        self,
        entity_id: int,
        limit: int = 50,
    ) -> list[dict]:
        """Get passages containing an entity."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.id, p.session_id, p.turn_index, p.session_seq, p.text, p.created_at
                FROM passages p
                JOIN entity_passages ep ON p.id = ep.passage_id
                WHERE ep.entity_id = %s
                ORDER BY p.id DESC
                LIMIT %s
                """,
                (entity_id, limit),
            )
            passages = []
            for row in cur.fetchall():
                passages.append({
                    "id": row[0],
                    "session_id": row[1],
                    "turn_index": row[2],
                    "session_seq": row[3],
                    "text": row[4][:2000] if row[4] else None,
                    "created_at": row[5].isoformat() if row[5] else None,
                })
            return passages

    def get_passage_entities(self, passage_id: int) -> list[dict]:
        """Get entities in a passage."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.id, e.name
                FROM entities e
                JOIN entity_passages ep ON e.id = ep.entity_id
                WHERE ep.passage_id = %s
                ORDER BY e.name
                """,
                (passage_id,),
            )
            return [{"entity_id": row[0], "name": row[1]} for row in cur.fetchall()]

    def get_passage_entity_ids(
        self,
        passage_ids: list[int],
    ) -> dict[int, set[int]]:
        """Get entity IDs for multiple passages."""
        if not passage_ids:
            return {}

        result: dict[int, set[int]] = {}
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT ep.passage_id, ep.entity_id
                FROM entity_passages ep
                WHERE ep.passage_id = ANY(%s)
                """,
                (list(passage_ids),),
            )
            for row in cur.fetchall():
                pid, eid = row
                if pid not in result:
                    result[pid] = set()
                result[pid].add(eid)

        return result

    def count_entity_passages(self) -> int:
        """Count total entity-passage links."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM entity_passages")
            return cur.fetchone()[0]

    # ========== Adjacency Operations ==========

    def create_adjacency(self, passage_id_a: int, passage_id_b: int) -> None:
        """Create passage adjacency (bidirectional)."""
        self._adjacency_repo.create(passage_id_a, passage_id_b)

    def get_adjacent_passages(self, passage_id: int) -> list[dict]:
        """Get passages adjacent to a passage."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.id, p.session_id, p.turn_index, p.session_seq, p.text
                FROM passages p
                WHERE p.id IN (
                    SELECT passage_id_b FROM passage_adjacency WHERE passage_id_a = %s
                    UNION
                    SELECT passage_id_a FROM passage_adjacency WHERE passage_id_b = %s
                )
                ORDER BY p.id
                """,
                (passage_id, passage_id),
            )
            passages = []
            for row in cur.fetchall():
                passages.append({
                    "passage_id": row[0],
                    "session_id": row[1],
                    "turn_index": row[2],
                    "session_seq": row[3],
                    "text": row[4][:200] if row[4] else None,
                })
            return passages

    def count_adjacencies(self) -> int:
        """Count total adjacencies."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM passage_adjacency")
            return cur.fetchone()[0]

    # ========== PPR Graph Traversal ==========

    def get_ppr_neighbors(
        self,
        active_entity_ids: list[int],
        active_passage_ids: list[int],
    ) -> list[PPRNeighborEdge]:
        """Get all PPR neighbor edges for active nodes."""
        edges = self._ppr_neighbor_repo.get_all_neighbors(
            active_entity_ids, active_passage_ids
        )
        return [
            PPRNeighborEdge(
                edge_type=e.edge_type,
                source_id=e.source_id,
                target_id=e.target_id,
                out_degree=e.out_degree,
            )
            for e in edges
        ]

    def run_ppr_server_side(
        self,
        seed_ids: list[int],
        seed_scores: list[float],
        damping: float,
        max_iter: int,
        min_score: float,
        tol: float,
    ) -> list[tuple[str, int, float]]:
        """Run PPR on server side (PostgreSQL function)."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT node_type, node_id, node_score FROM run_lazy_ppr(%s, %s, %s, %s, %s, %s)",
                (seed_ids, seed_scores, damping, max_iter, min_score, tol),
            )
            return list(cur.fetchall())

    # ========== Session State Operations ==========

    def get_session_state(
        self,
        profile_id: str,
        session_id: str,
    ) -> SessionIndexState | None:
        """Get session indexing state."""
        row = self._session_state_repo.get_by_session(profile_id, session_id)
        if row is None:
            return None
        return SessionIndexState(
            profile_id=row.profile_id,
            session_id=row.session_id,
            session_seq=row.session_seq,
            file_path=row.file_path,
            file_mtime=row.file_mtime,
            file_size_bytes=row.file_size_bytes,
            file_lines_count=row.file_lines_count,
            last_indexed_turn=row.last_indexed_turn,
            last_indexed_line=row.last_indexed_line,
            last_passage_id=row.last_passage_id,
            passages_indexed=row.total_passages_indexed,
            passages_dropped=row.total_passages_dropped,
            status=row.status,
            error_message=row.error_message,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

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
        row = self._session_state_repo.upsert(
            profile_id=profile_id,
            session_id=session_id,
            file_path=file_path,
            file_mtime=file_mtime,
            file_size_bytes=file_size_bytes,
            file_lines_count=file_lines_count,
            session_seq=session_seq,
        )
        return SessionIndexState(
            profile_id=row.profile_id,
            session_id=row.session_id,
            session_seq=row.session_seq,
            file_path=row.file_path,
            file_mtime=row.file_mtime,
            file_size_bytes=row.file_size_bytes,
            file_lines_count=row.file_lines_count,
            last_indexed_turn=row.last_indexed_turn,
            last_indexed_line=row.last_indexed_line,
            last_passage_id=row.last_passage_id,
            passages_indexed=row.total_passages_indexed,
            passages_dropped=row.total_passages_dropped,
            status=row.status,
            error_message=row.error_message,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

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
        self._session_state_repo.update_progress(
            profile_id=profile_id,
            session_id=session_id,
            last_indexed_turn=last_indexed_turn,
            last_indexed_line=last_indexed_line,
            last_passage_id=last_passage_id,
            passages_indexed_delta=passages_indexed_delta,
            passages_dropped_delta=passages_dropped_delta,
        )

    def mark_session_completed(self, profile_id: str, session_id: str) -> None:
        """Mark session as completed."""
        self._session_state_repo.mark_completed(profile_id, session_id)

    def mark_session_failed(
        self,
        profile_id: str,
        session_id: str,
        error_message: str,
    ) -> None:
        """Mark session as failed."""
        self._session_state_repo.mark_failed(profile_id, session_id, error_message)

    def reset_session_for_reindex(
        self,
        profile_id: str,
        session_id: str,
    ) -> None:
        """Reset session state for full reindex."""
        self._session_state_repo.reset_for_full_reindex(profile_id, session_id)

    def get_session_stats(self, profile_id: str) -> dict:
        """Get session indexing statistics."""
        return self._session_state_repo.get_stats(profile_id)

    # ========== Processing History ==========

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
        self._history_repo.record(
            profile_id=profile_id,
            session_id=session_id,
            turn_index=turn_index,
            action=action,
            drop_reason=drop_reason,
            raw_text=raw_text,
            preprocessed_text=preprocessed_text,
            entities_extracted=entities_extracted,
            passage_id=passage_id,
        )

    def delete_processing_history(
        self,
        profile_id: str,
        session_id: str,
    ) -> int:
        """Delete processing history for a session."""
        return self._history_repo.delete_by_session(profile_id, session_id)

    # ========== Statistics ==========

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM entities")
            entities = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM passages")
            passages = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM entity_passages")
            entity_passages = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM passage_adjacency")
            passage_adjacency = cur.fetchone()[0]

            cur.execute("SELECT COUNT(DISTINCT session_id) FROM passages")
            sessions = cur.fetchone()[0]

            cur.execute("SELECT MAX(created_at) FROM passages")
            row = cur.fetchone()
            last_indexed = str(row[0]) if row[0] else "never"

        return {
            "entities": entities,
            "passages": passages,
            "entity_passages": entity_passages,
            "passage_adjacency": passage_adjacency,
            "sessions": sessions,
            "last_indexed": last_indexed,
        }

    def count_distinct_sessions(self) -> int:
        """Count distinct sessions."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT session_id) FROM passages")
            return cur.fetchone()[0]

    def get_last_indexed_time(self) -> datetime | None:
        """Get last indexing timestamp."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT MAX(created_at) FROM passages")
            row = cur.fetchone()
            return row[0] if row else None


# Fix typo in function name
entity_row_to_data = _entity_row_to_data
