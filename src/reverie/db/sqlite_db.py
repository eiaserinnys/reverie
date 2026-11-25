"""SQLite implementation of ReverieDB.

Uses sqlite-vec extension for vector similarity search.
Implements the same interface as PostgresDB for seamless backend switching.
"""

import json
import logging
import random
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlite_vec

from reverie.db.reverie_db import (
    DualSearchResult,
    EntityData,
    PassageData,
    PPRNeighborEdge,
    ReverieDB,
    SessionIndexState,
)
from reverie.db.vector_utils import (
    cosine_distance_to_similarity,
    deserialize_vector,
    serialize_vector,
)

logger = logging.getLogger(__name__)


def _adapt_datetime(dt: datetime) -> str:
    """Adapt datetime to SQLite TEXT format."""
    return dt.isoformat()


def _convert_datetime(val: bytes) -> datetime:
    """Convert SQLite TEXT to datetime."""
    return datetime.fromisoformat(val.decode())


# Register adapters for datetime
sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("timestamp", _convert_datetime)


class SqliteDB(ReverieDB):
    """SQLite implementation of ReverieDB.

    Uses sqlite-vec for vector operations. Provides the same interface
    as PostgresDB to allow seamless backend switching.
    """

    def __init__(self, db_path: str | Path):
        """Initialize SQLite connection.

        Args:
            db_path: Path to SQLite database file.
                     Use ":memory:" for in-memory database.
        """
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(
            self._db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        self._conn.row_factory = sqlite3.Row

        # Enable foreign keys
        self._conn.execute("PRAGMA foreign_keys = ON")

        # Load sqlite-vec extension
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._closed = False

    # ========== Connection Management ==========

    def commit(self) -> None:
        """Commit the current transaction."""
        self._conn.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self._conn.rollback()

    def close(self) -> None:
        """Close the database connection."""
        if not self._closed:
            self._conn.close()
            self._closed = True

    @property
    def closed(self) -> bool:
        """Check if the connection is closed."""
        return self._closed

    @property
    def raw_connection(self):
        """Get the underlying sqlite3 connection.

        For backward compatibility during migration.
        """
        return self._conn

    # ========== Entity Operations ==========

    def upsert_entity(
        self,
        name: str,
        embedding: list[float] | None = None,
    ) -> int:
        """Insert or update an entity. Returns entity ID.

        Retries on database lock errors to handle concurrent access.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                cursor = self._conn.execute(
                    """
                    INSERT INTO entities (name, embedding)
                    VALUES (?, ?)
                    ON CONFLICT (name) DO UPDATE SET
                        embedding = COALESCE(excluded.embedding, entities.embedding)
                    RETURNING id
                    """,
                    (name, serialize_vector(embedding)),
                )
                row = cursor.fetchone()
                return row[0]
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    delay = (0.1 * (2**attempt)) + random.uniform(0, 0.05)
                    logger.warning(
                        "Entity upsert locked (attempt %d/%d), retrying in %.2fs",
                        attempt + 1, max_retries, delay,
                    )
                    time.sleep(delay)
                else:
                    raise

    def upsert_entity_with_similarity(
        self,
        name: str,
        embedding: list[float] | None = None,
        similarity_threshold: float | None = None,
    ) -> int:
        """Insert or update entity with embedding similarity matching.

        Note: In SQLite, this is simplified - we don't do cross-entity
        similarity matching like PostgreSQL. The similarity_threshold
        parameter is ignored for now.
        """
        # For SQLite, just do basic upsert (similarity matching across
        # all entities would be too slow without HNSW index)
        return self.upsert_entity(name, embedding)

    def get_entity_by_id(self, entity_id: int) -> EntityData | None:
        """Get entity by ID."""
        cursor = self._conn.execute(
            """
            SELECT id, name, embedding, created_at, out_degree
            FROM entities WHERE id = ?
            """,
            (entity_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return EntityData(
            id=row["id"],
            name=row["name"],
            embedding=deserialize_vector(row["embedding"]),
            created_at=_parse_datetime(row["created_at"]),
            out_degree=row["out_degree"] or 0,
        )

    def get_entity_by_name(self, name: str) -> EntityData | None:
        """Get entity by name (exact match)."""
        cursor = self._conn.execute(
            """
            SELECT id, name, embedding, created_at, out_degree
            FROM entities WHERE name = ?
            """,
            (name,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return EntityData(
            id=row["id"],
            name=row["name"],
            embedding=deserialize_vector(row["embedding"]),
            created_at=_parse_datetime(row["created_at"]),
            out_degree=row["out_degree"] or 0,
        )

    def resolve_entity_alias(self, name: str) -> EntityData | None:
        """Resolve an alias to its canonical entity (case-insensitive)."""
        # First try alias lookup (case-insensitive, matching PostgreSQL behavior)
        cursor = self._conn.execute(
            """
            SELECT e.id, e.name, e.embedding, e.created_at, e.out_degree
            FROM entity_aliases ea
            JOIN entities e ON e.id = ea.canonical_entity_id
            WHERE LOWER(ea.alias_name) = LOWER(?)
            """,
            (name,),
        )
        row = cursor.fetchone()
        if row is not None:
            return EntityData(
                id=row["id"],
                name=row["name"],
                embedding=deserialize_vector(row["embedding"]),
                created_at=_parse_datetime(row["created_at"]),
                out_degree=row["out_degree"] or 0,
            )

        # Fall back to direct entity lookup (case-insensitive)
        cursor = self._conn.execute(
            """
            SELECT id, name, embedding, created_at, out_degree
            FROM entities WHERE LOWER(name) = LOWER(?)
            """,
            (name,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return EntityData(
            id=row["id"],
            name=row["name"],
            embedding=deserialize_vector(row["embedding"]),
            created_at=_parse_datetime(row["created_at"]),
            out_degree=row["out_degree"] or 0,
        )

    def resolve_entity_aliases_batch(
        self, names: list[str]
    ) -> dict[str, EntityData]:
        """Resolve multiple aliases to canonical entities (case-insensitive).

        Returns dict mapping lowercased input name to EntityData,
        matching PostgreSQL behavior.
        """
        if not names:
            return {}

        result: dict[str, EntityData] = {}
        lower_names = [n.lower() for n in names]
        placeholders = ",".join("?" * len(lower_names))

        # Check aliases first (case-insensitive)
        cursor = self._conn.execute(
            f"""
            SELECT LOWER(ea.alias_name), e.id, e.name, e.embedding, e.created_at, e.out_degree
            FROM entity_aliases ea
            JOIN entities e ON e.id = ea.canonical_entity_id
            WHERE LOWER(ea.alias_name) IN ({placeholders})
            """,
            lower_names,
        )
        for row in cursor.fetchall():
            alias_lower = row[0]
            result[alias_lower] = EntityData(
                id=row[1],
                name=row[2],
                embedding=deserialize_vector(row[3]),
                created_at=_parse_datetime(row[4]),
                out_degree=row[5] or 0,
            )

        # For names not found in aliases, try direct entity lookup
        remaining = [n for n in lower_names if n not in result]
        if remaining:
            placeholders = ",".join("?" * len(remaining))
            cursor = self._conn.execute(
                f"""
                SELECT LOWER(name), id, name, embedding, created_at, out_degree
                FROM entities WHERE LOWER(name) IN ({placeholders})
                """,
                remaining,
            )
            for row in cursor.fetchall():
                name_lower = row[0]
                result[name_lower] = EntityData(
                    id=row[1],
                    name=row[2],
                    embedding=deserialize_vector(row[3]),
                    created_at=_parse_datetime(row[4]),
                    out_degree=row[5] or 0,
                )

        return result

    def get_entities_by_ids(self, entity_ids: list[int]) -> list[EntityData]:
        """Get multiple entities by IDs."""
        if not entity_ids:
            return []

        placeholders = ",".join("?" * len(entity_ids))
        cursor = self._conn.execute(
            f"""
            SELECT id, name, embedding, created_at, out_degree
            FROM entities WHERE id IN ({placeholders})
            """,
            entity_ids,
        )
        return [
            EntityData(
                id=row["id"],
                name=row["name"],
                embedding=deserialize_vector(row["embedding"]),
                created_at=_parse_datetime(row["created_at"]),
                out_degree=row["out_degree"] or 0,
            )
            for row in cursor.fetchall()
        ]

    def search_entities_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[EntityData, float]]:
        """Search entities by embedding similarity.

        Returns list of (entity, distance) tuples.
        """
        query_vec = serialize_vector(query_embedding)

        # Use sqlite-vec for cosine distance
        cursor = self._conn.execute(
            """
            SELECT e.id, e.name, e.embedding, e.created_at, e.out_degree,
                   vec_distance_cosine(e.embedding, ?) as distance
            FROM entities e
            WHERE e.embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT ?
            """,
            (query_vec, top_k),
        )

        results = []
        for row in cursor.fetchall():
            entity = EntityData(
                id=row["id"],
                name=row["name"],
                embedding=deserialize_vector(row["embedding"]),
                created_at=_parse_datetime(row["created_at"]),
                out_degree=row["out_degree"] or 0,
            )
            results.append((entity, row["distance"]))
        return results

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
        query_vec = serialize_vector(query_embedding)
        max_distance = 1.0 - tau  # Convert similarity threshold to distance

        cursor = self._conn.execute(
            """
            SELECT id, vec_distance_cosine(embedding, ?) as distance
            FROM entities
            WHERE embedding IS NOT NULL
              AND vec_distance_cosine(embedding, ?) <= ?
            ORDER BY distance ASC
            LIMIT ?
            """,
            (query_vec, query_vec, max_distance, max_k),
        )

        return [
            (row["id"], cosine_distance_to_similarity(row["distance"]))
            for row in cursor.fetchall()
        ]

    def update_entity_out_degree(self, entity_id: int, delta: int) -> None:
        """Update entity out_degree by delta."""
        self._conn.execute(
            "UPDATE entities SET out_degree = out_degree + ? WHERE id = ?",
            (delta, entity_id),
        )

    def count_entities(self) -> int:
        """Count total entities."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM entities")
        return cursor.fetchone()[0]

    def list_entities(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_field: str = "id",
        sort_order: str = "desc",
    ) -> tuple[list[dict], int]:
        """List entities with pagination."""
        field_map = {"id": "e.id", "name": "e.name", "created_at": "e.created_at"}
        order_map = {"asc": "ASC", "desc": "DESC"}

        sql_sort = field_map.get(sort_field, "e.id")
        sql_order = order_map.get(sort_order, "DESC")

        # Total count
        cursor = self._conn.execute("SELECT COUNT(*) FROM entities")
        total = cursor.fetchone()[0]

        # Paginated results
        cursor = self._conn.execute(
            f"""
            SELECT e.id, e.name, e.created_at,
                   (SELECT COUNT(*) FROM entity_passages ep WHERE ep.entity_id = e.id) as passage_count
            FROM entities e
            ORDER BY {sql_sort} {sql_order}
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

        entities = []
        for row in cursor.fetchall():
            entities.append({
                "id": row["id"],
                "name": row["name"],
                "created_at": row["created_at"] if row["created_at"] else None,
                "passage_count": row["passage_count"],
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
        cursor = self._conn.execute(
            """
            INSERT INTO passages (
                session_id, turn_index, text, metadata, embedding,
                user_text, user_embedding, assistant_text, assistant_embedding,
                profile_id, session_seq, out_degree
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                session_id,
                turn_index,
                text,
                json.dumps(metadata or {}),
                serialize_vector(embedding),
                user_text,
                serialize_vector(user_embedding),
                assistant_text,
                serialize_vector(assistant_embedding),
                profile_id,
                session_seq,
                out_degree,
            ),
        )
        return cursor.fetchone()[0]

    def get_passage_by_id(self, passage_id: int) -> PassageData | None:
        """Get passage by ID."""
        cursor = self._conn.execute(
            """
            SELECT id, session_id, turn_index, text, metadata, embedding, created_at,
                   user_text, user_embedding, assistant_text, assistant_embedding,
                   profile_id, session_seq, out_degree
            FROM passages WHERE id = ?
            """,
            (passage_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_passage_data(row)

    def get_passages_by_ids(self, passage_ids: list[int]) -> list[PassageData]:
        """Get multiple passages by IDs."""
        if not passage_ids:
            return []

        placeholders = ",".join("?" * len(passage_ids))
        cursor = self._conn.execute(
            f"""
            SELECT id, session_id, turn_index, text, metadata, embedding, created_at,
                   user_text, user_embedding, assistant_text, assistant_embedding,
                   profile_id, session_seq, out_degree
            FROM passages WHERE id IN ({placeholders})
            """,
            passage_ids,
        )
        return [_row_to_passage_data(row) for row in cursor.fetchall()]

    def get_passages_by_session(
        self,
        session_id: str,
        profile_id: str | None = None,
    ) -> list[PassageData]:
        """Get passages for a session."""
        if profile_id:
            cursor = self._conn.execute(
                """
                SELECT id, session_id, turn_index, text, metadata, embedding, created_at,
                       user_text, user_embedding, assistant_text, assistant_embedding,
                       profile_id, session_seq, out_degree
                FROM passages WHERE session_id = ? AND profile_id = ?
                ORDER BY turn_index
                """,
                (session_id, profile_id),
            )
        else:
            cursor = self._conn.execute(
                """
                SELECT id, session_id, turn_index, text, metadata, embedding, created_at,
                       user_text, user_embedding, assistant_text, assistant_embedding,
                       profile_id, session_seq, out_degree
                FROM passages WHERE session_id = ?
                ORDER BY turn_index
                """,
                (session_id,),
            )
        return [_row_to_passage_data(row) for row in cursor.fetchall()]

    def search_passages_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        profile_id: str | None = None,
    ) -> list[tuple[PassageData, float]]:
        """Search passages by embedding."""
        query_vec = serialize_vector(query_embedding)

        if profile_id:
            cursor = self._conn.execute(
                """
                SELECT p.id, p.session_id, p.turn_index, p.text, p.metadata, p.embedding,
                       p.created_at, p.user_text, p.user_embedding, p.assistant_text,
                       p.assistant_embedding, p.profile_id, p.session_seq, p.out_degree,
                       vec_distance_cosine(p.embedding, ?) as distance
                FROM passages p
                WHERE p.embedding IS NOT NULL AND p.profile_id = ?
                ORDER BY distance ASC
                LIMIT ?
                """,
                (query_vec, profile_id, top_k),
            )
        else:
            cursor = self._conn.execute(
                """
                SELECT p.id, p.session_id, p.turn_index, p.text, p.metadata, p.embedding,
                       p.created_at, p.user_text, p.user_embedding, p.assistant_text,
                       p.assistant_embedding, p.profile_id, p.session_seq, p.out_degree,
                       vec_distance_cosine(p.embedding, ?) as distance
                FROM passages p
                WHERE p.embedding IS NOT NULL
                ORDER BY distance ASC
                LIMIT ?
                """,
                (query_vec, top_k),
            )

        results = []
        for row in cursor.fetchall():
            passage = _row_to_passage_data(row)
            results.append((passage, row["distance"]))
        return results

    def search_passages_dual_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[DualSearchResult]:
        """Search passages using dual embeddings."""
        query_vec = serialize_vector(query_embedding)
        candidate_multiplier = 3

        # Get user embedding candidates
        cursor = self._conn.execute(
            """
            SELECT id, 1.0 - vec_distance_cosine(user_embedding, ?) / 2.0 as user_sim
            FROM passages
            WHERE user_embedding IS NOT NULL
            ORDER BY vec_distance_cosine(user_embedding, ?) ASC
            LIMIT ?
            """,
            (query_vec, query_vec, top_k * candidate_multiplier),
        )
        user_candidates = {row["id"]: row["user_sim"] for row in cursor.fetchall()}

        # Get assistant embedding candidates
        cursor = self._conn.execute(
            """
            SELECT id, 1.0 - vec_distance_cosine(assistant_embedding, ?) / 2.0 as assistant_sim
            FROM passages
            WHERE assistant_embedding IS NOT NULL
            ORDER BY vec_distance_cosine(assistant_embedding, ?) ASC
            LIMIT ?
            """,
            (query_vec, query_vec, top_k * candidate_multiplier),
        )
        assistant_candidates = {row["id"]: row["assistant_sim"] for row in cursor.fetchall()}

        # Combine candidates
        all_ids = set(user_candidates.keys()) | set(assistant_candidates.keys())
        combined = []
        for pid in all_ids:
            u_sim = user_candidates.get(pid, 0.0)
            a_sim = assistant_candidates.get(pid, 0.0)
            # Harmonic mean
            if u_sim > 0 and a_sim > 0:
                score = 2.0 * u_sim * a_sim / (u_sim + a_sim + 0.0001)
            else:
                score = 0.0
            if score > 0:
                combined.append((pid, u_sim, a_sim, score))

        # Sort by combined score
        combined.sort(key=lambda x: x[3], reverse=True)
        combined = combined[:top_k]

        if not combined:
            return []

        # Fetch passage texts
        pids = [c[0] for c in combined]
        placeholders = ",".join("?" * len(pids))
        cursor = self._conn.execute(
            f"SELECT id, user_text, assistant_text FROM passages WHERE id IN ({placeholders})",
            pids,
        )
        texts = {row["id"]: (row["user_text"], row["assistant_text"]) for row in cursor.fetchall()}

        results = []
        for pid, u_sim, a_sim, score in combined:
            user_text, assistant_text = texts.get(pid, (None, None))
            results.append(DualSearchResult(
                passage_id=pid,
                user_similarity=u_sim,
                assistant_similarity=a_sim,
                combined_score=score,
                user_text=user_text,
                assistant_text=assistant_text,
            ))

        return results

    def delete_passages_by_session(self, session_id: str) -> int:
        """Delete passages for a session. Returns deleted count."""
        cursor = self._conn.execute(
            "DELETE FROM passages WHERE session_id = ?",
            (session_id,),
        )
        return cursor.rowcount

    def update_passage_out_degree(self, passage_id: int, delta: int) -> None:
        """Update passage out_degree by delta."""
        self._conn.execute(
            "UPDATE passages SET out_degree = out_degree + ? WHERE id = ?",
            (delta, passage_id),
        )

    def count_passages(self) -> int:
        """Count total passages."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM passages")
        return cursor.fetchone()[0]

    def list_passages(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_field: str = "id",
        sort_order: str = "desc",
    ) -> tuple[list[dict], int]:
        """List passages with pagination."""
        field_map = {
            "id": "p.id",
            "session_seq": "p.session_seq",
            "turn_index": "p.turn_index",
        }
        order_map = {"asc": "ASC", "desc": "DESC"}

        sql_sort = field_map.get(sort_field, "p.id")
        sql_order = order_map.get(sort_order, "DESC")

        # Total count
        cursor = self._conn.execute("SELECT COUNT(*) FROM passages")
        total = cursor.fetchone()[0]

        # Paginated results
        cursor = self._conn.execute(
            f"""
            SELECT p.id, p.session_id, p.session_seq, p.turn_index,
                   SUBSTR(p.text, 1, 200) as text_preview,
                   (SELECT COUNT(*) FROM entity_passages ep WHERE ep.passage_id = p.id) as entity_count
            FROM passages p
            ORDER BY {sql_sort} {sql_order}
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

        passages = []
        for row in cursor.fetchall():
            passages.append({
                "id": row["id"],
                "session_id": row["session_id"],
                "session_seq": row["session_seq"],
                "turn_index": row["turn_index"],
                "text_preview": row["text_preview"],
                "entity_count": row["entity_count"],
            })

        return passages, total

    # ========== Entity-Passage Link Operations ==========

    def link_entity_passage(self, entity_id: int, passage_id: int) -> bool:
        """Create entity-passage link."""
        # First check if link already exists
        cursor = self._conn.execute(
            "SELECT 1 FROM entity_passages WHERE entity_id = ? AND passage_id = ?",
            (entity_id, passage_id),
        )
        if cursor.fetchone() is not None:
            return False  # Link already exists

        try:
            self._conn.execute(
                "INSERT INTO entity_passages (entity_id, passage_id) VALUES (?, ?)",
                (entity_id, passage_id),
            )
            # Update out_degree since we created a new link
            self._conn.execute(
                "UPDATE entities SET out_degree = out_degree + 1 WHERE id = ?",
                (entity_id,),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_entity_passages(
        self,
        entity_id: int,
        limit: int = 50,
    ) -> list[dict]:
        """Get passages containing an entity."""
        cursor = self._conn.execute(
            """
            SELECT p.id, p.session_id, p.turn_index, p.session_seq, p.text, p.created_at
            FROM passages p
            JOIN entity_passages ep ON p.id = ep.passage_id
            WHERE ep.entity_id = ?
            ORDER BY p.id DESC
            LIMIT ?
            """,
            (entity_id, limit),
        )
        passages = []
        for row in cursor.fetchall():
            passages.append({
                "id": row["id"],
                "session_id": row["session_id"],
                "turn_index": row["turn_index"],
                "session_seq": row["session_seq"],
                "text": row["text"][:2000] if row["text"] else None,
                "created_at": row["created_at"] if row["created_at"] else None,
            })
        return passages

    def get_passage_entities(self, passage_id: int) -> list[dict]:
        """Get entities in a passage."""
        cursor = self._conn.execute(
            """
            SELECT e.id, e.name
            FROM entities e
            JOIN entity_passages ep ON e.id = ep.entity_id
            WHERE ep.passage_id = ?
            ORDER BY e.name
            """,
            (passage_id,),
        )
        return [{"entity_id": row["id"], "name": row["name"]} for row in cursor.fetchall()]

    def get_passage_entity_ids(
        self,
        passage_ids: list[int],
    ) -> dict[int, set[int]]:
        """Get entity IDs for multiple passages."""
        if not passage_ids:
            return {}

        result: dict[int, set[int]] = {}
        placeholders = ",".join("?" * len(passage_ids))
        cursor = self._conn.execute(
            f"""
            SELECT ep.passage_id, ep.entity_id
            FROM entity_passages ep
            WHERE ep.passage_id IN ({placeholders})
            """,
            passage_ids,
        )
        for row in cursor.fetchall():
            pid, eid = row["passage_id"], row["entity_id"]
            if pid not in result:
                result[pid] = set()
            result[pid].add(eid)

        return result

    def count_entity_passages(self) -> int:
        """Count total entity-passage links."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM entity_passages")
        return cursor.fetchone()[0]

    # ========== Adjacency Operations ==========

    def create_adjacency(self, passage_id_a: int, passage_id_b: int) -> None:
        """Create passage adjacency (bidirectional)."""
        # Normalize: smaller ID first
        if passage_id_a > passage_id_b:
            passage_id_a, passage_id_b = passage_id_b, passage_id_a

        self._conn.execute(
            """
            INSERT INTO passage_adjacency (passage_id_a, passage_id_b)
            VALUES (?, ?)
            ON CONFLICT (passage_id_a, passage_id_b) DO NOTHING
            """,
            (passage_id_a, passage_id_b),
        )

    def get_adjacent_passages(self, passage_id: int) -> list[dict]:
        """Get passages adjacent to a passage."""
        cursor = self._conn.execute(
            """
            SELECT p.id, p.session_id, p.turn_index, p.session_seq, p.text
            FROM passages p
            WHERE p.id IN (
                SELECT passage_id_b FROM passage_adjacency WHERE passage_id_a = ?
                UNION
                SELECT passage_id_a FROM passage_adjacency WHERE passage_id_b = ?
            )
            ORDER BY p.id
            """,
            (passage_id, passage_id),
        )
        passages = []
        for row in cursor.fetchall():
            passages.append({
                "passage_id": row["id"],
                "session_id": row["session_id"],
                "turn_index": row["turn_index"],
                "session_seq": row["session_seq"],
                "text": row["text"][:200] if row["text"] else None,
            })
        return passages

    def count_adjacencies(self) -> int:
        """Count total adjacencies."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM passage_adjacency")
        return cursor.fetchone()[0]

    # ========== PPR Graph Traversal ==========

    def get_ppr_neighbors(
        self,
        active_entity_ids: list[int],
        active_passage_ids: list[int],
    ) -> list[PPRNeighborEdge]:
        """Get all PPR neighbor edges for active nodes."""
        edges: list[PPRNeighborEdge] = []

        # Entity to Passage edges (e2p)
        if active_entity_ids:
            placeholders = ",".join("?" * len(active_entity_ids))
            cursor = self._conn.execute(
                f"""
                SELECT ep.entity_id, ep.passage_id, e.out_degree
                FROM entity_passages ep
                JOIN entities e ON e.id = ep.entity_id
                WHERE ep.entity_id IN ({placeholders})
                """,
                active_entity_ids,
            )
            for row in cursor.fetchall():
                edges.append(PPRNeighborEdge(
                    edge_type="e2p",
                    source_id=row["entity_id"],
                    target_id=row["passage_id"],
                    out_degree=row["out_degree"] or 1,
                ))

        # Passage to Entity edges (p2e)
        if active_passage_ids:
            placeholders = ",".join("?" * len(active_passage_ids))
            cursor = self._conn.execute(
                f"""
                SELECT ep.passage_id, ep.entity_id, p.out_degree
                FROM entity_passages ep
                JOIN passages p ON p.id = ep.passage_id
                WHERE ep.passage_id IN ({placeholders})
                """,
                active_passage_ids,
            )
            for row in cursor.fetchall():
                edges.append(PPRNeighborEdge(
                    edge_type="p2e",
                    source_id=row["passage_id"],
                    target_id=row["entity_id"],
                    out_degree=row["out_degree"] or 1,
                ))

            # Passage to Passage edges (p2p_a and p2p_b)
            cursor = self._conn.execute(
                f"""
                SELECT pa.passage_id_a, pa.passage_id_b, p.out_degree
                FROM passage_adjacency pa
                JOIN passages p ON p.id = pa.passage_id_a
                WHERE pa.passage_id_a IN ({placeholders})
                """,
                active_passage_ids,
            )
            for row in cursor.fetchall():
                edges.append(PPRNeighborEdge(
                    edge_type="p2p_a",
                    source_id=row["passage_id_a"],
                    target_id=row["passage_id_b"],
                    out_degree=row["out_degree"] or 1,
                ))

            cursor = self._conn.execute(
                f"""
                SELECT pa.passage_id_b, pa.passage_id_a, p.out_degree
                FROM passage_adjacency pa
                JOIN passages p ON p.id = pa.passage_id_b
                WHERE pa.passage_id_b IN ({placeholders})
                """,
                active_passage_ids,
            )
            for row in cursor.fetchall():
                edges.append(PPRNeighborEdge(
                    edge_type="p2p_b",
                    source_id=row["passage_id_b"],
                    target_id=row["passage_id_a"],
                    out_degree=row["out_degree"] or 1,
                ))

        return edges

    def run_ppr_server_side(
        self,
        seed_ids: list[int],
        seed_scores: list[float],
        damping: float,
        max_iter: int,
        min_score: float,
        tol: float,
    ) -> list[tuple[str, int, float]]:
        """Run PPR on server side.

        SQLite doesn't support server-side PPR like PostgreSQL.
        This returns empty list - client should use Python-side PPR.
        """
        # SQLite doesn't have stored procedures for PPR
        # Return empty to signal that client-side PPR should be used
        logger.warning("Server-side PPR not supported in SQLite. Use client-side PPR.")
        return []

    # ========== Session State Operations ==========

    def get_session_state(
        self,
        profile_id: str,
        session_id: str,
    ) -> SessionIndexState | None:
        """Get session indexing state."""
        cursor = self._conn.execute(
            """
            SELECT profile_id, session_id, session_seq, file_path, file_mtime,
                   file_size_bytes, file_lines_count, last_indexed_turn, last_indexed_line,
                   last_passage_id, total_passages_indexed, total_passages_dropped,
                   status, error_message, created_at, updated_at
            FROM session_index_state
            WHERE profile_id = ? AND session_id = ?
            """,
            (profile_id, session_id),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return SessionIndexState(
            profile_id=row["profile_id"],
            session_id=row["session_id"],
            session_seq=row["session_seq"],
            file_path=row["file_path"],
            file_mtime=_parse_datetime(row["file_mtime"]),
            file_size_bytes=row["file_size_bytes"],
            file_lines_count=row["file_lines_count"],
            last_indexed_turn=row["last_indexed_turn"],
            last_indexed_line=row["last_indexed_line"],
            last_passage_id=row["last_passage_id"],
            passages_indexed=row["total_passages_indexed"],
            passages_dropped=row["total_passages_dropped"],
            status=row["status"],
            error_message=row["error_message"],
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
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
        # Get next session_seq if not provided
        if session_seq is None:
            cursor = self._conn.execute(
                """
                SELECT COALESCE(MAX(session_seq), 0) + 1 as next_seq
                FROM session_index_state
                WHERE profile_id = ?
                """,
                (profile_id,),
            )
            session_seq = cursor.fetchone()["next_seq"]

        self._conn.execute(
            """
            INSERT INTO session_index_state (
                profile_id, session_id, session_seq, file_path, file_mtime,
                file_size_bytes, file_lines_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (profile_id, session_id) DO UPDATE SET
                file_path = excluded.file_path,
                file_mtime = excluded.file_mtime,
                file_size_bytes = excluded.file_size_bytes,
                file_lines_count = excluded.file_lines_count,
                updated_at = CURRENT_TIMESTAMP
            """,
            (profile_id, session_id, session_seq, file_path, file_mtime.isoformat(),
             file_size_bytes, file_lines_count),
        )

        return self.get_session_state(profile_id, session_id)  # type: ignore

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
        self._conn.execute(
            """
            UPDATE session_index_state SET
                last_indexed_turn = ?,
                last_indexed_line = ?,
                last_passage_id = ?,
                total_passages_indexed = total_passages_indexed + ?,
                total_passages_dropped = total_passages_dropped + ?,
                status = 'in_progress',
                updated_at = CURRENT_TIMESTAMP
            WHERE profile_id = ? AND session_id = ?
            """,
            (last_indexed_turn, last_indexed_line, last_passage_id,
             passages_indexed_delta, passages_dropped_delta,
             profile_id, session_id),
        )

    def mark_session_completed(self, profile_id: str, session_id: str) -> None:
        """Mark session as completed."""
        self._conn.execute(
            """
            UPDATE session_index_state SET
                status = 'completed',
                last_indexed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE profile_id = ? AND session_id = ?
            """,
            (profile_id, session_id),
        )

    def mark_session_failed(
        self,
        profile_id: str,
        session_id: str,
        error_message: str,
    ) -> None:
        """Mark session as failed."""
        self._conn.execute(
            """
            UPDATE session_index_state SET
                status = 'failed',
                error_message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE profile_id = ? AND session_id = ?
            """,
            (error_message, profile_id, session_id),
        )

    def reset_session_for_reindex(
        self,
        profile_id: str,
        session_id: str,
    ) -> None:
        """Reset session state for full reindex."""
        self._conn.execute(
            """
            UPDATE session_index_state SET
                last_indexed_turn = -1,
                last_indexed_line = 0,
                last_passage_id = NULL,
                total_passages_indexed = 0,
                total_passages_dropped = 0,
                status = 'pending',
                error_message = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE profile_id = ? AND session_id = ?
            """,
            (profile_id, session_id),
        )

    def get_session_stats(self, profile_id: str) -> dict:
        """Get session indexing statistics."""
        cursor = self._conn.execute(
            """
            SELECT
                COUNT(*) as total_sessions,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
                SUM(total_passages_indexed) as total_passages_indexed,
                SUM(total_passages_dropped) as total_passages_dropped
            FROM session_index_state
            WHERE profile_id = ?
            """,
            (profile_id,),
        )
        row = cursor.fetchone()
        return {
            "total_sessions": row["total_sessions"] or 0,
            "completed": row["completed"] or 0,
            "failed": row["failed"] or 0,
            "pending": row["pending"] or 0,
            "in_progress": row["in_progress"] or 0,
            "total_passages_indexed": row["total_passages_indexed"] or 0,
            "total_passages_dropped": row["total_passages_dropped"] or 0,
        }

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
        self._conn.execute(
            """
            INSERT INTO passage_processing_history (
                profile_id, session_id, turn_index, action, drop_reason,
                raw_text, preprocessed_text, entities_extracted, passage_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (profile_id, session_id, turn_index) DO UPDATE SET
                action = excluded.action,
                drop_reason = excluded.drop_reason,
                raw_text = excluded.raw_text,
                preprocessed_text = excluded.preprocessed_text,
                entities_extracted = excluded.entities_extracted,
                passage_id = excluded.passage_id,
                processed_at = CURRENT_TIMESTAMP
            """,
            (profile_id, session_id, turn_index, action, drop_reason,
             raw_text, preprocessed_text,
             json.dumps(entities_extracted) if entities_extracted else None,
             passage_id),
        )

    def delete_processing_history(
        self,
        profile_id: str,
        session_id: str,
    ) -> int:
        """Delete processing history for a session."""
        cursor = self._conn.execute(
            """
            DELETE FROM passage_processing_history
            WHERE profile_id = ? AND session_id = ?
            """,
            (profile_id, session_id),
        )
        return cursor.rowcount

    # ========== Statistics ==========

    def get_stats(self) -> dict:
        """Get database statistics."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM entities")
        entities = cursor.fetchone()[0]

        cursor = self._conn.execute("SELECT COUNT(*) FROM passages")
        passages = cursor.fetchone()[0]

        cursor = self._conn.execute("SELECT COUNT(*) FROM entity_passages")
        entity_passages = cursor.fetchone()[0]

        cursor = self._conn.execute("SELECT COUNT(*) FROM passage_adjacency")
        passage_adjacency = cursor.fetchone()[0]

        cursor = self._conn.execute("SELECT COUNT(DISTINCT session_id) FROM passages")
        sessions = cursor.fetchone()[0]

        cursor = self._conn.execute("SELECT MAX(created_at) FROM passages")
        row = cursor.fetchone()
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
        cursor = self._conn.execute("SELECT COUNT(DISTINCT session_id) FROM passages")
        return cursor.fetchone()[0]

    def get_last_indexed_time(self) -> datetime | None:
        """Get last indexing timestamp."""
        cursor = self._conn.execute("SELECT MAX(created_at) FROM passages")
        row = cursor.fetchone()
        return _parse_datetime(row[0]) if row[0] else None


def _parse_datetime(val: Any) -> datetime:
    """Parse datetime from SQLite value."""
    if val is None:
        return datetime.now()
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        # Handle ISO format strings
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    return datetime.now()


def _row_to_passage_data(row: sqlite3.Row) -> PassageData:
    """Convert SQLite row to PassageData."""
    return PassageData(
        id=row["id"],
        session_id=row["session_id"],
        turn_index=row["turn_index"],
        text=row["text"],
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        embedding=deserialize_vector(row["embedding"]),
        created_at=_parse_datetime(row["created_at"]),
        user_text=row["user_text"],
        user_embedding=deserialize_vector(row["user_embedding"]),
        assistant_text=row["assistant_text"],
        assistant_embedding=deserialize_vector(row["assistant_embedding"]),
        profile_id=row["profile_id"],
        session_seq=row["session_seq"],
        out_degree=row["out_degree"] or 0,
    )
