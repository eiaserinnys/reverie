"""Repository for entities table.

CRUD operations for named entities with UPSERT and pgvector embedding search.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EntityRow:
    """Data class for an entities table row."""

    id: int
    name: str
    embedding: list[float] | None
    created_at: datetime
    out_degree: int = 0  # Number of passages this entity appears in (for Lazy PPR)


class EntityRepository:
    """CRUD operations for the entities table."""

    def __init__(self, conn):
        """Initialize with a psycopg2 connection."""
        self._conn = conn

    def upsert(
        self,
        name: str,
        embedding: list[float] | None = None,
    ) -> int:
        """Insert or update an entity by name. Returns entity ID.

        Uses ON CONFLICT to prevent duplicates.
        If entity exists and new embedding is provided, updates the embedding.

        Args:
            name: Entity name.
            embedding: Optional embedding vector (1536 dims).

        Returns:
            Entity ID (existing or newly created).
        """
        with self._conn.cursor() as cur:
            if embedding is not None:
                # Step 1: Try exact name match
                cur.execute(
                    """
                    INSERT INTO entities (name, embedding)
                    VALUES (%s, %s::vector)
                    ON CONFLICT (name) DO UPDATE SET embedding = EXCLUDED.embedding
                    RETURNING id
                    """,
                    (name, str(embedding)),
                )
                return cur.fetchone()[0]
            else:
                cur.execute(
                    """
                    INSERT INTO entities (name)
                    VALUES (%s)
                    ON CONFLICT (name) DO NOTHING
                    RETURNING id
                    """,
                    (name,),
                )
                result = cur.fetchone()
                if result is not None:
                    return result[0]
                # Entity already exists, fetch its ID
                cur.execute(
                    "SELECT id FROM entities WHERE name = %s",
                    (name,),
                )
                return cur.fetchone()[0]

    def upsert_with_similarity(
        self,
        name: str,
        embedding: list[float] | None = None,
        similarity_threshold: float | None = None,
    ) -> int:
        """Insert or update with embedding similarity matching.

        Extended version of upsert that checks embedding similarity
        when exact name match fails:
        1. Try exact name match via ON CONFLICT.
        2. If name is new AND embedding is provided → similarity search.
        3. If similar entity found (above threshold) → return its ID.
        4. Otherwise → create new entity.

        Args:
            name: Entity name.
            embedding: Optional embedding vector (1536 dims).
            similarity_threshold: Cosine similarity threshold for dedup.
                If None, uses config.entity_similarity_threshold.

        Returns:
            Entity ID (existing or newly created).
        """
        if similarity_threshold is None:
            from reverie.config import get_settings
            similarity_threshold = get_settings().entity_similarity_threshold

        with self._conn.cursor() as cur:
            # Step 1: Check if name already exists
            cur.execute(
                "SELECT id FROM entities WHERE name = %s",
                (name,),
            )
            existing = cur.fetchone()
            if existing is not None:
                # Update embedding if provided
                if embedding is not None:
                    cur.execute(
                        "UPDATE entities SET embedding = %s::vector WHERE id = %s",
                        (str(embedding), existing[0]),
                    )
                return existing[0]

            # Step 2: Name is new. If embedding provided, try similarity match
            if embedding is not None:
                similar_id = self._find_similar_entity(
                    embedding, similarity_threshold
                )
                if similar_id is not None:
                    return similar_id

            # Step 3: Create new entity (use ON CONFLICT for race condition safety)
            if embedding is not None:
                cur.execute(
                    """
                    INSERT INTO entities (name, embedding)
                    VALUES (%s, %s::vector)
                    ON CONFLICT (name) DO UPDATE SET embedding = EXCLUDED.embedding
                    RETURNING id
                    """,
                    (name, str(embedding)),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO entities (name)
                    VALUES (%s)
                    ON CONFLICT (name) DO NOTHING
                    RETURNING id
                    """,
                    (name,),
                )
                result = cur.fetchone()
                if result is not None:
                    return result[0]
                # Race: another transaction inserted it first
                cur.execute(
                    "SELECT id FROM entities WHERE name = %s",
                    (name,),
                )
            return cur.fetchone()[0]

    def _find_similar_entity(
        self, embedding: list[float], similarity_threshold: float
    ) -> int | None:
        """Find an existing entity by embedding cosine similarity.

        Args:
            embedding: Query embedding vector.
            similarity_threshold: Minimum similarity (1 - distance) to match.

        Returns:
            Entity ID if a match is found, None otherwise.
        """
        # Cosine distance threshold: distance = 1 - similarity
        distance_threshold = 1.0 - similarity_threshold

        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, embedding <=> %s::vector AS distance
                FROM entities
                WHERE embedding IS NOT NULL
                ORDER BY distance
                LIMIT 1
                """,
                (str(embedding),),
            )
            row = cur.fetchone()
            if row is not None and row[2] <= distance_threshold:
                logger.info(
                    "Similarity match: '%s' (distance=%.4f, threshold=%.4f)",
                    row[1], row[2], distance_threshold,
                )
                return row[0]
        return None

    def get_by_name(self, name: str) -> EntityRow | None:
        """Get an entity by its name."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, embedding, created_at, out_degree
                FROM entities WHERE name = %s
                """,
                (name,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_entity(row)

    def resolve_alias(self, name: str) -> EntityRow | None:
        """Resolve an alias to its canonical entity.

        Checks if name matches an alias and returns the canonical entity.
        If no alias match, tries exact name match on entities table.

        Args:
            name: Entity name or alias to resolve.

        Returns:
            Canonical EntityRow if found, None otherwise.
        """
        with self._conn.cursor() as cur:
            # First check alias table (case-insensitive)
            cur.execute(
                """
                SELECT e.id, e.name, e.embedding, e.created_at, e.out_degree
                FROM entity_aliases ea
                JOIN entities e ON ea.canonical_entity_id = e.id
                WHERE LOWER(ea.alias_name) = LOWER(%s)
                """,
                (name,),
            )
            row = cur.fetchone()
            if row is not None:
                return self._row_to_entity(row)

            # Fall back to direct entity lookup (case-insensitive)
            cur.execute(
                """
                SELECT id, name, embedding, created_at, out_degree
                FROM entities WHERE LOWER(name) = LOWER(%s)
                """,
                (name,),
            )
            row = cur.fetchone()
            if row is not None:
                return self._row_to_entity(row)

            return None

    def resolve_aliases_batch(self, names: list[str]) -> dict[str, EntityRow]:
        """Resolve multiple aliases to canonical entities in batch.

        Args:
            names: List of entity names or aliases to resolve.

        Returns:
            Dict mapping input name (lowercased) to canonical EntityRow.
        """
        if not names:
            return {}

        result = {}
        lower_names = [n.lower() for n in names]

        with self._conn.cursor() as cur:
            # Check aliases
            cur.execute(
                """
                SELECT LOWER(ea.alias_name), e.id, e.name, e.embedding, e.created_at, e.out_degree
                FROM entity_aliases ea
                JOIN entities e ON ea.canonical_entity_id = e.id
                WHERE LOWER(ea.alias_name) = ANY(%s)
                """,
                (lower_names,),
            )
            for row in cur.fetchall():
                alias_lower = row[0]
                result[alias_lower] = self._row_to_entity(row[1:])

            # For names not found in aliases, try direct entity lookup
            remaining = [n for n in lower_names if n not in result]
            if remaining:
                cur.execute(
                    """
                    SELECT LOWER(name), id, name, embedding, created_at, out_degree
                    FROM entities WHERE LOWER(name) = ANY(%s)
                    """,
                    (remaining,),
                )
                for row in cur.fetchall():
                    name_lower = row[0]
                    result[name_lower] = self._row_to_entity(row[1:])

        return result

    def get_by_id(self, entity_id: int) -> EntityRow | None:
        """Get an entity by its ID."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, embedding, created_at, out_degree
                FROM entities WHERE id = %s
                """,
                (entity_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_entity(row)

    def get_all(self) -> list[EntityRow]:
        """Get all entities."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, embedding, created_at, out_degree
                FROM entities ORDER BY id
                """
            )
            return [self._row_to_entity(row) for row in cur.fetchall()]

    def get_by_ids(self, entity_ids: list[int]) -> list["EntityRow"]:
        """Get multiple entities by their IDs in a single query."""
        if not entity_ids:
            return []
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, embedding, created_at, out_degree
                FROM entities WHERE id = ANY(%s)
                """,
                (list(entity_ids),),
            )
            return [self._row_to_entity(row) for row in cur.fetchall()]

    def search_by_embedding(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[EntityRow, float]]:
        """Search entities by cosine similarity.

        Returns list of (EntityRow, cosine_distance) sorted by distance.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, embedding, created_at, out_degree,
                       embedding <=> %s::vector AS distance
                FROM entities
                WHERE embedding IS NOT NULL
                ORDER BY distance
                LIMIT %s
                """,
                (str(query_embedding), top_k),
            )
            results = []
            for row in cur.fetchall():
                entity = self._row_to_entity(row[:5])
                distance = row[5]
                results.append((entity, distance))
            return results

    def update_out_degree(self, entity_id: int, delta: int) -> None:
        """Update out_degree of an entity by a delta value.

        Args:
            entity_id: Entity ID.
            delta: Value to add to out_degree (can be negative).
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE entities SET out_degree = out_degree + %s WHERE id = %s",
                (delta, entity_id),
            )

    def find_seeds_above_tau(
        self,
        query_embedding: list[float],
        tau: float,
        initial_k: int = 100,
        max_k: int = 100000,
    ) -> list[tuple[int, float]]:
        """Find all entity seeds with similarity >= tau using Progressive Widening.

        Uses pgvector ORDER BY + LIMIT pattern to leverage the HNSW/IVFFlat index.
        Progressively doubles batch size until we find entities below tau threshold.

        Algorithm (Lazy PPR Phase 3):
        1. Fetch top-k entities by cosine distance
        2. If k-th entity has similarity >= tau, double k and repeat
        3. Return all entities with similarity >= tau

        This is mathematically equivalent to scanning all entities but uses
        the pgvector index for efficiency.

        Args:
            query_embedding: Query embedding vector.
            tau: Similarity threshold (0 < tau <= 1). Only entities with
                 cosine similarity >= tau are returned.
            initial_k: Initial batch size (default 100).
            max_k: Maximum batch size to prevent runaway queries (default 100000).

        Returns:
            List of (entity_id, similarity) tuples for all entities with
            similarity >= tau, sorted by similarity descending.
        """
        # Convert embedding to string once for reuse
        embedding_str = str(query_embedding)
        collected: list[tuple[int, float]] = []
        offset = 0
        fetch_size = initial_k

        with self._conn.cursor() as cur:
            while True:
                cur.execute(
                    """
                    SELECT id, 1 - (embedding <=> %s::vector) AS similarity
                    FROM entities
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s OFFSET %s
                    """,
                    (embedding_str, embedding_str, fetch_size, offset),
                )
                batch = cur.fetchall()

                if not batch:
                    # No more entities
                    break

                # Add all entities above tau from this batch
                for row in batch:
                    entity_id, similarity = row[0], row[1]
                    if similarity >= tau:
                        collected.append((entity_id, similarity))
                    else:
                        # Found boundary - no need to fetch more
                        return collected

                # Check if we've hit max_k limit
                offset += len(batch)
                if offset >= max_k:
                    logger.warning(
                        "find_seeds_above_tau hit max_k=%d, tau=%.4f. "
                        "All %d entities have similarity >= tau. Returning all.",
                        max_k, tau, len(collected),
                    )
                    return collected

                # Fetch size stays the same (no doubling needed with OFFSET)

        return collected

    def _row_to_entity(self, row: tuple) -> EntityRow:
        """Convert a database row tuple to EntityRow.

        Expected row format (5 columns):
        0: id, 1: name, 2: embedding, 3: created_at, 4: out_degree
        """
        import json

        embedding = row[2]
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        return EntityRow(
            id=row[0],
            name=row[1],
            embedding=embedding,
            created_at=row[3],
            out_degree=row[4] if len(row) > 4 and row[4] is not None else 0,
        )
