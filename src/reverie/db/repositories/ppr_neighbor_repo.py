"""Repository for PPR neighbor queries.

Provides efficient batch queries for Lazy PPR iteration.
Uses UNION ALL to fetch all neighbor types in a single query per iteration.
"""

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class PPRNeighborEdge:
    """A single edge in the PPR graph with source, target, and normalization weight."""

    edge_type: Literal["e2p", "p2e", "p2p_a", "p2p_b"]
    source_id: int  # Entity ID for e2p, Passage ID for others
    target_id: int  # Passage ID for e2p, Entity ID for p2e, Passage ID for p2p
    out_degree: int  # Source node's out_degree for normalization


class PPRNeighborRepository:
    """Repository for PPR neighbor queries.

    Provides a single UNION ALL query to fetch all neighbor edges
    for active entities and passages in one database round-trip.
    """

    def __init__(self, conn):
        """Initialize with a psycopg2 connection."""
        self._conn = conn

    def get_all_neighbors(
        self,
        active_entity_ids: list[int],
        active_passage_ids: list[int],
    ) -> list[PPRNeighborEdge]:
        """Get all neighbor edges for active nodes in a single query.

        Uses UNION ALL to combine:
        1. e2p: Entity → Passage edges (entity contains passage)
        2. p2e: Passage → Entity edges (passage mentions entity)
        3. p2p_a: Passage → Passage edges (adjacency, a→b direction)
        4. p2p_b: Passage → Passage edges (adjacency, b→a direction)

        Args:
            active_entity_ids: List of currently active entity IDs.
            active_passage_ids: List of currently active passage IDs.

        Returns:
            List of PPRNeighborEdge containing all edges from active nodes.
        """
        # Handle empty inputs
        if not active_entity_ids and not active_passage_ids:
            return []

        # Build dynamic query based on which node types are active
        query_parts = []
        params = []

        if active_entity_ids:
            # e2p: Entity → Passage
            query_parts.append("""
                SELECT 'e2p' as type, ep.entity_id as src, ep.passage_id as dst, e.out_degree
                FROM entity_passages ep
                JOIN entities e ON e.id = ep.entity_id
                WHERE ep.entity_id = ANY(%s)
            """)
            params.append(active_entity_ids)

        if active_passage_ids:
            # p2e: Passage → Entity
            query_parts.append("""
                SELECT 'p2e', ep.passage_id, ep.entity_id, p.out_degree
                FROM entity_passages ep
                JOIN passages p ON p.id = ep.passage_id
                WHERE ep.passage_id = ANY(%s)
            """)
            params.append(active_passage_ids)

            # p2p_a: Passage adjacency (a → b)
            # Only include edges where source (passage_id_a) is active
            query_parts.append("""
                SELECT 'p2p_a', a.passage_id_a, a.passage_id_b, pa.out_degree
                FROM passage_adjacency a
                JOIN passages pa ON pa.id = a.passage_id_a
                WHERE a.passage_id_a = ANY(%s)
            """)
            params.append(active_passage_ids)

            # p2p_b: Passage adjacency (b → a)
            # Only include edges where source (passage_id_b) is active
            query_parts.append("""
                SELECT 'p2p_b', a.passage_id_b, a.passage_id_a, pb.out_degree
                FROM passage_adjacency a
                JOIN passages pb ON pb.id = a.passage_id_b
                WHERE a.passage_id_b = ANY(%s)
            """)
            params.append(active_passage_ids)

        if not query_parts:
            return []

        # Combine all query parts with UNION ALL
        full_query = " UNION ALL ".join(query_parts)

        logger.debug(
            "PPR neighbor query: %d entities, %d passages",
            len(active_entity_ids),
            len(active_passage_ids),
        )

        with self._conn.cursor() as cur:
            cur.execute(full_query, params)
            rows = cur.fetchall()

        edges = [
            PPRNeighborEdge(
                edge_type=row[0],
                source_id=row[1],
                target_id=row[2],
                out_degree=row[3] if row[3] is not None else 0,
            )
            for row in rows
        ]

        # Log edge type distribution
        if logger.isEnabledFor(logging.DEBUG):
            type_counts = {"e2p": 0, "p2e": 0, "p2p_a": 0, "p2p_b": 0}
            for edge in edges:
                type_counts[edge.edge_type] += 1
            logger.debug(
                "PPR edges fetched: total=%d, e2p=%d, p2e=%d, p2p_a=%d, p2p_b=%d",
                len(edges),
                type_counts["e2p"],
                type_counts["p2e"],
                type_counts["p2p_a"],
                type_counts["p2p_b"],
            )

        return edges
