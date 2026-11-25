"""Indexing pipeline orchestrator for MemoryLink v2.

Combines parser, preprocessing, embedding, OpenIE (NER only), entity filtering,
and DB repository modules to index session JSONL files into the bipartite graph.

Pipeline per session:
1. parse_session() → passages (chunks)
2. For each passage:
   a. Preprocess (filter/refine turn text) → drop or refined text
   b. OpenIE NER (extract entities, no triple extraction)
   c. EntityFilter (drop noise entities, normalize names, deduplicate)
   d. Generate embeddings (passage, entities)
   e. Store in DB (passages, entities, entity_passages)
3. Create passage_adjacency edges (consecutive passages)
4. Update processed_sessions status
"""

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from reverie.db.connection import get_connection
from reverie.db.repositories import (
    AdjacencyRepository,
    EntityPassageRepository,
    EntityRepository,
    PassageRepository,
)
from reverie.config import get_settings
from reverie.indexer.entity_filter import EntityFilter
from reverie.indexer.embedding import EmbeddingClient
from reverie.indexer.openie import OpenIEClient
from reverie.indexer.preprocessor import PreprocessorClient
from reverie.parser import parse_session, serialize_passage

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Timing statistics for indexing stages."""

    parse_ms: float = 0.0
    preprocess_ms: float = 0.0
    ner_ms: float = 0.0
    embedding_combined_ms: float = 0.0
    embedding_user_ms: float = 0.0
    embedding_assistant_ms: float = 0.0
    embedding_entity_ms: float = 0.0
    db_write_ms: float = 0.0
    adjacency_ms: float = 0.0
    total_ms: float = 0.0

    def summary(self) -> str:
        """Return a formatted summary of timing stats."""
        return (
            f"parse={self.parse_ms:.0f}ms, "
            f"preprocess={self.preprocess_ms:.0f}ms, "
            f"NER={self.ner_ms:.0f}ms, "
            f"emb[c={self.embedding_combined_ms:.0f}/u={self.embedding_user_ms:.0f}/"
            f"a={self.embedding_assistant_ms:.0f}/e={self.embedding_entity_ms:.0f}]ms, "
            f"db={self.db_write_ms:.0f}ms, "
            f"adj={self.adjacency_ms:.0f}ms, "
            f"total={self.total_ms:.0f}ms"
        )


@dataclass
class IndexResult:
    """Result of indexing a single session."""

    session_id: str
    success: bool
    passages_count: int = 0
    passages_dropped: int = 0  # Dropped by preprocessor
    entities_count: int = 0
    error: str | None = None
    timing: TimingStats = field(default_factory=TimingStats)


class IndexingPipeline:
    """Orchestrates the full indexing pipeline for MemoryLink v2.

    v2 uses:
    - entities, passages (core nodes)
    - entity_passages (Entity↔Passage contain edges)
    - passage_adjacency (Passage↔Passage temporal edges)

    No fact/triple extraction - only NER for entity extraction.

    Usage:
        pipeline = IndexingPipeline()
        result = pipeline.index_session(Path("session.jsonl"))
        results = pipeline.index_batch([Path("a.jsonl"), Path("b.jsonl")])
    """

    def __init__(self, verbose: bool = False):
        """Initialize with default clients from config.

        Args:
            verbose: If True, log detailed timing information per passage.
        """
        settings = get_settings()
        self._embedding_client = EmbeddingClient()
        self._openie_client = OpenIEClient()
        self._entity_filter = EntityFilter()
        self._verbose = verbose
        # Preprocessor is optional (controlled by config)
        self._preprocessor = (
            PreprocessorClient() if settings.preprocessor_enabled else None
        )

    @property
    def embedding_client(self) -> EmbeddingClient:
        """Expose embedding client for external use."""
        return self._embedding_client

    @property
    def openie_client(self) -> OpenIEClient:
        """Expose OpenIE client for external use (NER only)."""
        return self._openie_client

    def index_session(self, session_path: Path) -> IndexResult:
        """Index a single session JSONL file.

        Steps:
        1. Check if already processed
        2. Parse into passages
        3. For each passage: NER + embeddings + DB store
        4. Create passage adjacency edges
        5. Update processed_sessions

        Returns IndexResult with success/failure info.
        """
        session_id = session_path.stem
        logger.info("Indexing session: %s", session_id)

        session_start = time.time()
        timing = TimingStats()

        try:
            with get_connection() as conn:
                # Check if already processed
                if self._is_processed(conn, session_id):
                    logger.info("Session %s already processed, skipping", session_id)
                    return IndexResult(
                        session_id=session_id,
                        success=True,
                        passages_count=0,
                    )

                # Parse session into passages
                t0 = time.time()
                passages = list(parse_session(session_path))
                timing.parse_ms = (time.time() - t0) * 1000

                if not passages:
                    logger.info("Session %s has no passages", session_id)
                    self._mark_processed(conn, session_id, session_path, 0)
                    return IndexResult(
                        session_id=session_id,
                        success=True,
                        passages_count=0,
                    )

                # Initialize repositories
                passage_repo = PassageRepository(conn)
                entity_repo = EntityRepository(conn)
                entity_passage_repo = EntityPassageRepository(conn)
                adjacency_repo = AdjacencyRepository(conn)

                total_entities: set[str] = set()
                passages_dropped = 0
                passages_indexed = 0

                for i, passage in enumerate(passages):
                    # Preprocess: filter/refine passage text
                    t0 = time.time()
                    preprocessed_text = self._preprocess_passage(passage)
                    timing.preprocess_ms += (time.time() - t0) * 1000

                    if preprocessed_text is None:
                        passages_dropped += 1
                        continue

                    passage_timing = self._index_passage(
                        passage=passage,
                        preprocessed_text=preprocessed_text,
                        passage_repo=passage_repo,
                        entity_repo=entity_repo,
                        entity_passage_repo=entity_passage_repo,
                        total_entities=total_entities,
                        conn=conn,
                    )

                    # Accumulate timing
                    timing.ner_ms += passage_timing.ner_ms
                    timing.embedding_combined_ms += passage_timing.embedding_combined_ms
                    timing.embedding_user_ms += passage_timing.embedding_user_ms
                    timing.embedding_assistant_ms += passage_timing.embedding_assistant_ms
                    timing.embedding_entity_ms += passage_timing.embedding_entity_ms
                    timing.db_write_ms += passage_timing.db_write_ms

                    passages_indexed += 1

                    if self._verbose and (i + 1) % 5 == 0:
                        logger.info(
                            "  [%s] Passage %d/%d processed",
                            session_id[:8],
                            i + 1,
                            len(passages),
                        )

                # Create passage adjacency edges for MemoryLink v2
                # This connects consecutive passages within the session
                t0 = time.time()
                settings = get_settings()
                adjacency_window = settings.v2_adjacency_window
                adjacency_count = adjacency_repo.create_for_session(
                    session_id, window=adjacency_window
                )
                timing.adjacency_ms = (time.time() - t0) * 1000

                if adjacency_count > 0:
                    logger.debug(
                        "Created %d passage adjacencies for session %s",
                        adjacency_count,
                        session_id,
                    )

                self._mark_processed(
                    conn, session_id, session_path, passages_indexed
                )

                timing.total_ms = (time.time() - session_start) * 1000

                result = IndexResult(
                    session_id=session_id,
                    success=True,
                    passages_count=passages_indexed,
                    passages_dropped=passages_dropped,
                    entities_count=len(total_entities),
                    timing=timing,
                )

                if self._verbose:
                    logger.info(
                        "Session %s timing: %s",
                        session_id[:8],
                        timing.summary(),
                    )

                logger.info(
                    "Session %s indexed: %d passages (%d dropped), %d entities",
                    session_id,
                    result.passages_count,
                    result.passages_dropped,
                    result.entities_count,
                )
                return result

        except Exception as e:
            logger.error("Failed to index session %s: %s", session_id, e)
            # Try to record the failure
            try:
                with get_connection() as conn:
                    self._mark_failed(conn, session_id, session_path, str(e))
            except Exception:
                pass
            return IndexResult(
                session_id=session_id,
                success=False,
                error=str(e),
            )

    def index_batch(self, session_paths: list[Path]) -> list[IndexResult]:
        """Index multiple sessions sequentially.

        Continues processing after individual failures.
        """
        results = []
        total = len(session_paths)

        for i, path in enumerate(session_paths, 1):
            logger.info("Batch progress: %d/%d", i, total)
            result = self.index_session(path)
            results.append(result)

        succeeded = sum(1 for r in results if r.success)
        failed = total - succeeded
        logger.info(
            "Batch complete: %d/%d succeeded, %d failed",
            succeeded,
            total,
            failed,
        )
        return results

    def _preprocess_passage(self, passage) -> str | None:
        """Preprocess a passage: filter or refine its text.

        Returns refined text if kept, None if dropped.
        """
        if self._preprocessor is None:
            # Preprocessing disabled, return original text
            return passage.text

        result = self._preprocessor.preprocess_turn(passage.text)
        if result.action == "drop":
            logger.debug(
                "Passage dropped by preprocessor: %s (reason: %s)",
                passage.session_id,
                result.reason,
            )
            return None

        return result.refined_text

    def _index_passage(
        self,
        passage,
        preprocessed_text: str,
        passage_repo: PassageRepository,
        entity_repo: EntityRepository,
        entity_passage_repo: EntityPassageRepository,
        total_entities: set[str],
        conn,
    ) -> TimingStats:
        """Index a single passage: NER + embeddings + DB store.

        v2 simplified pipeline:
        1. Serialize passage to text
        2. NER to extract entities (no triples)
        3. Filter and normalize entities
        4. Store passage with embeddings (combined + user/assistant)
        5. Upsert entities with embeddings
        6. Create entity-passage links

        Returns:
            TimingStats for this passage's processing.
        """
        timing = TimingStats()

        # Serialize passage to text (with metadata header for DB/embedding)
        text = serialize_passage(passage, body_override=preprocessed_text)

        # Run NER only (no triple extraction in v2)
        t0 = time.time()
        raw_entities = self._openie_client.extract_entities(preprocessed_text)
        timing.ner_ms = (time.time() - t0) * 1000

        # Apply entity normalization and filtering
        entities = self._entity_filter.filter_entities(raw_entities)

        # Generate passage embeddings
        # Combined embedding (for backward compatibility)
        t0 = time.time()
        passage_embedding = self._embedding_client.embed_text(text)
        timing.embedding_combined_ms = (time.time() - t0) * 1000

        # User embedding (from user_text)
        user_embedding = None
        if passage.user_text:
            t0 = time.time()
            user_embedding = self._embedding_client.embed_text(passage.user_text)
            timing.embedding_user_ms = (time.time() - t0) * 1000

        # Assistant embedding (from assistant_text)
        assistant_embedding = None
        if passage.assistant_text:
            t0 = time.time()
            assistant_embedding = self._embedding_client.embed_text(
                passage.assistant_text
            )
            timing.embedding_assistant_ms = (time.time() - t0) * 1000

        # Store passage with all embeddings
        t0 = time.time()
        passage_id = passage_repo.insert(
            session_id=passage.session_id,
            turn_index=passage.turn_index,
            text=text,
            metadata={"token_count": passage.token_count},
            embedding=passage_embedding,
            user_text=passage.user_text,
            user_embedding=user_embedding,
            assistant_text=passage.assistant_text,
            assistant_embedding=assistant_embedding,
        )
        timing.db_write_ms = (time.time() - t0) * 1000

        # Process entities (retry on deadlock)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with conn.cursor() as cur:
                    cur.execute("SAVEPOINT entity_upsert")
                    for entity_name in entities:
                        # Generate entity embedding
                        t0 = time.time()
                        entity_embedding = self._embedding_client.embed_text(
                            entity_name
                        )
                        timing.embedding_entity_ms += (time.time() - t0) * 1000

                        # Upsert entity using DB procedure (with alias resolution)
                        t0 = time.time()
                        cur.execute(
                            "SELECT entity_id, resolved_from_alias FROM upsert_entity(%s, %s::vector)",
                            (entity_name, str(entity_embedding)),
                        )
                        result = cur.fetchone()
                        entity_id = result[0]
                        total_entities.add(entity_name)

                        # Create entity-passage link
                        cur.execute(
                            """INSERT INTO entity_passages (entity_id, passage_id)
                               VALUES (%s, %s)
                               ON CONFLICT (entity_id, passage_id) DO NOTHING""",
                            (entity_id, passage_id),
                        )
                        timing.db_write_ms += (time.time() - t0) * 1000
                    cur.execute("RELEASE SAVEPOINT entity_upsert")
                break  # Success
            except Exception as e:
                err_name = type(e).__name__
                if err_name == "DeadlockDetected" and attempt < max_retries - 1:
                    with conn.cursor() as cur:
                        cur.execute("ROLLBACK TO SAVEPOINT entity_upsert")
                    delay = (0.1 * (2**attempt)) + random.uniform(0, 0.05)
                    logger.warning(
                        "Deadlock on entity upsert (attempt %d/%d), "
                        "retrying in %.2fs",
                        attempt + 1, max_retries, delay,
                    )
                    time.sleep(delay)
                else:
                    raise

        return timing

    def _is_processed(self, conn, session_id: str) -> bool:
        """Check if session is already processed."""
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM processed_sessions
                WHERE session_id = %s AND status = 'completed'
                """,
                (session_id,),
            )
            return cur.fetchone() is not None

    def _mark_processed(
        self, conn, session_id: str, file_path: Path, chunk_count: int
    ) -> None:
        """Mark session as successfully processed."""
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO processed_sessions
                    (session_id, file_path, status, chunk_count)
                VALUES (%s, %s, 'completed', %s)
                ON CONFLICT (session_id) DO UPDATE
                    SET status = 'completed',
                        chunk_count = EXCLUDED.chunk_count,
                        processed_at = NOW()
                """,
                (session_id, str(file_path), chunk_count),
            )

    def _mark_failed(
        self, conn, session_id: str, file_path: Path, error: str
    ) -> None:
        """Mark session as failed."""
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO processed_sessions
                    (session_id, file_path, status, error_message)
                VALUES (%s, %s, 'failed', %s)
                ON CONFLICT (session_id) DO UPDATE
                    SET status = 'failed',
                        error_message = EXCLUDED.error_message,
                        processed_at = NOW()
                """,
                (session_id, str(file_path), error),
            )
