"""Incremental indexer for turn-by-turn session indexing.

Supports atomic transactions per turn with adjacency chain preservation,
file change detection, and resumable indexing from any failure point.
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from reverie.config import get_settings
from reverie.db.connection import get_connection
from reverie.db.repositories import (
    AdjacencyRepository,
    EntityPassageRepository,
    EntityRepository,
    PassageProcessingHistoryRepository,
    PassageRepository,
    SessionIndexStateRepository,
)
from reverie.indexer.entity_filter import EntityFilter
from reverie.indexer.embedding import EmbeddingClient
from reverie.indexer.openie import OpenIEClient
from reverie.indexer.preprocessor import PreprocessorClient
from reverie.parser import serialize_passage
from reverie.parser.models import AssistantEntry, Passage, UserEntry

logger = logging.getLogger(__name__)


@dataclass
class TurnData:
    """Parsed turn data from JSONL entries."""
    turn_index: int
    start_line: int  # 0-based line number where turn starts
    end_line: int    # 0-based line number where turn ends (exclusive)
    entries: list    # Raw JSONL entries for this turn
    passage: Passage | None = None  # Parsed passage (after chunking)


@dataclass
class ProcessResult:
    """Result of processing a single turn."""
    action: str  # 'indexed', 'dropped', 'skipped'
    passage_id: int | None = None
    entities: list[str] = field(default_factory=list)
    drop_reason: str | None = None


@dataclass
class IndexResult:
    """Result of indexing a session."""
    session_id: str
    profile_id: str
    success: bool
    new_passages: int = 0
    dropped: int = 0
    skipped: int = 0
    error: str | None = None
    elapsed_ms: float = 0.0


class IncrementalIndexer:
    """Turn-by-turn incremental indexer with atomic transactions.

    Features:
    - Parses new turns from last indexed line
    - Commits each turn atomically with its adjacency
    - Detects file shrinkage for full reindex
    - Restores adjacency chain from last_passage_id

    Usage:
        indexer = IncrementalIndexer()
        result = await indexer.index_session(
            profile_id="my_profile",
            session_id="abc123",
            file_path=Path("/path/to/session.jsonl")
        )
    """

    def __init__(self):
        """Initialize with default clients from config."""
        settings = get_settings()
        self._embedding_client = EmbeddingClient()
        self._openie_client = OpenIEClient()
        self._entity_filter = EntityFilter()
        self._preprocessor = (
            PreprocessorClient() if settings.preprocessor_enabled else None
        )
        self._settings = settings

    def index_session(
        self,
        profile_id: str,
        session_id: str,
        file_path: Path,
        indexing_path: str = "",
    ) -> IndexResult:
        """Index a session file incrementally.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier (typically the file stem).
            file_path: Path to the session JSONL file.
            indexing_path: The base directory this session belongs to (for multi-folder support).

        Returns:
            IndexResult with success/failure info and statistics.
        """
        start_time = time.time()
        logger.info(f"[{profile_id}] Indexing session: {session_id}")

        try:
            with get_connection() as conn:
                # Get or create session state
                state_repo = SessionIndexStateRepository(conn)
                state = state_repo.get_by_session(
                    profile_id, session_id, indexing_path=indexing_path
                )

                # Check file stats
                file_stat = file_path.stat()
                file_mtime = datetime.fromtimestamp(
                    file_stat.st_mtime, tz=timezone.utc
                )
                file_size = file_stat.st_size

                # Handle file shrinkage (file was rewritten)
                if state and file_size < state.file_size_bytes:
                    logger.warning(
                        f"[{profile_id}] File shrunk for {session_id}, "
                        f"triggering full reindex"
                    )
                    return self._full_reindex(
                        conn, profile_id, session_id, file_path, state_repo,
                        indexing_path=indexing_path
                    )

                # Check if file is unchanged
                if state and state.status == "completed":
                    if file_stat.st_mtime <= state.file_mtime.timestamp():
                        elapsed = (time.time() - start_time) * 1000
                        logger.debug(f"[{profile_id}] Session {session_id} unchanged")
                        return IndexResult(
                            session_id=session_id,
                            profile_id=profile_id,
                            success=True,
                            elapsed_ms=elapsed,
                        )

                # Count lines in file
                line_count = self._count_lines(file_path)

                # Create or update session state
                if state is None:
                    state = state_repo.upsert(
                        profile_id=profile_id,
                        session_id=session_id,
                        file_path=str(file_path),
                        file_mtime=file_mtime,
                        file_size_bytes=file_size,
                        file_lines_count=line_count,
                        indexing_path=indexing_path,
                    )
                    conn.commit()
                else:
                    # Update file info
                    state_repo.upsert(
                        profile_id=profile_id,
                        session_id=session_id,
                        file_path=str(file_path),
                        file_mtime=file_mtime,
                        file_size_bytes=file_size,
                        file_lines_count=line_count,
                        session_seq=state.session_seq,  # Preserve existing seq
                        indexing_path=indexing_path,
                    )
                    conn.commit()

                # Refresh state after upsert
                state = state_repo.get_by_session(
                    profile_id, session_id, indexing_path=indexing_path
                )

                # Parse new turns from last indexed line
                start_line = state.last_indexed_line
                new_turns = list(self._parse_new_turns(file_path, start_line))

                if not new_turns:
                    # No new turns
                    state_repo.mark_completed(
                        profile_id, session_id, indexing_path=indexing_path
                    )
                    conn.commit()
                    elapsed = (time.time() - start_time) * 1000
                    return IndexResult(
                        session_id=session_id,
                        profile_id=profile_id,
                        success=True,
                        elapsed_ms=elapsed,
                    )

                # Initialize repositories
                passage_repo = PassageRepository(conn)
                entity_repo = EntityRepository(conn)
                entity_passage_repo = EntityPassageRepository(conn)
                adjacency_repo = AdjacencyRepository(conn)
                history_repo = PassageProcessingHistoryRepository(conn)

                # Track previous passage for adjacency
                prev_passage_id = state.last_passage_id
                session_seq = state.session_seq

                indexed = 0
                dropped = 0
                skipped = 0

                for turn in new_turns:
                    # Process turn (preprocess + NER + embed + store)
                    result = self._process_turn(
                        conn=conn,
                        profile_id=profile_id,
                        session_id=session_id,
                        session_seq=session_seq,
                        turn=turn,
                        prev_passage_id=prev_passage_id,
                        passage_repo=passage_repo,
                        entity_repo=entity_repo,
                        entity_passage_repo=entity_passage_repo,
                        adjacency_repo=adjacency_repo,
                        history_repo=history_repo,
                    )

                    if result.action == "indexed":
                        indexed += 1
                        prev_passage_id = result.passage_id
                    elif result.action == "dropped":
                        dropped += 1
                        # Dropped turns don't break adjacency chain
                    elif result.action == "skipped":
                        skipped += 1

                    # Update progress and commit (atomic per turn)
                    state_repo.update_progress(
                        profile_id=profile_id,
                        session_id=session_id,
                        last_indexed_turn=turn.turn_index,
                        last_indexed_line=turn.end_line,
                        last_passage_id=prev_passage_id,
                        passages_indexed_delta=1 if result.action == "indexed" else 0,
                        passages_dropped_delta=1 if result.action == "dropped" else 0,
                        indexing_path=indexing_path,
                    )
                    conn.commit()

                # Mark session as completed
                state_repo.mark_completed(
                    profile_id, session_id, indexing_path=indexing_path
                )
                conn.commit()

                elapsed = (time.time() - start_time) * 1000
                logger.info(
                    f"[{profile_id}] Session {session_id} indexed: "
                    f"{indexed} passages, {dropped} dropped, {skipped} skipped "
                    f"({elapsed:.0f}ms)"
                )

                return IndexResult(
                    session_id=session_id,
                    profile_id=profile_id,
                    success=True,
                    new_passages=indexed,
                    dropped=dropped,
                    skipped=skipped,
                    elapsed_ms=elapsed,
                )

        except Exception as e:
            logger.error(
                f"[{profile_id}] Failed to index session {session_id}: {e}",
                exc_info=True,
            )
            # Try to mark as failed
            try:
                with get_connection() as conn:
                    state_repo = SessionIndexStateRepository(conn)
                    state_repo.mark_failed(
                        profile_id, session_id, str(e), indexing_path=indexing_path
                    )
                    conn.commit()
            except Exception:
                pass

            elapsed = (time.time() - start_time) * 1000
            return IndexResult(
                session_id=session_id,
                profile_id=profile_id,
                success=False,
                error=str(e),
                elapsed_ms=elapsed,
            )

    def _full_reindex(
        self,
        conn,
        profile_id: str,
        session_id: str,
        file_path: Path,
        state_repo: SessionIndexStateRepository,
        indexing_path: str = "",
    ) -> IndexResult:
        """Perform full reindex when file was rewritten.

        Deletes all existing passages and reindexes from scratch.
        """
        # Reset state but keep session_seq
        state_repo.reset_for_full_reindex(
            profile_id, session_id, indexing_path=indexing_path
        )

        # Delete existing passages for this session
        passage_repo = PassageRepository(conn)
        deleted = passage_repo.delete_by_session(session_id)
        logger.info(f"[{profile_id}] Deleted {deleted} passages for full reindex")

        # Delete processing history
        history_repo = PassageProcessingHistoryRepository(conn)
        history_repo.delete_by_session(profile_id, session_id)

        conn.commit()

        # Reindex from scratch
        return self.index_session(
            profile_id, session_id, file_path, indexing_path=indexing_path
        )

    def _count_lines(self, file_path: Path) -> int:
        """Count total lines in a file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def _parse_new_turns(
        self, file_path: Path, start_line: int
    ) -> Iterator[TurnData]:
        """Parse new turns from file starting at given line.

        Reads file from start_line and groups entries into turns.
        A turn starts at each user text message.

        Args:
            file_path: Path to JSONL file.
            start_line: 0-based line number to start from.

        Yields:
            TurnData objects for each complete turn.
        """
        entries = []
        current_turn_start = start_line
        turn_index = -1  # Will be incremented on first user message
        line_num = 0

        with open(file_path, "r", encoding="utf-8") as f:
            # Skip to start_line
            for _ in range(start_line):
                next(f, None)
                line_num += 1

            for line in f:
                line = line.strip()
                if not line:
                    line_num += 1
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # Incomplete JSON - stop processing
                    logger.warning(
                        f"Incomplete JSON at line {line_num}, stopping"
                    )
                    break

                entry_type = data.get("type")

                # Check for turn boundary (user text message)
                is_turn_boundary = (
                    entry_type == "user"
                    and isinstance(data.get("message", {}).get("content"), str)
                )

                if is_turn_boundary and entries:
                    # Next user text means previous turn is complete
                    # (matching turn_chunker.py behavior)
                    turn_data = self._build_turn_data(
                        entries=entries,
                        turn_index=turn_index,
                        start_line=current_turn_start,
                        end_line=line_num,
                    )
                    if turn_data:
                        yield turn_data

                    # Start new turn
                    entries = [data]
                    current_turn_start = line_num
                    turn_index += 1
                else:
                    if is_turn_boundary and not entries:
                        # First user message in session
                        turn_index = 0
                    entries.append(data)

                line_num += 1

            # Handle last turn - complete if has any assistant response
            if entries and self._has_assistant_response(entries):
                turn_data = self._build_turn_data(
                    entries=entries,
                    turn_index=turn_index,
                    start_line=current_turn_start,
                    end_line=line_num,
                )
                if turn_data:
                    yield turn_data

    def _has_assistant_response(self, entries: list[dict]) -> bool:
        """Check if turn has any assistant response.

        For the last turn in a file, we consider it complete if there's
        at least one assistant entry with content.
        """
        for entry in entries:
            if entry.get("type") == "assistant":
                content = entry.get("message", {}).get("content", [])
                if content:
                    return True
        return False

    def _build_turn_data(
        self,
        entries: list[dict],
        turn_index: int,
        start_line: int,
        end_line: int,
    ) -> TurnData | None:
        """Build TurnData from raw entries.

        Converts entries to Passage using existing turn_chunker logic.
        """
        from reverie.parser.jsonl_reader import filter_messages
        from reverie.parser.models import AssistantEntry, UserEntry
        from reverie.parser.turn_chunker import chunk_by_turns

        # Convert dict entries to model objects
        messages = []
        for entry in entries:
            entry_type = entry.get("type")
            try:
                if entry_type == "user":
                    messages.append(UserEntry.model_validate(entry))
                elif entry_type == "assistant":
                    messages.append(AssistantEntry.model_validate(entry))
            except Exception as e:
                logger.debug(f"Skipping invalid entry: {e}")
                continue

        if not messages:
            return None

        # Use existing chunker to create passages
        # Note: chunk_by_turns may create multiple passages for oversized turns
        passages = list(chunk_by_turns(
            iter(messages),
            token_limit=self._settings.chunk_token_limit,
            tool_result_max_lines=self._settings.tool_result_max_lines,
        ))

        if not passages:
            return None

        # For incremental indexing, we handle one passage per turn
        # If oversized turn creates multiple passages, use the first
        passage = passages[0]

        return TurnData(
            turn_index=turn_index,
            start_line=start_line,
            end_line=end_line,
            entries=entries,
            passage=passage,
        )

    def _process_turn(
        self,
        conn,
        profile_id: str,
        session_id: str,
        session_seq: int,
        turn: TurnData,
        prev_passage_id: int | None,
        passage_repo: PassageRepository,
        entity_repo: EntityRepository,
        entity_passage_repo: EntityPassageRepository,
        adjacency_repo: AdjacencyRepository,
        history_repo: PassageProcessingHistoryRepository,
    ) -> ProcessResult:
        """Process a single turn: preprocess, NER, embed, store.

        All operations happen in the same transaction (caller commits).

        Args:
            conn: Database connection.
            profile_id: Profile identifier.
            session_id: Session identifier.
            session_seq: Session sequence number for display.
            turn: Turn data to process.
            prev_passage_id: Previous passage ID for adjacency.
            passage_repo: Passage repository.
            entity_repo: Entity repository.
            entity_passage_repo: Entity-passage repository.
            adjacency_repo: Adjacency repository.
            history_repo: Processing history repository.

        Returns:
            ProcessResult with action and passage_id if indexed.
        """
        passage = turn.passage
        if passage is None:
            # No valid passage (filtering removed all content)
            history_repo.record(
                profile_id=profile_id,
                session_id=session_id,
                turn_index=turn.turn_index,
                action="skipped",
                drop_reason="No valid passage after filtering",
            )
            return ProcessResult(action="skipped")

        # Step 1: Preprocess (filter/refine)
        if self._preprocessor:
            preprocess_result = self._preprocessor.preprocess_turn(passage.text)
            if preprocess_result.action == "drop":
                history_repo.record(
                    profile_id=profile_id,
                    session_id=session_id,
                    turn_index=turn.turn_index,
                    action="dropped",
                    drop_reason=preprocess_result.reason,
                    raw_text=passage.text[:500] if passage.text else None,
                )
                return ProcessResult(
                    action="dropped",
                    drop_reason=preprocess_result.reason,
                )
            preprocessed_text = preprocess_result.refined_text or passage.text
        else:
            preprocessed_text = passage.text

        # Step 2: NER (entity extraction)
        raw_entities = self._openie_client.extract_entities(preprocessed_text)
        entities = self._entity_filter.filter_entities(raw_entities)

        # Step 3: Generate embeddings
        text_for_embedding = serialize_passage(passage, body_override=preprocessed_text)
        passage_embedding = self._embedding_client.embed_text(text_for_embedding)

        user_embedding = None
        if passage.user_text:
            user_embedding = self._embedding_client.embed_text(passage.user_text)

        assistant_embedding = None
        if passage.assistant_text:
            assistant_embedding = self._embedding_client.embed_text(passage.assistant_text)

        # Step 4: Store passage with initial out_degree
        # out_degree starts with entity count; adjacency will add more via create()
        initial_out_degree = len(entities)
        passage_id = passage_repo.insert(
            session_id=session_id,
            turn_index=turn.turn_index,
            text=text_for_embedding,
            metadata={
                "token_count": passage.token_count,
                "profile_id": profile_id,
                "session_seq": session_seq,
            },
            embedding=passage_embedding,
            user_text=passage.user_text,
            user_embedding=user_embedding,
            assistant_text=passage.assistant_text,
            assistant_embedding=assistant_embedding,
            profile_id=profile_id,
            session_seq=session_seq,
            out_degree=initial_out_degree,
        )

        # Step 5: Store entities and links, update entity out_degree
        # Retry on deadlock (concurrent workers upserting same entity)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with conn.cursor() as cur:
                    cur.execute("SAVEPOINT entity_upsert")
                    for entity_name in entities:
                        entity_embedding = self._embedding_client.embed_text(
                            entity_name
                        )
                        cur.execute(
                            "SELECT entity_id, resolved_from_alias FROM upsert_entity(%s, %s::vector)",
                            (entity_name, str(entity_embedding)),
                        )
                        result = cur.fetchone()
                        entity_id = result[0]

                        # Insert entity-passage link
                        cur.execute(
                            """INSERT INTO entity_passages (entity_id, passage_id)
                               VALUES (%s, %s)
                               ON CONFLICT (entity_id, passage_id) DO NOTHING
                               RETURNING entity_id""",
                            (entity_id, passage_id),
                        )
                        # Only increment out_degree if the insert actually happened
                        if cur.fetchone() is not None:
                            cur.execute(
                                "UPDATE entities SET out_degree = out_degree + 1 WHERE id = %s",
                                (entity_id,),
                            )
                    cur.execute("RELEASE SAVEPOINT entity_upsert")
                break  # Success
            except Exception as e:
                err_name = type(e).__name__
                if err_name == "DeadlockDetected" and attempt < max_retries - 1:
                    with conn.cursor() as cur:
                        cur.execute("ROLLBACK TO SAVEPOINT entity_upsert")
                    delay = (0.1 * (2**attempt)) + random.uniform(0, 0.05)
                    logger.warning(
                        f"[{profile_id}] Deadlock on entity upsert "
                        f"(attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    raise

        # Step 6: Create adjacency (same transaction)
        if prev_passage_id is not None:
            adjacency_repo.create(prev_passage_id, passage_id)

        # Step 7: Record history
        history_repo.record(
            profile_id=profile_id,
            session_id=session_id,
            turn_index=turn.turn_index,
            action="indexed",
            preprocessed_text=preprocessed_text[:500] if preprocessed_text else None,
            entities_extracted=entities,
            passage_id=passage_id,
        )

        return ProcessResult(
            action="indexed",
            passage_id=passage_id,
            entities=entities,
        )
