"""Startup scanner for incremental indexing.

Scans session directories on service startup and enqueues sessions
that need indexing (new, changed, or previously failed).
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from reverie.config import get_profiles, ProfileConfig
from reverie.db.connection import get_connection
from reverie.db.repositories import SessionIndexStateRepository
from reverie.indexer.work_queue import IndexTask, TaskPriority, WorkQueue

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result of scanning a profile's session directory."""
    profile_id: str
    new_sessions: int = 0
    changed_sessions: int = 0
    failed_sessions: int = 0
    unchanged_sessions: int = 0
    errors: int = 0
    elapsed_ms: float = 0.0

    @property
    def total_enqueued(self) -> int:
        """Total sessions enqueued for indexing."""
        return self.new_sessions + self.changed_sessions + self.failed_sessions


class StartupScanner:
    """Scans session directories and enqueues sessions for indexing.

    Compares filesystem state with database session_index_state to find:
    - New sessions: No DB entry exists
    - Changed sessions: File mtime > DB mtime
    - Failed sessions: DB status = 'failed'

    Usage:
        queue = WorkQueue()
        scanner = StartupScanner(queue)

        # Scan all profiles
        results = await scanner.scan_all()

        # Or scan specific profile
        result = await scanner.scan_profile("my_profile")
    """

    def __init__(self, queue: WorkQueue):
        """Initialize the scanner.

        Args:
            queue: Work queue to enqueue indexing tasks.
        """
        self._queue = queue

    async def scan_all(self) -> list[ScanResult]:
        """Scan all configured profiles.

        Returns:
            List of ScanResult for each profile.
        """
        profiles = get_profiles()
        results = []

        logger.info(f"Starting scan for {len(profiles)} profiles")

        for profile_id, profile in profiles.items():
            try:
                result = await self.scan_profile(profile)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to scan profile {profile_id}: {e}")
                results.append(ScanResult(
                    profile_id=profile_id,
                    errors=1,
                ))

        total_enqueued = sum(r.total_enqueued for r in results)
        logger.info(f"Startup scan complete: {total_enqueued} sessions enqueued")

        return results

    async def scan_profile(self, profile: ProfileConfig) -> ScanResult:
        """Scan a single profile's session directories.

        Supports multiple sessions directories per profile for multi-folder indexing.

        Args:
            profile: Profile configuration.

        Returns:
            ScanResult with aggregated statistics from all directories.
        """
        start_time = time.time()
        result = ScanResult(profile_id=profile.id)

        # Get all sessions directories (multi-folder support)
        sessions_dirs = profile.sessions_paths

        if not sessions_dirs:
            logger.warning(f"[{profile.id}] No sessions directories configured")
            return result

        # Scan each directory
        for sessions_dir in sessions_dirs:
            dir_result = await self._scan_directory(
                profile.id, sessions_dir
            )
            # Aggregate results
            result.new_sessions += dir_result.new_sessions
            result.changed_sessions += dir_result.changed_sessions
            result.failed_sessions += dir_result.failed_sessions
            result.unchanged_sessions += dir_result.unchanged_sessions
            result.errors += dir_result.errors

        result.elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            f"[{profile.id}] Scan complete ({len(sessions_dirs)} dirs): "
            f"new={result.new_sessions}, changed={result.changed_sessions}, "
            f"failed={result.failed_sessions}, unchanged={result.unchanged_sessions} "
            f"({result.elapsed_ms:.0f}ms)"
        )

        return result

    async def _scan_directory(
        self, profile_id: str, sessions_dir: Path
    ) -> ScanResult:
        """Scan a single directory for session files.

        Args:
            profile_id: Profile identifier.
            sessions_dir: Directory to scan.

        Returns:
            ScanResult for this directory.
        """
        result = ScanResult(profile_id=profile_id)
        indexing_path = str(sessions_dir)

        if not sessions_dir.exists():
            logger.warning(f"[{profile_id}] Sessions directory not found: {sessions_dir}")
            return result

        # Get all .jsonl session files, sorted by mtime (newest first)
        session_files = sorted(
            sessions_dir.glob("*.jsonl"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        logger.info(f"[{profile_id}] Found {len(session_files)} session files in {sessions_dir}")

        if not session_files:
            return result

        # Get existing session states from DB for this specific path
        with get_connection() as conn:
            state_repo = SessionIndexStateRepository(conn)
            existing_states = state_repo.get_by_profile(
                profile_id, indexing_path=indexing_path
            )

        # Build lookup map
        state_map = {s.session_id: s for s in existing_states}

        for file_path in session_files:
            try:
                session_id = file_path.stem  # filename without extension
                file_stat = file_path.stat()
                file_mtime = datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc)

                state = state_map.get(session_id)

                if state is None:
                    # New session
                    await self._enqueue(
                        profile_id, session_id, file_path, TaskPriority.LOW,
                        file_stat.st_mtime, indexing_path
                    )
                    result.new_sessions += 1
                    logger.debug(f"[{profile_id}] New session: {session_id} ({indexing_path})")

                elif state.status in ("failed", "in_progress"):
                    # Previously failed or interrupted - retry
                    await self._enqueue(
                        profile_id, session_id, file_path, TaskPriority.LOW,
                        file_stat.st_mtime, indexing_path
                    )
                    result.failed_sessions += 1
                    logger.debug(f"[{profile_id}] Retrying {state.status} session: {session_id}")

                elif file_mtime > state.file_mtime:
                    # File changed since last index
                    await self._enqueue(
                        profile_id, session_id, file_path, TaskPriority.LOW,
                        file_stat.st_mtime, indexing_path
                    )
                    result.changed_sessions += 1
                    logger.debug(f"[{profile_id}] Changed session: {session_id}")

                else:
                    # Unchanged
                    result.unchanged_sessions += 1

            except Exception as e:
                logger.error(f"[{profile_id}] Error processing {file_path.name}: {e}")
                result.errors += 1

        return result

    async def _enqueue(
        self,
        profile_id: str,
        session_id: str,
        file_path: Path,
        priority: TaskPriority,
        file_mtime: float | None = None,
        indexing_path: str = "",
    ) -> bool:
        """Enqueue a session for indexing.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            file_path: Path to session JSONL file.
            priority: Task priority.
            file_mtime: File modification time (used for ordering newer files first).
            indexing_path: Folder path being indexed (for multi-folder support).

        Returns:
            True if enqueued, False if duplicate.
        """
        # Use negative mtime so newer files (higher mtime) get lower created_at
        # which means higher priority in the queue.
        # Fall back to current time if mtime not provided.
        created_at = -file_mtime if file_mtime is not None else time.time()

        task = IndexTask(
            priority=priority,
            created_at=created_at,
            profile_id=profile_id,
            session_id=session_id,
            file_path=file_path,
            indexing_path=indexing_path,
        )
        return await self._queue.put(task)
