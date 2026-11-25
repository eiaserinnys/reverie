"""Index scheduler for coordinating incremental indexing.

Integrates StartupScanner, SessionFileWatcher, and IndexWorkerPool
with flush delay handling for SDK callbacks.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reverie.config import get_profile, get_profiles
from reverie.indexer.file_watcher import SessionFileWatcher
from reverie.indexer.startup_scanner import StartupScanner
from reverie.indexer.work_queue import IndexTask, TaskPriority, WorkQueue
from reverie.indexer.worker_pool import IndexWorkerPool

logger = logging.getLogger(__name__)


# Retry delays in seconds (exponential backoff)
RETRY_DELAYS = [0.5, 1.0, 2.0, 4.0, 8.0]
MAX_RETRIES = len(RETRY_DELAYS)


@dataclass
class PendingSession:
    """Tracks a session pending indexing after SDK callback."""
    profile_id: str
    session_id: str
    file_path: Path
    indexing_path: str = ""
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class SchedulerStats:
    """Statistics for the index scheduler."""
    sdk_callbacks: int = 0
    sdk_retries: int = 0
    sdk_fallbacks: int = 0
    sdk_successes: int = 0


class IndexScheduler:
    """Coordinates all indexing components.

    Manages:
    - StartupScanner: Initial scan on startup
    - SessionFileWatcher: Continuous file monitoring
    - IndexWorkerPool: Parallel session processing
    - Retry handler: Flush delay handling for SDK callbacks

    Usage:
        scheduler = IndexScheduler()
        await scheduler.start()

        # Trigger indexing from SDK callback
        await scheduler.trigger_session("my_profile", "session_123")

        # Graceful shutdown
        await scheduler.stop()
    """

    def __init__(
        self,
        num_workers: int | None = None,
        debounce_seconds: float = 2.0,
    ):
        """Initialize the scheduler.

        Args:
            num_workers: Number of indexing workers (default from config).
            debounce_seconds: File watcher debounce delay.
        """
        self._queue = WorkQueue()
        self._pool = IndexWorkerPool(
            queue=self._queue,
            num_workers=num_workers,
            on_complete=self._on_index_complete,
        )
        self._scanner = StartupScanner(self._queue)
        self._watcher = SessionFileWatcher(
            queue=self._queue,
            debounce_seconds=debounce_seconds,
        )

        # Pending sessions awaiting flush (SDK callback tracking)
        self._pending_sessions: dict[str, PendingSession] = {}
        self._pending_lock = asyncio.Lock()

        self._stats = SchedulerStats()
        self._running = False
        self._retry_task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    @property
    def stats(self) -> dict[str, Any]:
        """Get aggregated scheduler statistics."""
        return {
            "running": self._running,
            "queue": self._queue.stats,
            "pool": self._pool.stats,
            "watcher": self._watcher.stats,
            "sdk": {
                "callbacks": self._stats.sdk_callbacks,
                "retries": self._stats.sdk_retries,
                "fallbacks": self._stats.sdk_fallbacks,
                "successes": self._stats.sdk_successes,
                "pending": len(self._pending_sessions),
            },
        }

    async def start(self, run_startup_scan: bool = True) -> None:
        """Start the scheduler and all components.

        Args:
            run_startup_scan: Whether to run initial scan (default True).
        """
        if self._running:
            logger.warning("Scheduler already running")
            return

        logger.info("Starting index scheduler")
        self._running = True

        # Start worker pool
        await self._pool.start()

        # Start file watcher
        await self._watcher.start()

        # Run startup scan
        if run_startup_scan:
            await self._scanner.scan_all()

        # Start retry handler
        self._retry_task = asyncio.create_task(
            self._retry_loop(),
            name="index-retry-handler",
        )

        logger.info("Index scheduler started")

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the scheduler and all components.

        Args:
            timeout: Maximum time to wait for workers to finish.
        """
        if not self._running:
            return

        logger.info("Stopping index scheduler")
        self._running = False

        # Stop retry handler
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await asyncio.wait_for(self._retry_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._retry_task = None

        # Stop file watcher
        await self._watcher.stop()

        # Stop worker pool (waits for current tasks)
        await self._pool.stop(timeout=timeout)

        logger.info("Index scheduler stopped")

    async def trigger_session(
        self,
        profile_id: str,
        session_id: str,
        indexing_path: str | None = None,
    ) -> bool:
        """Trigger indexing for a session (SDK callback).

        Called after a Claude Code task completes. Handles flush delay
        by retrying if the file is empty or hasn't been written yet.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            indexing_path: Optional folder path (auto-detected if not provided).

        Returns:
            True if enqueued immediately, False if scheduled for retry.
        """
        self._stats.sdk_callbacks += 1

        profile = get_profile(profile_id)
        if profile is None:
            logger.error(f"Unknown profile: {profile_id}")
            return False

        # Find the file in the specified or detected folder
        file_path, resolved_indexing_path = self._find_session_file(
            profile, session_id, indexing_path
        )

        if file_path is None:
            # File not found in any folder
            logger.warning(
                f"[{profile_id}] Session file not found: {session_id}"
            )
            return False

        # Check if file exists and has content
        if self._file_has_content(file_path):
            # File ready - enqueue immediately with high priority
            task = IndexTask(
                priority=TaskPriority.HIGH,
                created_at=time.time(),
                profile_id=profile_id,
                session_id=session_id,
                file_path=file_path,
                indexing_path=resolved_indexing_path,
            )
            await self._queue.put(task)
            self._stats.sdk_successes += 1
            logger.info(f"[{profile_id}] SDK trigger: {session_id} (immediate)")
            return True

        # File not ready - schedule retry
        async with self._pending_lock:
            key = f"{profile_id}:{session_id}:{resolved_indexing_path}"
            if key not in self._pending_sessions:
                self._pending_sessions[key] = PendingSession(
                    profile_id=profile_id,
                    session_id=session_id,
                    file_path=file_path,
                    indexing_path=resolved_indexing_path,
                )
                logger.info(
                    f"[{profile_id}] SDK trigger: {session_id} (scheduled for retry)"
                )

        return False

    def _find_session_file(
        self,
        profile,
        session_id: str,
        indexing_path: str | None = None,
    ) -> tuple[Path | None, str]:
        """Find session file in profile's sessions directories.

        Args:
            profile: Profile configuration.
            session_id: Session identifier.
            indexing_path: Optional explicit folder path.

        Returns:
            Tuple of (file_path, indexing_path) or (None, "") if not found.
        """
        filename = f"{session_id}.jsonl"

        if indexing_path:
            # Explicit path provided - use it directly
            file_path = Path(indexing_path) / filename
            if file_path.exists():
                return file_path, indexing_path
            # Also try in case indexing_path is just the base folder
            return file_path, indexing_path

        # Auto-detect: search all sessions directories
        for sessions_dir in profile.sessions_paths:
            file_path = sessions_dir / filename
            if file_path.exists():
                return file_path, str(sessions_dir)

        # Not found - return first path as fallback for retry
        if profile.sessions_paths:
            first_dir = profile.sessions_paths[0]
            return first_dir / filename, str(first_dir)

        return None, ""

    async def trigger_reindex(
        self,
        profile_id: str | None = None,
        session_ids: list[str] | None = None,
    ) -> int:
        """Trigger reindexing for a profile or specific sessions.

        Args:
            profile_id: Profile to reindex (None = all profiles).
            session_ids: Specific sessions to reindex (None = all).

        Returns:
            Number of sessions enqueued.
        """
        if profile_id is None:
            # Reindex all profiles
            results = await self._scanner.scan_all()
            return sum(r.total_enqueued for r in results)

        profile = get_profile(profile_id)
        if profile is None:
            logger.error(f"Unknown profile: {profile_id}")
            return 0

        if session_ids:
            # Reindex specific sessions - search all folders
            count = 0
            for session_id in session_ids:
                file_path, indexing_path = self._find_session_file(
                    profile, session_id
                )
                if file_path and file_path.exists():
                    task = IndexTask(
                        priority=TaskPriority.LOW,
                        created_at=time.time(),
                        profile_id=profile_id,
                        session_id=session_id,
                        file_path=file_path,
                        indexing_path=indexing_path,
                    )
                    if await self._queue.put(task):
                        count += 1
            return count

        # Reindex entire profile
        result = await self._scanner.scan_profile(profile)
        return result.total_enqueued

    def get_session_status(
        self,
        profile_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Get indexing status for a specific session.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.

        Returns:
            Status dict or None if not found.
        """
        from reverie.db.connection import get_connection
        from reverie.db.repositories import (
            PassageProcessingHistoryRepository,
            SessionIndexStateRepository,
        )

        with get_connection() as conn:
            state_repo = SessionIndexStateRepository(conn)
            state = state_repo.get_by_session(profile_id, session_id)

            if state is None:
                return None

            history_repo = PassageProcessingHistoryRepository(conn)
            history = history_repo.get_by_session(profile_id, session_id, limit=10)

            return {
                "profile_id": state.profile_id,
                "session_id": state.session_id,
                "session_seq": state.session_seq,
                "status": state.status,
                "last_indexed_turn": state.last_indexed_turn,
                "total_passages": state.total_passages_indexed,
                "total_dropped": state.total_passages_dropped,
                "error_message": state.error_message,
                "last_indexed_at": (
                    state.last_indexed_at.isoformat() if state.last_indexed_at else None
                ),
                "history": [
                    {
                        "turn_index": h.turn_index,
                        "action": h.action,
                        "drop_reason": h.drop_reason,
                        "processed_at": h.processed_at.isoformat(),
                    }
                    for h in history
                ],
            }

    def _file_has_content(self, file_path: Path) -> bool:
        """Check if a file exists and has content.

        Args:
            file_path: Path to check.

        Returns:
            True if file exists and has > 0 bytes.
        """
        try:
            return file_path.exists() and file_path.stat().st_size > 0
        except OSError:
            return False

    def _on_index_complete(self, result) -> None:
        """Callback when a session indexing completes.

        Args:
            result: IndexResult from the worker.
        """
        # Clear from pending if it was a retry
        # Search for matching keys (may have indexing_path suffix)
        prefix = f"{result.profile_id}:{result.session_id}:"
        keys_to_remove = [
            key for key in self._pending_sessions.keys()
            if key.startswith(prefix)
        ]
        for key in keys_to_remove:
            del self._pending_sessions[key]

    async def _retry_loop(self) -> None:
        """Background loop that retries pending sessions.

        Implements exponential backoff for flush delay handling.
        """
        logger.debug("Retry handler started")

        while self._running:
            try:
                await asyncio.sleep(0.5)  # Check every 500ms

                async with self._pending_lock:
                    keys_to_remove = []

                    for key, pending in list(self._pending_sessions.items()):
                        # Check if file now has content
                        if self._file_has_content(pending.file_path):
                            # File ready - enqueue
                            task = IndexTask(
                                priority=TaskPriority.HIGH,
                                created_at=time.time(),
                                profile_id=pending.profile_id,
                                session_id=pending.session_id,
                                file_path=pending.file_path,
                                indexing_path=pending.indexing_path,
                            )
                            await self._queue.put(task)
                            keys_to_remove.append(key)
                            self._stats.sdk_successes += 1
                            logger.info(
                                f"[{pending.profile_id}] Retry success: "
                                f"{pending.session_id} (attempt {pending.retry_count + 1})"
                            )
                            continue

                        # Check if retry is due
                        if pending.retry_count >= MAX_RETRIES:
                            # Max retries exceeded - fall back to file watcher
                            keys_to_remove.append(key)
                            self._stats.sdk_fallbacks += 1
                            logger.warning(
                                f"[{pending.profile_id}] Max retries exceeded: "
                                f"{pending.session_id}, falling back to file watcher"
                            )
                            continue

                        # Check if it's time to retry
                        delay = RETRY_DELAYS[pending.retry_count]
                        elapsed = time.time() - pending.created_at
                        total_delay = sum(RETRY_DELAYS[: pending.retry_count + 1])

                        if elapsed >= total_delay:
                            # Time to retry
                            pending.retry_count += 1
                            self._stats.sdk_retries += 1
                            logger.debug(
                                f"[{pending.profile_id}] Retry {pending.retry_count}: "
                                f"{pending.session_id}"
                            )

                    for key in keys_to_remove:
                        del self._pending_sessions[key]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retry loop error: {e}", exc_info=True)

        logger.debug("Retry handler stopped")
