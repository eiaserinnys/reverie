"""File watcher for incremental indexing.

Uses watchdog to monitor session directories for file changes
and enqueues modified sessions for indexing.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from reverie.config import get_profiles, ProfileConfig
from reverie.indexer.work_queue import IndexTask, TaskPriority, WorkQueue

logger = logging.getLogger(__name__)


# Default debounce delay in seconds
DEFAULT_DEBOUNCE_SECONDS = 2.0


@dataclass
class WatcherStats:
    """Statistics for the file watcher."""
    events_received: int = 0
    events_debounced: int = 0
    sessions_enqueued: int = 0
    errors: int = 0


class SessionEventHandler(FileSystemEventHandler):
    """Handles file system events for session files.

    Implements debouncing to prevent excessive triggering during
    active conversation (file is modified frequently).
    """

    def __init__(
        self,
        profile_id: str,
        on_session_changed: Callable[[str, str, Path, str], None],
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        indexing_path: str = "",
    ):
        """Initialize the handler.

        Args:
            profile_id: Profile identifier for this watched directory.
            on_session_changed: Callback when session file changes.
                Signature: (profile_id, session_id, file_path, indexing_path) -> None
            debounce_seconds: Delay before processing change event.
            indexing_path: Folder path being watched (for multi-folder support).
        """
        super().__init__()
        self._profile_id = profile_id
        self._on_session_changed = on_session_changed
        self._debounce_seconds = debounce_seconds
        self._indexing_path = indexing_path
        self._pending: dict[str, float] = {}  # session_id -> last_event_time
        self._lock = asyncio.Lock()
        self._stats = WatcherStats()

    @property
    def stats(self) -> WatcherStats:
        """Get handler statistics."""
        return self._stats

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification event.

        Only processes .jsonl files, ignores directories.
        """
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Only watch .jsonl session files
        if file_path.suffix != ".jsonl":
            return

        session_id = file_path.stem
        self._stats.events_received += 1

        # Schedule debounced processing
        self._schedule_process(session_id, file_path)

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation event.

        Treats creation the same as modification.
        """
        self.on_modified(event)

    def _schedule_process(self, session_id: str, file_path: Path) -> None:
        """Schedule debounced processing of a session change.

        If called multiple times within debounce_seconds, only the
        last call triggers processing.

        Args:
            session_id: Session identifier.
            file_path: Path to session file.
        """
        current_time = time.time()
        previous_time = self._pending.get(session_id)

        # Update pending time
        self._pending[session_id] = current_time

        if previous_time is not None:
            # Already scheduled - will be handled by existing timer
            self._stats.events_debounced += 1
            return

        # Start new debounce timer in background thread
        def debounce_callback():
            time.sleep(self._debounce_seconds)

            # Check if this is still the latest event
            scheduled_time = self._pending.get(session_id)
            if scheduled_time is None:
                return  # Already processed

            # Check if a newer event occurred
            if scheduled_time > current_time + 0.1:  # Allow 100ms tolerance
                # Newer event will handle this
                return

            # Clear pending and trigger callback
            del self._pending[session_id]

            try:
                self._on_session_changed(
                    self._profile_id, session_id, file_path, self._indexing_path
                )
                self._stats.sessions_enqueued += 1
            except Exception as e:
                logger.error(
                    f"[{self._profile_id}] Error processing {session_id}: {e}"
                )
                self._stats.errors += 1

        # Run debounce timer in background
        thread = Thread(target=debounce_callback, daemon=True)
        thread.start()


class SessionFileWatcher:
    """Watches session directories for file changes.

    Uses watchdog's Observer to monitor multiple directories
    (one per profile) and enqueues changed sessions for indexing.

    Usage:
        queue = WorkQueue()
        watcher = SessionFileWatcher(queue)

        await watcher.start()
        # ... watcher runs in background ...
        await watcher.stop()
    """

    def __init__(
        self,
        queue: WorkQueue,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
    ):
        """Initialize the file watcher.

        Args:
            queue: Work queue to enqueue indexing tasks.
            debounce_seconds: Delay before processing change event.
        """
        self._queue = queue
        self._debounce_seconds = debounce_seconds
        self._observer: Observer | None = None
        # handlers: profile_id -> list of (indexing_path, handler)
        self._handlers: dict[str, list[tuple[str, SessionEventHandler]]] = {}
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    @property
    def stats(self) -> dict:
        """Get aggregated watcher statistics."""
        all_handlers = [
            handler
            for handlers_list in self._handlers.values()
            for (_, handler) in handlers_list
        ]
        total_events = sum(h.stats.events_received for h in all_handlers)
        total_debounced = sum(h.stats.events_debounced for h in all_handlers)
        total_enqueued = sum(h.stats.sessions_enqueued for h in all_handlers)
        total_errors = sum(h.stats.errors for h in all_handlers)

        return {
            "running": self._running,
            "profiles_watched": len(self._handlers),
            "directories_watched": len(all_handlers),
            "events_received": total_events,
            "events_debounced": total_debounced,
            "sessions_enqueued": total_enqueued,
            "errors": total_errors,
        }

    async def start(self) -> None:
        """Start watching all profile session directories."""
        if self._running:
            logger.warning("File watcher already running")
            return

        self._loop = asyncio.get_event_loop()
        self._observer = Observer()
        profiles = get_profiles()

        for profile_id, profile in profiles.items():
            try:
                self._watch_profile(profile)
            except Exception as e:
                logger.error(f"Failed to watch profile {profile_id}: {e}")

        self._observer.start()
        self._running = True

        logger.info(f"File watcher started, watching {len(self._handlers)} profiles")

    async def stop(self) -> None:
        """Stop watching all directories."""
        if not self._running:
            return

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        self._handlers.clear()
        self._running = False

        logger.info("File watcher stopped")

    def _watch_profile(self, profile: ProfileConfig) -> None:
        """Set up watching for a single profile.

        Supports multiple sessions directories per profile.

        Args:
            profile: Profile configuration.
        """
        sessions_dirs = profile.sessions_paths

        if not sessions_dirs:
            logger.warning(f"[{profile.id}] No sessions directories configured")
            return

        handlers_list: list[tuple[str, SessionEventHandler]] = []

        for sessions_dir in sessions_dirs:
            if not sessions_dir.exists():
                logger.warning(
                    f"[{profile.id}] Sessions directory not found: {sessions_dir}"
                )
                continue

            indexing_path = str(sessions_dir)
            handler = SessionEventHandler(
                profile_id=profile.id,
                on_session_changed=self._on_session_changed,
                debounce_seconds=self._debounce_seconds,
                indexing_path=indexing_path,
            )

            self._observer.schedule(
                handler,
                str(sessions_dir),
                recursive=False,  # Don't watch subdirectories
            )

            handlers_list.append((indexing_path, handler))

            logger.info(f"[{profile.id}] Watching directory: {sessions_dir}")

        if handlers_list:
            self._handlers[profile.id] = handlers_list

    def _on_session_changed(
        self,
        profile_id: str,
        session_id: str,
        file_path: Path,
        indexing_path: str = "",
    ) -> None:
        """Callback when a session file changes.

        Enqueues the session for indexing.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            file_path: Path to session file.
            indexing_path: Folder path being indexed (for multi-folder support).
        """
        task = IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id=profile_id,
            session_id=session_id,
            file_path=file_path,
            indexing_path=indexing_path,
        )

        # Use thread-safe put (runs in watchdog thread)
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._queue.put(task),
                self._loop,
            )
            logger.debug(f"[{profile_id}] Enqueued changed session: {session_id} ({indexing_path})")
