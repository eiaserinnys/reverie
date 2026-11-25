"""Work queue for incremental indexing with deduplication.

Provides a priority queue with built-in deduplication to prevent
processing the same session multiple times concurrently.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TaskPriority(IntEnum):
    """Priority levels for indexing tasks.

    Lower values = higher priority (processed first).
    """
    HIGH = 1      # SDK callback - just completed task
    NORMAL = 2    # File watcher - detected change
    LOW = 3       # Startup scanner - batch processing


@dataclass(order=True)
class IndexTask:
    """Task for indexing a session file.

    Ordered by priority first, then by creation time.
    """
    priority: int
    created_at: float = field(compare=True)
    profile_id: str = field(compare=False)
    session_id: str = field(compare=False)
    file_path: Path = field(compare=False)
    indexing_path: str = field(default="", compare=False)

    def key(self) -> str:
        """Get unique key for deduplication.

        Includes indexing_path to allow same session from different folders.
        """
        return f"{self.profile_id}:{self.session_id}:{self.indexing_path}"


class WorkQueue:
    """Async work queue with deduplication and priority support.

    Features:
    - Priority queue: HIGH > NORMAL > LOW
    - Deduplication: Same session won't be queued twice
    - Thread-safe: Uses asyncio.Lock for concurrent access
    - Non-blocking: put() returns immediately if already queued

    Usage:
        queue = WorkQueue()

        # Add tasks
        await queue.put(IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="my_profile",
            session_id="abc123",
            file_path=Path("/path/to/session.jsonl")
        ))

        # Get tasks (blocks until available)
        task = await queue.get()
    """

    def __init__(self, maxsize: int = 0):
        """Initialize the work queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited).
        """
        self._queue: asyncio.PriorityQueue[IndexTask] = asyncio.PriorityQueue(
            maxsize=maxsize
        )
        self._pending: set[str] = set()
        self._lock = asyncio.Lock()
        self._stats = {
            "enqueued": 0,
            "dequeued": 0,
            "duplicates_skipped": 0,
        }

    async def put(self, task: IndexTask) -> bool:
        """Add a task to the queue if not already pending.

        Args:
            task: The indexing task to add.

        Returns:
            True if task was added, False if duplicate (already pending).
        """
        async with self._lock:
            key = task.key()
            if key in self._pending:
                self._stats["duplicates_skipped"] += 1
                logger.debug(f"Skipping duplicate task: {key}")
                return False
            self._pending.add(key)

        await self._queue.put(task)
        self._stats["enqueued"] += 1
        logger.debug(f"Enqueued task: {task.key()} (priority={task.priority})")
        return True

    async def put_nowait(self, task: IndexTask) -> bool:
        """Add a task without blocking (raises if full).

        Args:
            task: The indexing task to add.

        Returns:
            True if task was added, False if duplicate.

        Raises:
            asyncio.QueueFull: If queue is full and maxsize > 0.
        """
        async with self._lock:
            key = task.key()
            if key in self._pending:
                self._stats["duplicates_skipped"] += 1
                return False
            self._pending.add(key)

        self._queue.put_nowait(task)
        self._stats["enqueued"] += 1
        return True

    async def get(self) -> IndexTask:
        """Get the next task from the queue (blocks if empty).

        Note: Task remains in _pending until mark_done() is called.
        This prevents duplicate enqueuing while processing.

        Returns:
            The highest priority task.
        """
        task = await self._queue.get()
        # Keep in _pending - will be removed in mark_done()
        self._stats["dequeued"] += 1
        logger.debug(f"Dequeued task: {task.key()}")
        return task

    async def mark_done(self, task: IndexTask) -> None:
        """Mark a task as fully completed, allowing re-enqueue.

        Call this after indexing is complete (success or failure).

        Args:
            task: The completed task.
        """
        async with self._lock:
            self._pending.discard(task.key())
        self._queue.task_done()

    def get_nowait(self) -> IndexTask:
        """Get a task without blocking (raises if empty).

        Note: Task remains in _pending until mark_done() is called.

        Returns:
            The highest priority task.

        Raises:
            asyncio.QueueEmpty: If queue is empty.
        """
        task = self._queue.get_nowait()
        # Keep in _pending - will be removed in mark_done()
        self._stats["dequeued"] += 1
        return task

    async def join(self) -> None:
        """Wait until all tasks are processed."""
        await self._queue.join()

    def qsize(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    def is_pending(
        self, profile_id: str, session_id: str, indexing_path: str = ""
    ) -> bool:
        """Check if a session is currently pending in the queue.

        Args:
            profile_id: Profile identifier.
            session_id: Session identifier.
            indexing_path: Folder path being indexed (default: empty string).

        Returns:
            True if session is pending, False otherwise.
        """
        key = f"{profile_id}:{session_id}:{indexing_path}"
        return key in self._pending

    @property
    def stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            **self._stats,
            "current_size": self.qsize(),
            "pending_count": len(self._pending),
        }

    def clear(self) -> int:
        """Clear all pending tasks.

        Returns:
            Number of tasks cleared.
        """
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        self._pending.clear()
        return count
