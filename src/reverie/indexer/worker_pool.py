"""Multi-worker indexing pool for parallel session processing.

Manages multiple worker tasks that consume from WorkQueue
and process sessions independently.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable

from reverie.config import get_settings
from reverie.indexer.incremental import IncrementalIndexer, IndexResult
from reverie.indexer.work_queue import IndexTask, WorkQueue
from reverie.mcp.server import set_indexer_state

logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    """Statistics for a single worker."""
    worker_id: int
    sessions_processed: int = 0
    passages_indexed: int = 0
    passages_dropped: int = 0
    errors: int = 0
    total_elapsed_ms: float = 0.0


class IndexWorkerPool:
    """Multi-worker pool for parallel indexing.

    Features:
    - Configurable number of workers (REVERIE_INDEX_WORKERS)
    - Independent session processing (no contention)
    - Graceful start/stop
    - Per-worker statistics
    - Optional callback on completion

    Usage:
        queue = WorkQueue()
        pool = IndexWorkerPool(queue, num_workers=3)

        # Start workers
        await pool.start()

        # Add tasks to queue
        await queue.put(IndexTask(...))

        # Stop workers gracefully
        await pool.stop()
    """

    def __init__(
        self,
        queue: WorkQueue,
        num_workers: int | None = None,
        on_complete: Callable[[IndexResult], None] | None = None,
    ):
        """Initialize the worker pool.

        Args:
            queue: Work queue to consume tasks from.
            num_workers: Number of workers (default from config).
            on_complete: Optional callback when a session is indexed.
        """
        settings = get_settings()
        self._queue = queue
        self._num_workers = num_workers or settings.index_workers
        self._on_complete = on_complete
        self._workers: list[asyncio.Task] = []
        self._worker_stats: dict[int, WorkerStats] = {}
        self._current_sessions: dict[int, str] = {}  # worker_id -> session_id
        self._running = False
        self._shutdown_event = asyncio.Event()

    @property
    def is_running(self) -> bool:
        """Check if pool is running."""
        return self._running

    @property
    def num_workers(self) -> int:
        """Get number of workers."""
        return self._num_workers

    @property
    def stats(self) -> dict:
        """Get aggregated pool statistics."""
        total_sessions = sum(s.sessions_processed for s in self._worker_stats.values())
        total_passages = sum(s.passages_indexed for s in self._worker_stats.values())
        total_dropped = sum(s.passages_dropped for s in self._worker_stats.values())
        total_errors = sum(s.errors for s in self._worker_stats.values())
        total_elapsed = sum(s.total_elapsed_ms for s in self._worker_stats.values())

        return {
            "running": self._running,
            "num_workers": self._num_workers,
            "active_workers": len([w for w in self._workers if not w.done()]),
            "queue_size": self._queue.qsize(),
            "active_sessions": list(self._current_sessions.values()),
            "total_sessions_processed": total_sessions,
            "total_passages_indexed": total_passages,
            "total_passages_dropped": total_dropped,
            "total_errors": total_errors,
            "total_elapsed_ms": total_elapsed,
            "workers": {
                w_id: {
                    "sessions_processed": s.sessions_processed,
                    "passages_indexed": s.passages_indexed,
                    "passages_dropped": s.passages_dropped,
                    "errors": s.errors,
                }
                for w_id, s in self._worker_stats.items()
            },
        }

    async def start(self) -> None:
        """Start all workers."""
        if self._running:
            logger.warning("Worker pool already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self._worker_stats.clear()

        logger.info(f"Starting {self._num_workers} index workers")

        for i in range(self._num_workers):
            self._worker_stats[i] = WorkerStats(worker_id=i)
            worker = asyncio.create_task(
                self._worker_loop(i),
                name=f"index-worker-{i}",
            )
            self._workers.append(worker)

        logger.info(f"Started {self._num_workers} index workers")

        # Report initial state
        set_indexer_state(workers=self._num_workers)

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop all workers gracefully.

        Args:
            timeout: Maximum time to wait for workers to finish.
        """
        if not self._running:
            return

        logger.info("Stopping worker pool...")
        self._running = False
        self._shutdown_event.set()

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("Worker pool shutdown timed out")

        self._workers.clear()
        logger.info("Worker pool stopped")

    async def _worker_loop(self, worker_id: int) -> None:
        """Main loop for a single worker.

        Continuously pulls tasks from queue and processes them.

        Args:
            worker_id: Unique identifier for this worker.
        """
        indexer = IncrementalIndexer()
        stats = self._worker_stats[worker_id]

        logger.debug(f"[Worker-{worker_id}] Started")

        while self._running:
            try:
                # Wait for task with timeout (to check shutdown)
                try:
                    task = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    # Check if we should stop
                    if self._shutdown_event.is_set():
                        break
                    continue

                logger.info(
                    f"[Worker-{worker_id}] Processing {task.profile_id}:{task.session_id}"
                )

                # Track current session for this worker
                self._current_sessions[worker_id] = task.session_id

                # Report indexer state
                set_indexer_state(
                    running=True,
                    current_session=task.session_id,
                    current_turn=0,
                    queue_size=self._queue.qsize(),
                )

                start_time = time.time()

                # Run indexing in thread pool (it's sync)
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: indexer.index_session(
                        task.profile_id,
                        task.session_id,
                        task.file_path,
                        indexing_path=task.indexing_path,
                    ),
                )

                elapsed = (time.time() - start_time) * 1000

                # Update stats
                stats.sessions_processed += 1
                stats.passages_indexed += result.new_passages
                stats.passages_dropped += result.dropped
                stats.total_elapsed_ms += elapsed

                if not result.success:
                    stats.errors += 1
                    logger.error(
                        f"[Worker-{worker_id}] Failed {task.session_id}: {result.error}"
                    )
                else:
                    logger.info(
                        f"[Worker-{worker_id}] Completed {task.session_id}: "
                        f"indexed={result.new_passages}, dropped={result.dropped} "
                        f"({elapsed:.0f}ms)"
                    )

                # Mark task as done (removes from pending, allows re-enqueue)
                await self._queue.mark_done(task)

                # Clear current session for this worker
                self._current_sessions.pop(worker_id, None)

                # Update indexer state (queue size changed)
                queue_size = self._queue.qsize()
                if queue_size == 0:
                    # No more tasks - mark as idle
                    set_indexer_state(
                        running=False,
                        current_session=None,
                        current_turn=None,
                        queue_size=0,
                    )
                else:
                    set_indexer_state(queue_size=queue_size)

                # Call completion callback if provided
                if self._on_complete:
                    try:
                        self._on_complete(result)
                    except Exception as e:
                        logger.error(f"[Worker-{worker_id}] Callback error: {e}")

            except asyncio.CancelledError:
                logger.debug(f"[Worker-{worker_id}] Cancelled")
                break
            except Exception as e:
                logger.error(f"[Worker-{worker_id}] Unexpected error: {e}", exc_info=True)
                stats.errors += 1
                # Continue processing other tasks

        logger.debug(f"[Worker-{worker_id}] Stopped")

    async def wait_until_empty(self, timeout: float | None = None) -> bool:
        """Wait until queue is empty and all tasks are done.

        Args:
            timeout: Maximum time to wait (None = forever).

        Returns:
            True if queue emptied, False if timed out.
        """
        try:
            await asyncio.wait_for(
                self._queue.join(),
                timeout=timeout,
            )
            return True
        except asyncio.TimeoutError:
            return False
