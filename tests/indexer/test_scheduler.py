"""Tests for IndexScheduler."""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reverie.indexer.scheduler import (
    IndexScheduler,
    MAX_RETRIES,
    PendingSession,
    RETRY_DELAYS,
    SchedulerStats,
)
from reverie.indexer.work_queue import TaskPriority


class TestIndexScheduler:
    """Tests for IndexScheduler class."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create a temporary sessions directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler instance."""
        return IndexScheduler(
            num_workers=1,
            debounce_seconds=0.1,
        )

    @pytest.mark.asyncio
    async def test_start_stop(self, scheduler):
        """Test starting and stopping the scheduler."""
        with patch.object(scheduler._pool, "start", new_callable=AsyncMock):
            with patch.object(scheduler._watcher, "start", new_callable=AsyncMock):
                with patch.object(scheduler._scanner, "scan_all", new_callable=AsyncMock):
                    await scheduler.start(run_startup_scan=False)
                    assert scheduler.is_running

        with patch.object(scheduler._pool, "stop", new_callable=AsyncMock):
            with patch.object(scheduler._watcher, "stop", new_callable=AsyncMock):
                await scheduler.stop()
                assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_trigger_session_immediate(self, scheduler, temp_sessions_dir):
        """Test triggering a session that's ready immediately."""
        # Create a session file with content
        session_file = temp_sessions_dir / "session_123.jsonl"
        session_file.write_text('{"type": "user"}\n')

        mock_profile = MagicMock()
        mock_profile.sessions_path = temp_sessions_dir
        mock_profile.sessions_paths = [temp_sessions_dir]

        with patch(
            "reverie.indexer.scheduler.get_profile"
        ) as mock_get_profile:
            mock_get_profile.return_value = mock_profile

            result = await scheduler.trigger_session("test", "session_123")

        assert result is True  # Immediate enqueue
        assert scheduler._queue.qsize() == 1
        assert scheduler._stats.sdk_successes == 1

    @pytest.mark.asyncio
    async def test_trigger_session_retry_scheduled(self, scheduler, temp_sessions_dir):
        """Test triggering a session that needs retry (empty file)."""
        # Create an empty session file
        session_file = temp_sessions_dir / "session_123.jsonl"
        session_file.touch()

        mock_profile = MagicMock()
        mock_profile.sessions_path = temp_sessions_dir
        mock_profile.sessions_paths = [temp_sessions_dir]

        with patch(
            "reverie.indexer.scheduler.get_profile"
        ) as mock_get_profile:
            mock_get_profile.return_value = mock_profile

            result = await scheduler.trigger_session("test", "session_123")

        assert result is False  # Scheduled for retry
        # Key now includes indexing_path
        key = f"test:session_123:{temp_sessions_dir}"
        assert key in scheduler._pending_sessions
        assert scheduler._stats.sdk_callbacks == 1

    @pytest.mark.asyncio
    async def test_trigger_session_unknown_profile(self, scheduler):
        """Test triggering with unknown profile."""
        with patch(
            "reverie.indexer.scheduler.get_profile"
        ) as mock_get_profile:
            mock_get_profile.return_value = None

            result = await scheduler.trigger_session("unknown", "session_123")

        assert result is False
        assert scheduler._queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_trigger_reindex_all(self, scheduler):
        """Test triggering reindex for all profiles."""
        mock_result = MagicMock()
        mock_result.total_enqueued = 10

        with patch.object(
            scheduler._scanner,
            "scan_all",
            new_callable=AsyncMock,
            return_value=[mock_result],
        ):
            count = await scheduler.trigger_reindex()

        assert count == 10

    @pytest.mark.asyncio
    async def test_trigger_reindex_specific_profile(self, scheduler, temp_sessions_dir):
        """Test triggering reindex for specific profile."""
        mock_result = MagicMock()
        mock_result.total_enqueued = 5

        mock_profile = MagicMock()
        mock_profile.sessions_path = temp_sessions_dir
        mock_profile.sessions_paths = [temp_sessions_dir]

        with patch(
            "reverie.indexer.scheduler.get_profile"
        ) as mock_get_profile:
            mock_get_profile.return_value = mock_profile

            with patch.object(
                scheduler._scanner,
                "scan_profile",
                new_callable=AsyncMock,
                return_value=mock_result,
            ):
                count = await scheduler.trigger_reindex(profile_id="test")

        assert count == 5

    @pytest.mark.asyncio
    async def test_trigger_reindex_specific_sessions(self, scheduler, temp_sessions_dir):
        """Test triggering reindex for specific sessions."""
        # Create session files
        (temp_sessions_dir / "session_1.jsonl").write_text('{"type": "user"}\n')
        (temp_sessions_dir / "session_2.jsonl").write_text('{"type": "user"}\n')

        mock_profile = MagicMock()
        mock_profile.sessions_path = temp_sessions_dir
        mock_profile.sessions_paths = [temp_sessions_dir]

        with patch(
            "reverie.indexer.scheduler.get_profile"
        ) as mock_get_profile:
            mock_get_profile.return_value = mock_profile

            count = await scheduler.trigger_reindex(
                profile_id="test",
                session_ids=["session_1", "session_2"],
            )

        assert count == 2
        assert scheduler._queue.qsize() == 2

    @pytest.mark.asyncio
    async def test_retry_loop_succeeds_on_file_ready(self, scheduler, temp_sessions_dir):
        """Test retry loop succeeds when file becomes ready."""
        session_file = temp_sessions_dir / "session_123.jsonl"

        # Add to pending
        scheduler._pending_sessions["test:session_123"] = PendingSession(
            profile_id="test",
            session_id="session_123",
            file_path=session_file,
        )

        # Start retry loop
        scheduler._running = True
        retry_task = asyncio.create_task(scheduler._retry_loop())

        # Wait a bit, then create file with content
        await asyncio.sleep(0.1)
        session_file.write_text('{"type": "user"}\n')

        # Wait for retry loop to process
        await asyncio.sleep(0.6)

        # Stop retry loop
        scheduler._running = False
        retry_task.cancel()
        try:
            await retry_task
        except asyncio.CancelledError:
            pass

        # Session should be enqueued
        assert scheduler._queue.qsize() == 1
        assert "test:session_123" not in scheduler._pending_sessions

    @pytest.mark.asyncio
    async def test_retry_loop_fallback_after_max_retries(self, scheduler, temp_sessions_dir):
        """Test retry loop falls back after max retries."""
        session_file = temp_sessions_dir / "session_123.jsonl"
        session_file.touch()  # Empty file

        # Add to pending with max retries already done
        pending = PendingSession(
            profile_id="test",
            session_id="session_123",
            file_path=session_file,
            retry_count=MAX_RETRIES,
            created_at=time.time() - 20,  # Old enough
        )
        scheduler._pending_sessions["test:session_123"] = pending

        # Run one iteration of retry loop
        scheduler._running = True
        retry_task = asyncio.create_task(scheduler._retry_loop())

        await asyncio.sleep(0.6)

        scheduler._running = False
        retry_task.cancel()
        try:
            await retry_task
        except asyncio.CancelledError:
            pass

        # Session should be removed from pending (fallback to file watcher)
        assert "test:session_123" not in scheduler._pending_sessions
        assert scheduler._stats.sdk_fallbacks == 1

    def test_file_has_content_true(self, scheduler, temp_sessions_dir):
        """Test _file_has_content returns True for non-empty file."""
        file_path = temp_sessions_dir / "test.jsonl"
        file_path.write_text("content")

        assert scheduler._file_has_content(file_path) is True

    def test_file_has_content_empty_file(self, scheduler, temp_sessions_dir):
        """Test _file_has_content returns False for empty file."""
        file_path = temp_sessions_dir / "test.jsonl"
        file_path.touch()

        assert scheduler._file_has_content(file_path) is False

    def test_file_has_content_missing_file(self, scheduler):
        """Test _file_has_content returns False for missing file."""
        file_path = Path("/nonexistent/file.jsonl")

        assert scheduler._file_has_content(file_path) is False

    def test_stats(self, scheduler):
        """Test stats property returns all components."""
        stats = scheduler.stats

        assert "running" in stats
        assert "queue" in stats
        assert "pool" in stats
        assert "watcher" in stats
        assert "sdk" in stats
        assert "callbacks" in stats["sdk"]
        assert "retries" in stats["sdk"]
        assert "fallbacks" in stats["sdk"]
        assert "pending" in stats["sdk"]


class TestRetryDelays:
    """Tests for retry delay configuration."""

    def test_retry_delays_count(self):
        """Test retry delays has expected count."""
        assert len(RETRY_DELAYS) == 5

    def test_retry_delays_exponential(self):
        """Test retry delays follow exponential pattern."""
        expected = [0.5, 1.0, 2.0, 4.0, 8.0]
        assert RETRY_DELAYS == expected

    def test_max_retries_matches(self):
        """Test MAX_RETRIES matches RETRY_DELAYS length."""
        assert MAX_RETRIES == len(RETRY_DELAYS)


class TestPendingSession:
    """Tests for PendingSession dataclass."""

    def test_default_values(self):
        """Test default values."""
        pending = PendingSession(
            profile_id="test",
            session_id="session_123",
            file_path=Path("/path/to/file.jsonl"),
        )

        assert pending.retry_count == 0
        assert pending.created_at > 0


class TestSchedulerStats:
    """Tests for SchedulerStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = SchedulerStats()

        assert stats.sdk_callbacks == 0
        assert stats.sdk_retries == 0
        assert stats.sdk_fallbacks == 0
        assert stats.sdk_successes == 0


class TestMultiFolderScheduler:
    """Tests for multi-folder indexing support in scheduler."""

    @pytest.fixture
    def temp_multi_dirs(self):
        """Create multiple temporary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project1 = Path(tmpdir) / "project1"
            project2 = Path(tmpdir) / "project2"
            project1.mkdir()
            project2.mkdir()
            yield (project1, project2)

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler instance."""
        return IndexScheduler(
            num_workers=1,
            debounce_seconds=0.1,
        )

    @pytest.mark.asyncio
    async def test_trigger_session_with_indexing_path(
        self, scheduler, temp_multi_dirs
    ):
        """Test trigger_session with explicit indexing_path."""
        project1, project2 = temp_multi_dirs

        # Create same session in both projects
        (project1 / "session_123.jsonl").write_text('{"type": "user"}\n')
        (project2 / "session_123.jsonl").write_text('{"type": "user"}\n')

        mock_profile = MagicMock()
        mock_profile.sessions_path = project1
        mock_profile.sessions_paths = [project1, project2]

        with patch(
            "reverie.indexer.scheduler.get_profile"
        ) as mock_get_profile:
            mock_get_profile.return_value = mock_profile

            # Trigger for project1
            result1 = await scheduler.trigger_session(
                "test", "session_123", indexing_path=str(project1)
            )
            # Trigger for project2
            result2 = await scheduler.trigger_session(
                "test", "session_123", indexing_path=str(project2)
            )

        assert result1 is True
        assert result2 is True
        # Both should be enqueued (different indexing_path)
        assert scheduler._queue.qsize() == 2

    @pytest.mark.asyncio
    async def test_trigger_session_auto_detect_folder(
        self, scheduler, temp_multi_dirs
    ):
        """Test trigger_session auto-detects folder when indexing_path not provided."""
        project1, project2 = temp_multi_dirs

        # Create session only in project2
        (project2 / "session_456.jsonl").write_text('{"type": "user"}\n')

        mock_profile = MagicMock()
        mock_profile.sessions_path = project1
        mock_profile.sessions_paths = [project1, project2]

        with patch(
            "reverie.indexer.scheduler.get_profile"
        ) as mock_get_profile:
            mock_get_profile.return_value = mock_profile

            # Trigger without explicit path - should find in project2
            result = await scheduler.trigger_session("test", "session_456")

        assert result is True
        assert scheduler._queue.qsize() == 1

        # Verify the task has correct indexing_path
        task = await scheduler._queue.get()
        assert task.indexing_path == str(project2)
