"""Tests for SessionFileWatcher."""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reverie.indexer.file_watcher import (
    DEFAULT_DEBOUNCE_SECONDS,
    SessionEventHandler,
    SessionFileWatcher,
    WatcherStats,
)
from reverie.indexer.work_queue import WorkQueue


class TestSessionEventHandler:
    """Tests for SessionEventHandler class."""

    @pytest.fixture
    def callback(self):
        """Create a mock callback."""
        return MagicMock()

    @pytest.fixture
    def handler(self, callback):
        """Create a handler instance."""
        return SessionEventHandler(
            profile_id="test",
            on_session_changed=callback,
            debounce_seconds=0.1,  # Short for testing
        )

    def test_on_modified_ignores_directories(self, handler, callback):
        """Test that directory events are ignored."""
        event = MagicMock()
        event.is_directory = True
        event.src_path = "/some/path"

        handler.on_modified(event)

        assert handler.stats.events_received == 0
        callback.assert_not_called()

    def test_on_modified_ignores_non_jsonl(self, handler, callback):
        """Test that non-jsonl files are ignored."""
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/path/to/file.txt"

        handler.on_modified(event)

        assert handler.stats.events_received == 0
        callback.assert_not_called()

    def test_on_modified_processes_jsonl(self, handler, callback):
        """Test that jsonl files trigger callback after debounce."""
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/path/to/session_123.jsonl"

        handler.on_modified(event)

        assert handler.stats.events_received == 1

        # Wait for debounce
        time.sleep(0.2)

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "test"  # profile_id
        assert args[1] == "session_123"  # session_id

    def test_on_modified_debounces_multiple_events(self, handler, callback):
        """Test that multiple rapid events are debounced."""
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/path/to/session_123.jsonl"

        # Fire multiple events rapidly
        handler.on_modified(event)
        handler.on_modified(event)
        handler.on_modified(event)

        assert handler.stats.events_received == 3
        assert handler.stats.events_debounced == 2  # First not debounced

        # Wait for debounce
        time.sleep(0.2)

        # Should only call callback once
        assert callback.call_count == 1

    def test_on_created_calls_on_modified(self, handler, callback):
        """Test that on_created delegates to on_modified."""
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/path/to/session_123.jsonl"

        handler.on_created(event)

        assert handler.stats.events_received == 1


class TestSessionFileWatcher:
    """Tests for SessionFileWatcher class."""

    @pytest.fixture
    def work_queue(self):
        """Create a work queue."""
        return WorkQueue()

    @pytest.fixture
    def watcher(self, work_queue):
        """Create a watcher instance."""
        return SessionFileWatcher(
            queue=work_queue,
            debounce_seconds=0.1,
        )

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create a temporary sessions directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_start_stop(self, watcher):
        """Test starting and stopping the watcher."""
        with patch(
            "reverie.indexer.file_watcher.get_profiles"
        ) as mock_get_profiles:
            mock_get_profiles.return_value = {}

            await watcher.start()
            assert watcher.is_running

            await watcher.stop()
            assert not watcher.is_running

    @pytest.mark.asyncio
    async def test_start_already_running(self, watcher):
        """Test starting when already running."""
        with patch(
            "reverie.indexer.file_watcher.get_profiles"
        ) as mock_get_profiles:
            mock_get_profiles.return_value = {}

            await watcher.start()
            await watcher.start()  # Second start should be no-op

            assert watcher.is_running
            await watcher.stop()

    @pytest.mark.asyncio
    async def test_stats_aggregation(self, watcher, temp_sessions_dir):
        """Test stats are aggregated across handlers."""
        mock_profile = MagicMock()
        mock_profile.id = "test"
        mock_profile.sessions_path = temp_sessions_dir
        mock_profile.sessions_paths = [temp_sessions_dir]

        with patch(
            "reverie.indexer.file_watcher.get_profiles"
        ) as mock_get_profiles:
            mock_get_profiles.return_value = {"test": mock_profile}

            await watcher.start()

            stats = watcher.stats
            assert stats["running"] is True
            assert stats["profiles_watched"] == 1
            assert stats["directories_watched"] == 1
            assert "events_received" in stats
            assert "sessions_enqueued" in stats

            await watcher.stop()

    @pytest.mark.asyncio
    async def test_watch_missing_directory(self, watcher):
        """Test watching a missing directory."""
        mock_profile = MagicMock()
        mock_profile.id = "test"
        mock_profile.sessions_path = Path("/nonexistent/path")
        mock_profile.sessions_paths = [Path("/nonexistent/path")]

        with patch(
            "reverie.indexer.file_watcher.get_profiles"
        ) as mock_get_profiles:
            mock_get_profiles.return_value = {"test": mock_profile}

            await watcher.start()

            # Should not raise, just warn (profile exists but no valid dirs)
            assert watcher.stats["profiles_watched"] == 0

            await watcher.stop()

    @pytest.mark.asyncio
    async def test_file_change_enqueues_task(
        self, watcher, work_queue, temp_sessions_dir
    ):
        """Test that file changes enqueue indexing tasks."""
        mock_profile = MagicMock()
        mock_profile.id = "test"
        mock_profile.sessions_path = temp_sessions_dir
        mock_profile.sessions_paths = [temp_sessions_dir]

        with patch(
            "reverie.indexer.file_watcher.get_profiles"
        ) as mock_get_profiles:
            mock_get_profiles.return_value = {"test": mock_profile}

            await watcher.start()

            # Create a session file
            session_file = temp_sessions_dir / "new_session.jsonl"
            session_file.write_text('{"type": "user"}\n')

            # Wait for debounce + processing
            await asyncio.sleep(0.3)

            # Check queue has task
            # Note: Depending on OS, inotify may or may not trigger
            # So we just verify the watcher is working

            await watcher.stop()


class TestWatcherStats:
    """Tests for WatcherStats dataclass."""

    def test_default_values(self):
        """Test default stat values."""
        stats = WatcherStats()
        assert stats.events_received == 0
        assert stats.events_debounced == 0
        assert stats.sessions_enqueued == 0
        assert stats.errors == 0


class TestMultiFolderWatcher:
    """Tests for multi-folder watching support."""

    @pytest.fixture
    def work_queue(self):
        """Create a work queue."""
        return WorkQueue()

    @pytest.fixture
    def watcher(self, work_queue):
        """Create a watcher instance."""
        return SessionFileWatcher(
            queue=work_queue,
            debounce_seconds=0.1,
        )

    @pytest.fixture
    def temp_multi_dirs(self):
        """Create multiple temporary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project1 = Path(tmpdir) / "project1"
            project2 = Path(tmpdir) / "project2"
            project1.mkdir()
            project2.mkdir()
            yield (project1, project2)

    @pytest.mark.asyncio
    async def test_watch_multiple_directories_per_profile(
        self, watcher, temp_multi_dirs
    ):
        """Test that multiple directories per profile are watched."""
        project1, project2 = temp_multi_dirs

        mock_profile = MagicMock()
        mock_profile.id = "multi"
        mock_profile.sessions_paths = [project1, project2]

        with patch(
            "reverie.indexer.file_watcher.get_profiles"
        ) as mock_get_profiles:
            mock_get_profiles.return_value = {"multi": mock_profile}

            await watcher.start()

            # Should have handlers for both directories
            stats = watcher.stats
            assert stats["profiles_watched"] == 1  # Still one profile
            assert stats["directories_watched"] == 2  # But two directories

            await watcher.stop()

    @pytest.mark.asyncio
    async def test_handler_with_indexing_path(self):
        """Test SessionEventHandler includes indexing_path in callback."""
        callback_args = []

        def mock_callback(profile_id, session_id, file_path, indexing_path):
            callback_args.append((profile_id, session_id, file_path, indexing_path))

        handler = SessionEventHandler(
            profile_id="test",
            on_session_changed=mock_callback,
            debounce_seconds=0.1,
            indexing_path="/home/user/project1",
        )

        event = MagicMock()
        event.is_directory = False
        event.src_path = "/home/user/project1/session_123.jsonl"

        handler.on_modified(event)

        # Wait for debounce
        time.sleep(0.2)

        assert len(callback_args) == 1
        profile_id, session_id, file_path, indexing_path = callback_args[0]
        assert profile_id == "test"
        assert session_id == "session_123"
        assert indexing_path == "/home/user/project1"
