"""Tests for StartupScanner."""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reverie.indexer.startup_scanner import StartupScanner, ScanResult
from reverie.indexer.work_queue import WorkQueue


class TestStartupScanner:
    """Tests for StartupScanner class."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create a temporary directory with mock session files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir)

            # Create some mock session files
            (sessions_dir / "session_1.jsonl").write_text('{"type": "user"}\n')
            (sessions_dir / "session_2.jsonl").write_text('{"type": "user"}\n')
            (sessions_dir / "session_3.jsonl").write_text('{"type": "user"}\n')

            yield sessions_dir

    @pytest.fixture
    def mock_profile(self, temp_sessions_dir):
        """Create a mock profile config."""
        profile = MagicMock()
        profile.id = "test_profile"
        profile.name = "Test Profile"
        profile.sessions_path = temp_sessions_dir
        # Multi-folder support: sessions_paths returns list
        profile.sessions_paths = [temp_sessions_dir]
        return profile

    @pytest.fixture
    def work_queue(self):
        """Create a work queue."""
        return WorkQueue()

    @pytest.fixture
    def scanner(self, work_queue):
        """Create a StartupScanner instance."""
        return StartupScanner(work_queue)

    @pytest.mark.asyncio
    async def test_scan_profile_new_sessions(self, scanner, mock_profile, work_queue):
        """Test scanning finds new sessions when DB is empty."""
        # Mock DB to return no existing states
        with patch(
            "reverie.indexer.startup_scanner.get_connection"
        ) as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor

            # Mock SessionIndexStateRepository
            with patch(
                "reverie.indexer.startup_scanner.SessionIndexStateRepository"
            ) as mock_repo_class:
                mock_repo = MagicMock()
                mock_repo.get_by_profile.return_value = []
                mock_repo_class.return_value = mock_repo

                result = await scanner.scan_profile(mock_profile)

        assert result.profile_id == "test_profile"
        assert result.new_sessions == 3
        assert result.changed_sessions == 0
        assert result.failed_sessions == 0
        assert result.unchanged_sessions == 0
        assert work_queue.qsize() == 3

    @pytest.mark.asyncio
    async def test_scan_profile_changed_sessions(
        self, scanner, mock_profile, work_queue, temp_sessions_dir
    ):
        """Test scanning detects changed sessions."""
        # Create mock existing state with old mtime
        old_mtime = datetime.now(timezone.utc) - timedelta(hours=1)

        mock_state = MagicMock()
        mock_state.session_id = "session_1"
        mock_state.file_mtime = old_mtime
        mock_state.status = "completed"

        with patch(
            "reverie.indexer.startup_scanner.get_connection"
        ) as mock_conn:
            mock_conn.return_value.__enter__.return_value = MagicMock()

            with patch(
                "reverie.indexer.startup_scanner.SessionIndexStateRepository"
            ) as mock_repo_class:
                mock_repo = MagicMock()
                mock_repo.get_by_profile.return_value = [mock_state]
                mock_repo_class.return_value = mock_repo

                result = await scanner.scan_profile(mock_profile)

        # session_1 should be changed, session_2 and session_3 are new
        assert result.new_sessions == 2
        assert result.changed_sessions == 1

    @pytest.mark.asyncio
    async def test_scan_profile_failed_sessions(
        self, scanner, mock_profile, work_queue
    ):
        """Test scanning retries failed sessions."""
        mock_state = MagicMock()
        mock_state.session_id = "session_1"
        mock_state.file_mtime = datetime.now(timezone.utc)
        mock_state.status = "failed"

        with patch(
            "reverie.indexer.startup_scanner.get_connection"
        ) as mock_conn:
            mock_conn.return_value.__enter__.return_value = MagicMock()

            with patch(
                "reverie.indexer.startup_scanner.SessionIndexStateRepository"
            ) as mock_repo_class:
                mock_repo = MagicMock()
                mock_repo.get_by_profile.return_value = [mock_state]
                mock_repo_class.return_value = mock_repo

                result = await scanner.scan_profile(mock_profile)

        assert result.failed_sessions == 1
        assert result.new_sessions == 2

    @pytest.mark.asyncio
    async def test_scan_profile_unchanged_sessions(
        self, scanner, mock_profile, work_queue, temp_sessions_dir
    ):
        """Test scanning skips unchanged sessions."""
        # Get actual file mtime
        file_path = temp_sessions_dir / "session_1.jsonl"
        file_mtime = datetime.fromtimestamp(
            file_path.stat().st_mtime, tz=timezone.utc
        )

        # Create mock state with same mtime
        mock_state = MagicMock()
        mock_state.session_id = "session_1"
        mock_state.file_mtime = file_mtime + timedelta(seconds=1)  # Slightly newer
        mock_state.status = "completed"

        with patch(
            "reverie.indexer.startup_scanner.get_connection"
        ) as mock_conn:
            mock_conn.return_value.__enter__.return_value = MagicMock()

            with patch(
                "reverie.indexer.startup_scanner.SessionIndexStateRepository"
            ) as mock_repo_class:
                mock_repo = MagicMock()
                mock_repo.get_by_profile.return_value = [mock_state]
                mock_repo_class.return_value = mock_repo

                result = await scanner.scan_profile(mock_profile)

        assert result.unchanged_sessions == 1
        assert result.new_sessions == 2

    @pytest.mark.asyncio
    async def test_scan_profile_missing_directory(self, scanner, work_queue):
        """Test scanning handles missing directory gracefully."""
        mock_profile = MagicMock()
        mock_profile.id = "test"
        mock_profile.sessions_path = Path("/nonexistent/path")

        result = await scanner.scan_profile(mock_profile)

        assert result.new_sessions == 0
        assert result.errors == 0
        assert work_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_scan_all(self, scanner, mock_profile):
        """Test scanning all profiles."""
        with patch(
            "reverie.indexer.startup_scanner.get_profiles"
        ) as mock_get_profiles:
            mock_get_profiles.return_value = {"test": mock_profile}

            with patch(
                "reverie.indexer.startup_scanner.get_connection"
            ) as mock_conn:
                mock_conn.return_value.__enter__.return_value = MagicMock()

                with patch(
                    "reverie.indexer.startup_scanner.SessionIndexStateRepository"
                ) as mock_repo_class:
                    mock_repo = MagicMock()
                    mock_repo.get_by_profile.return_value = []
                    mock_repo_class.return_value = mock_repo

                    results = await scanner.scan_all()

        assert len(results) == 1
        assert results[0].profile_id == "test_profile"

    def test_scan_result_total_enqueued(self):
        """Test ScanResult.total_enqueued property."""
        result = ScanResult(
            profile_id="test",
            new_sessions=5,
            changed_sessions=3,
            failed_sessions=2,
            unchanged_sessions=10,
        )

        assert result.total_enqueued == 10  # 5 + 3 + 2

    @pytest.mark.asyncio
    async def test_scan_profile_orders_newest_files_first(self, work_queue):
        """Test that newest files are prioritized in the queue.

        Files with more recent mtime should have lower created_at values
        (using negative mtime), so they get dequeued first.
        """
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir)

            # Create files with distinct mtimes
            old_file = sessions_dir / "old_session.jsonl"
            mid_file = sessions_dir / "mid_session.jsonl"
            new_file = sessions_dir / "new_session.jsonl"

            # Create files with different timestamps
            old_file.write_text('{"type": "user"}\n')
            time.sleep(0.1)  # Ensure different mtimes
            mid_file.write_text('{"type": "user"}\n')
            time.sleep(0.1)
            new_file.write_text('{"type": "user"}\n')

            # Create profile with this directory
            mock_profile = MagicMock()
            mock_profile.id = "test_profile"
            mock_profile.sessions_path = sessions_dir
            mock_profile.sessions_paths = [sessions_dir]

            scanner = StartupScanner(work_queue)

            with patch(
                "reverie.indexer.startup_scanner.get_connection"
            ) as mock_conn:
                mock_conn.return_value.__enter__.return_value = MagicMock()

                with patch(
                    "reverie.indexer.startup_scanner.SessionIndexStateRepository"
                ) as mock_repo_class:
                    mock_repo = MagicMock()
                    mock_repo.get_by_profile.return_value = []
                    mock_repo_class.return_value = mock_repo

                    await scanner.scan_profile(mock_profile)

            # Dequeue all tasks and verify order (newest first)
            assert work_queue.qsize() == 3

            task1 = await work_queue.get()
            task2 = await work_queue.get()
            task3 = await work_queue.get()

            # Newest file should come first (has most negative created_at)
            assert task1.session_id == "new_session", f"Expected new_session first, got {task1.session_id}"
            assert task2.session_id == "mid_session", f"Expected mid_session second, got {task2.session_id}"
            assert task3.session_id == "old_session", f"Expected old_session third, got {task3.session_id}"

            # Verify created_at values are negative (based on mtime)
            assert task1.created_at < 0, "created_at should be negative (using -mtime)"
            assert task2.created_at < 0
            assert task3.created_at < 0

            # Verify ordering: newer files have more negative created_at (lower value = higher priority)
            assert task1.created_at < task2.created_at < task3.created_at


class TestMultiFolderScanning:
    """Tests for multi-folder indexing support."""

    @pytest.fixture
    def temp_multi_dirs(self):
        """Create multiple temporary directories with mock session files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project1 = Path(tmpdir) / "project1"
            project2 = Path(tmpdir) / "project2"
            project1.mkdir()
            project2.mkdir()

            # Create session files in project1
            (project1 / "session_a.jsonl").write_text('{"type": "user"}\n')
            (project1 / "session_b.jsonl").write_text('{"type": "user"}\n')

            # Create session files in project2 (including one with same name)
            (project2 / "session_a.jsonl").write_text('{"type": "user"}\n')
            (project2 / "session_c.jsonl").write_text('{"type": "user"}\n')

            yield (project1, project2)

    @pytest.fixture
    def mock_multi_profile(self, temp_multi_dirs):
        """Create a mock profile with multiple sessions directories."""
        project1, project2 = temp_multi_dirs
        profile = MagicMock()
        profile.id = "multi_profile"
        profile.name = "Multi Profile"
        profile.sessions_paths = [project1, project2]
        # For backward compat, sessions_path returns first
        profile.sessions_path = project1
        return profile

    @pytest.fixture
    def work_queue(self):
        """Create a work queue."""
        return WorkQueue()

    @pytest.fixture
    def scanner(self, work_queue):
        """Create a StartupScanner instance."""
        return StartupScanner(work_queue)

    @pytest.mark.asyncio
    async def test_scan_profile_multi_folder(
        self, scanner, mock_multi_profile, work_queue, temp_multi_dirs
    ):
        """Test scanning profile with multiple sessions directories."""
        project1, project2 = temp_multi_dirs

        with patch(
            "reverie.indexer.startup_scanner.get_connection"
        ) as mock_conn:
            mock_conn.return_value.__enter__.return_value = MagicMock()

            with patch(
                "reverie.indexer.startup_scanner.SessionIndexStateRepository"
            ) as mock_repo_class:
                mock_repo = MagicMock()
                mock_repo.get_by_profile.return_value = []
                mock_repo_class.return_value = mock_repo

                result = await scanner.scan_profile(mock_multi_profile)

        # Should find 4 sessions total (2 from each folder)
        assert result.new_sessions == 4
        assert work_queue.qsize() == 4

        # Verify tasks have correct indexing_path
        tasks = []
        while not work_queue.empty():
            task = await work_queue.get()
            tasks.append(task)
            await work_queue.mark_done(task)

        # Should have tasks from both folders
        project1_tasks = [t for t in tasks if t.indexing_path == str(project1)]
        project2_tasks = [t for t in tasks if t.indexing_path == str(project2)]

        assert len(project1_tasks) == 2
        assert len(project2_tasks) == 2

        # session_a exists in both folders - should be separate tasks
        session_a_tasks = [t for t in tasks if t.session_id == "session_a"]
        assert len(session_a_tasks) == 2
        # They should have different indexing_path
        assert session_a_tasks[0].indexing_path != session_a_tasks[1].indexing_path

    @pytest.mark.asyncio
    async def test_scan_profile_multi_folder_with_existing_states(
        self, scanner, mock_multi_profile, work_queue, temp_multi_dirs
    ):
        """Test scanning with existing states filters by indexing_path."""
        project1, project2 = temp_multi_dirs

        # Mock existing state for session_a in project1 only
        mock_state = MagicMock()
        mock_state.session_id = "session_a"
        mock_state.file_mtime = datetime.now(timezone.utc) + timedelta(seconds=1)
        mock_state.status = "completed"
        mock_state.indexing_path = str(project1)

        with patch(
            "reverie.indexer.startup_scanner.get_connection"
        ) as mock_conn:
            mock_conn.return_value.__enter__.return_value = MagicMock()

            with patch(
                "reverie.indexer.startup_scanner.SessionIndexStateRepository"
            ) as mock_repo_class:
                mock_repo = MagicMock()
                # Return state only when filtering by project1 path
                def get_by_profile_side_effect(profile_id, indexing_path=None):
                    if indexing_path == str(project1):
                        return [mock_state]
                    return []
                mock_repo.get_by_profile.side_effect = get_by_profile_side_effect
                mock_repo_class.return_value = mock_repo

                result = await scanner.scan_profile(mock_multi_profile)

        # session_a in project1 is unchanged (existing state)
        # session_b in project1 is new
        # session_a in project2 is new (different folder)
        # session_c in project2 is new
        assert result.unchanged_sessions == 1
        assert result.new_sessions == 3
        assert work_queue.qsize() == 3
