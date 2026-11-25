"""Tests for Docker manager functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reverie.docker.docker_manager import (
    ContainerStatus,
    ContainerStatusResult,
    DockerCheckResult,
    DockerManager,
    OSType,
    get_os_type,
)


class TestDockerCheckResult:
    """Tests for DockerCheckResult dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        result = DockerCheckResult(installed=False)
        assert result.installed is False
        assert result.version is None
        assert result.compose_available is False
        assert result.compose_version is None
        assert result.error is None

    def test_with_values(self):
        """Should store provided values."""
        result = DockerCheckResult(
            installed=True,
            version="Docker version 24.0.0",
            compose_available=True,
            compose_version="v2.20.0",
        )
        assert result.installed is True
        assert result.version == "Docker version 24.0.0"
        assert result.compose_available is True
        assert result.compose_version == "v2.20.0"


class TestContainerStatus:
    """Tests for ContainerStatus enum."""

    def test_status_values(self):
        """Should have expected status values."""
        assert ContainerStatus.RUNNING.value == "running"
        assert ContainerStatus.STOPPED.value == "stopped"
        assert ContainerStatus.NOT_FOUND.value == "not_found"
        assert ContainerStatus.ERROR.value == "error"


class TestDockerManager:
    """Tests for DockerManager class."""

    def test_init_with_project_dir(self):
        """Should store project directory."""
        manager = DockerManager(project_dir="/tmp/test")
        assert manager.project_dir == Path("/tmp/test")

    def test_init_without_project_dir(self):
        """Should accept None project_dir."""
        manager = DockerManager()
        assert manager.project_dir is None

    @patch("shutil.which")
    def test_check_docker_not_installed(self, mock_which):
        """Should return not installed when docker not in PATH."""
        mock_which.return_value = None
        manager = DockerManager()
        result = manager.check_docker()

        assert result.installed is False
        assert "not found" in result.error.lower()

    @patch("subprocess.run")
    @patch("shutil.which")
    def test_check_docker_installed(self, mock_which, mock_run):
        """Should detect installed Docker."""
        mock_which.return_value = "/usr/bin/docker"

        # Mock docker --version
        version_result = MagicMock()
        version_result.returncode = 0
        version_result.stdout = "Docker version 24.0.0, build abcdef"

        # Mock docker info
        info_result = MagicMock()
        info_result.returncode = 0

        # Mock docker compose version
        compose_result = MagicMock()
        compose_result.returncode = 0
        compose_result.stdout = "Docker Compose version v2.20.0"

        mock_run.side_effect = [version_result, info_result, compose_result]

        manager = DockerManager()
        result = manager.check_docker()

        assert result.installed is True
        assert "24.0.0" in result.version
        assert result.compose_available is True

    @patch("subprocess.run")
    @patch("shutil.which")
    def test_check_docker_daemon_not_running(self, mock_which, mock_run):
        """Should detect when Docker daemon is not running."""
        mock_which.return_value = "/usr/bin/docker"

        version_result = MagicMock()
        version_result.returncode = 0
        version_result.stdout = "Docker version 24.0.0"

        info_result = MagicMock()
        info_result.returncode = 1
        info_result.stderr = "Cannot connect to the Docker daemon"

        mock_run.side_effect = [version_result, info_result]

        manager = DockerManager()
        result = manager.check_docker()

        # Docker is installed but daemon is not running
        assert result.installed is True
        assert result.daemon_running is False
        assert "daemon" in result.error.lower()

    def test_generate_compose(self):
        """Should generate compose file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DockerManager(project_dir=tmpdir)
            compose_path = manager.generate_compose(
                postgres_user="test",
                postgres_password="test",
                postgres_db="testdb",
            )

            assert compose_path.exists()
            content = compose_path.read_text()
            assert "POSTGRES_USER: test" in content

    @patch.object(DockerManager, "check_docker")
    def test_start_docker_not_installed(self, mock_check):
        """Should fail when Docker not installed."""
        mock_check.return_value = DockerCheckResult(
            installed=False,
            error="Docker not installed",
        )

        manager = DockerManager()
        success, message = manager.start()

        assert success is False
        assert "not installed" in message.lower()

    @patch.object(DockerManager, "check_docker")
    def test_start_compose_not_available(self, mock_check):
        """Should fail when Docker Compose not available."""
        mock_check.return_value = DockerCheckResult(
            installed=True,
            compose_available=False,
        )

        manager = DockerManager()
        success, message = manager.start()

        assert success is False
        assert "compose" in message.lower()

    @patch("subprocess.run")
    def test_get_status_container_running(self, mock_run):
        """Should detect running container."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123def456|running|healthy",
            stderr="",
        )

        manager = DockerManager()
        result = manager.get_status()

        assert result.status == ContainerStatus.RUNNING
        assert result.container_id == "abc123def456"
        assert result.health == "healthy"

    @patch("subprocess.run")
    def test_get_status_container_not_found(self, mock_run):
        """Should detect when container not found."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: No such object: reverie-postgres",
        )

        manager = DockerManager()
        result = manager.get_status()

        assert result.status == ContainerStatus.NOT_FOUND

    @patch("subprocess.run")
    def test_get_status_container_stopped(self, mock_run):
        """Should detect stopped container."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123def456|exited|",
            stderr="",
        )

        manager = DockerManager()
        result = manager.get_status()

        assert result.status == ContainerStatus.STOPPED

    def test_is_running_true(self):
        """Should return True when container is running."""
        manager = DockerManager()
        with patch.object(
            manager,
            "get_status",
            return_value=ContainerStatusResult(status=ContainerStatus.RUNNING),
        ):
            assert manager.is_running() is True

    def test_is_running_false(self):
        """Should return False when container is not running."""
        manager = DockerManager()
        with patch.object(
            manager,
            "get_status",
            return_value=ContainerStatusResult(status=ContainerStatus.STOPPED),
        ):
            assert manager.is_running() is False


class TestStartDockerDesktop:
    """Tests for start_docker_desktop method."""

    @patch("reverie.docker.docker_manager.get_os_type")
    def test_skip_on_non_windows(self, mock_os_type):
        """Should return False on non-Windows platforms."""
        mock_os_type.return_value = OSType.LINUX

        manager = DockerManager()
        success, msg = manager.start_docker_desktop()

        assert success is False
        assert "manually" in msg.lower()

    @patch("reverie.docker.docker_manager.get_os_type")
    def test_skip_on_macos(self, mock_os_type):
        """Should return False on macOS."""
        mock_os_type.return_value = OSType.MACOS

        manager = DockerManager()
        success, msg = manager.start_docker_desktop()

        assert success is False
        assert "manually" in msg.lower()

    @patch("os.environ.get")
    @patch("reverie.docker.docker_manager.get_os_type")
    def test_docker_desktop_not_found(self, mock_os_type, mock_env):
        """Should fail when Docker Desktop exe not found."""
        mock_os_type.return_value = OSType.WINDOWS
        mock_env.return_value = r"C:\Program Files"

        manager = DockerManager()

        # Patch Path.exists to return False for all paths
        with patch.object(Path, "exists", return_value=False):
            success, msg = manager.start_docker_desktop()

        assert success is False
        assert "not found" in msg.lower()

    @patch("subprocess.Popen")
    @patch("subprocess.run")
    @patch("reverie.docker.docker_manager.get_os_type")
    def test_docker_desktop_starts_successfully(
        self, mock_os_type, mock_run, mock_popen
    ):
        """Should start Docker Desktop and wait for daemon."""
        mock_os_type.return_value = OSType.WINDOWS

        # Mock docker info to succeed on second call
        mock_run.side_effect = [
            MagicMock(returncode=1),  # First check fails
            MagicMock(returncode=0),  # Second check succeeds
        ]

        manager = DockerManager()

        # Patch Path.exists to find Docker Desktop
        with patch.object(Path, "exists", return_value=True):
            success, msg = manager.start_docker_desktop(timeout=10)

        assert success is True
        assert "started" in msg.lower() or "success" in msg.lower()
        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
    @patch("subprocess.run")
    @patch("reverie.docker.docker_manager.get_os_type")
    def test_docker_desktop_timeout(self, mock_os_type, mock_run, mock_popen):
        """Should timeout if Docker daemon doesn't start."""
        mock_os_type.return_value = OSType.WINDOWS

        # Mock docker info to always fail
        mock_run.return_value = MagicMock(returncode=1)

        manager = DockerManager()

        with patch.object(Path, "exists", return_value=True):
            # Use short timeout for test
            success, msg = manager.start_docker_desktop(timeout=1)

        assert success is False
        assert "not ready" in msg.lower() or "timeout" in msg.lower()
