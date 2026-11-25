"""Docker container lifecycle management.

Manages PostgreSQL container using docker-compose:
- Check Docker installation
- Start/stop containers
- Check container status
- Wait for container readiness
- OS-specific installation guidance
"""

import logging
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from reverie.docker.compose_template import (
    compose_exists,
    generate_compose_file,
    get_compose_path,
    get_local_dir,
)

logger = logging.getLogger(__name__)


class OSType(str, Enum):
    """Operating system type."""

    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"


def get_os_type() -> OSType:
    """Detect the current operating system.

    Returns:
        OSType enum value.
    """
    system = platform.system().lower()
    if system == "windows":
        return OSType.WINDOWS
    elif system == "linux":
        return OSType.LINUX
    elif system == "darwin":
        return OSType.MACOS
    return OSType.UNKNOWN


def get_os_info() -> dict:
    """Get detailed OS information.

    Returns:
        Dictionary with OS details.
    """
    return {
        "type": get_os_type().value,
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }


# Docker installation guides per OS
DOCKER_INSTALL_GUIDES = {
    OSType.WINDOWS: {
        "title": "Docker Desktop for Windows",
        "url": "https://docs.docker.com/desktop/install/windows-install/",
        "steps": [
            "1. Download Docker Desktop from https://www.docker.com/products/docker-desktop",
            "2. Run the installer (requires administrator)",
            "3. Enable WSL 2 backend when prompted (recommended)",
            "4. Restart your computer after installation",
            "5. Start Docker Desktop from the Start menu",
            "6. Wait for Docker to start (icon in system tray turns green)",
        ],
        "notes": [
            "WSL 2 backend is recommended for better performance",
            "If you see 'WSL 2 installation incomplete', follow the WSL setup guide",
            "Ensure virtualization is enabled in BIOS/UEFI",
        ],
    },
    OSType.LINUX: {
        "title": "Docker Engine for Linux",
        "url": "https://docs.docker.com/engine/install/",
        "steps": [
            "# For Ubuntu/Debian:",
            "1. sudo apt-get update",
            "2. sudo apt-get install docker.io docker-compose-plugin",
            "3. sudo systemctl start docker",
            "4. sudo systemctl enable docker",
            "5. sudo usermod -aG docker $USER",
            "6. Log out and log back in (or run: newgrp docker)",
            "",
            "# For Fedora/RHEL:",
            "1. sudo dnf install docker docker-compose-plugin",
            "2. sudo systemctl start docker",
            "3. sudo systemctl enable docker",
            "4. sudo usermod -aG docker $USER",
        ],
        "notes": [
            "You must add your user to the 'docker' group to run without sudo",
            "After adding to group, log out and back in for changes to take effect",
            "Run 'docker run hello-world' to verify installation",
        ],
    },
    OSType.MACOS: {
        "title": "Docker Desktop for macOS",
        "url": "https://docs.docker.com/desktop/install/mac-install/",
        "steps": [
            "1. Download Docker Desktop from https://www.docker.com/products/docker-desktop",
            "2. Open the .dmg file and drag Docker to Applications",
            "3. Open Docker from Applications folder",
            "4. Accept the terms and grant permissions when prompted",
            "5. Wait for Docker to start (icon in menu bar shows 'running')",
        ],
        "notes": [
            "Choose the correct version for your Mac (Intel or Apple Silicon)",
            "First launch may take a few minutes to initialize",
        ],
    },
}


class ContainerStatus(str, Enum):
    """Container status enum."""

    RUNNING = "running"
    STOPPED = "stopped"
    NOT_FOUND = "not_found"
    ERROR = "error"


@dataclass
class DockerCheckResult:
    """Result of Docker installation check."""

    installed: bool
    version: str | None = None
    daemon_running: bool = False
    compose_available: bool = False
    compose_version: str | None = None
    error: str | None = None
    os_type: str = field(default_factory=lambda: get_os_type().value)
    install_guide: dict | None = None

    def __post_init__(self):
        """Add installation guide if Docker is not installed."""
        if not self.installed and self.install_guide is None:
            os_type = OSType(self.os_type) if self.os_type in [e.value for e in OSType] else OSType.UNKNOWN
            self.install_guide = DOCKER_INSTALL_GUIDES.get(os_type)


@dataclass
class ContainerStatusResult:
    """Result of container status check."""

    status: ContainerStatus
    container_id: str | None = None
    health: str | None = None
    error: str | None = None


class DockerManager:
    """Manages Docker containers for Reverie.

    Provides methods to:
    - Check Docker installation
    - Generate docker-compose.yml
    - Start/stop PostgreSQL container
    - Check container status and health
    """

    CONTAINER_NAME = "reverie-postgres"
    STARTUP_TIMEOUT = 60  # seconds to wait for container to be ready

    def __init__(self, project_dir: Path | str | None = None):
        """Initialize DockerManager.

        Args:
            project_dir: Optional project root directory.
        """
        self.project_dir = Path(project_dir) if project_dir else None

    def check_docker(self) -> DockerCheckResult:
        """Check if Docker is installed and available.

        Returns:
            DockerCheckResult with installation status.
        """
        result = DockerCheckResult(installed=False)

        # Check docker command
        docker_path = shutil.which("docker")
        if not docker_path:
            result.error = "Docker not found in PATH"
            return result

        try:
            # Get Docker version
            proc = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                result.installed = True
                result.version = proc.stdout.strip()
            else:
                result.error = proc.stderr.strip() or "Docker command failed"
                return result

            # Check if Docker daemon is running
            proc = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                # Docker is installed but daemon is not running
                result.daemon_running = False
                result.error = "Docker daemon is not running. Please start Docker Desktop."
                return result

            result.daemon_running = True

            # Check docker compose (v2 - built into docker)
            proc = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                result.compose_available = True
                result.compose_version = proc.stdout.strip()
            else:
                # Try legacy docker-compose command
                compose_path = shutil.which("docker-compose")
                if compose_path:
                    proc = subprocess.run(
                        ["docker-compose", "--version"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if proc.returncode == 0:
                        result.compose_available = True
                        result.compose_version = proc.stdout.strip()

        except subprocess.TimeoutExpired:
            result.error = "Docker command timed out"
        except Exception as e:
            result.error = str(e)

        return result

    def _get_compose_command(self) -> list[str]:
        """Get the docker compose command.

        Returns:
            Command list for docker compose.
        """
        # Prefer docker compose v2
        try:
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return ["docker", "compose"]
        except Exception:
            pass

        # Fallback to legacy docker-compose
        return ["docker-compose"]

    def generate_compose(
        self,
        postgres_user: str = "reverie",
        postgres_password: str = "reverie",
        postgres_db: str = "reverie",
        postgres_port: int = 5432,
    ) -> Path:
        """Generate docker-compose.yml file.

        Args:
            postgres_user: PostgreSQL username.
            postgres_password: PostgreSQL password.
            postgres_db: PostgreSQL database name.
            postgres_port: Host port for PostgreSQL.

        Returns:
            Path to the generated docker-compose.yml.
        """
        return generate_compose_file(
            project_dir=self.project_dir,
            postgres_user=postgres_user,
            postgres_password=postgres_password,
            postgres_db=postgres_db,
            postgres_port=postgres_port,
        )

    def start(
        self,
        postgres_user: str = "reverie",
        postgres_password: str = "reverie",
        postgres_db: str = "reverie",
        postgres_port: int = 5432,
        wait_ready: bool = True,
    ) -> tuple[bool, str]:
        """Start PostgreSQL container.

        Generates docker-compose.yml if it doesn't exist,
        then runs docker compose up.

        Args:
            postgres_user: PostgreSQL username.
            postgres_password: PostgreSQL password.
            postgres_db: PostgreSQL database name.
            postgres_port: Host port for PostgreSQL.
            wait_ready: Whether to wait for container to be ready.

        Returns:
            Tuple of (success, message).
        """
        # Check Docker first
        docker_check = self.check_docker()
        if not docker_check.installed:
            return False, docker_check.error or "Docker not installed"
        if not docker_check.compose_available:
            return False, "Docker Compose not available"

        # Generate compose file if needed
        if not compose_exists(self.project_dir):
            try:
                self.generate_compose(
                    postgres_user=postgres_user,
                    postgres_password=postgres_password,
                    postgres_db=postgres_db,
                    postgres_port=postgres_port,
                )
            except Exception as e:
                return False, f"Failed to generate docker-compose.yml: {e}"

        compose_path = get_compose_path(self.project_dir)
        local_dir = get_local_dir(self.project_dir)

        # Run docker compose up
        compose_cmd = self._get_compose_command()
        try:
            proc = subprocess.run(
                [*compose_cmd, "-f", str(compose_path), "up", "-d"],
                cwd=str(local_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )

            if proc.returncode != 0:
                error_msg = proc.stderr.strip() or proc.stdout.strip()
                return False, f"Failed to start container: {error_msg}"

            logger.info("Container started successfully")

            if wait_ready:
                ready, msg = self.wait_ready()
                if not ready:
                    return False, msg

            return True, "PostgreSQL container started successfully"

        except subprocess.TimeoutExpired:
            return False, "Container start timed out"
        except Exception as e:
            return False, f"Failed to start container: {e}"

    def stop(self) -> tuple[bool, str]:
        """Stop PostgreSQL container.

        Returns:
            Tuple of (success, message).
        """
        compose_path = get_compose_path(self.project_dir)
        if not compose_path.exists():
            return True, "No docker-compose.yml found, nothing to stop"

        local_dir = get_local_dir(self.project_dir)
        compose_cmd = self._get_compose_command()

        try:
            proc = subprocess.run(
                [*compose_cmd, "-f", str(compose_path), "down"],
                cwd=str(local_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )

            if proc.returncode != 0:
                error_msg = proc.stderr.strip() or proc.stdout.strip()
                return False, f"Failed to stop container: {error_msg}"

            return True, "PostgreSQL container stopped"

        except subprocess.TimeoutExpired:
            return False, "Container stop timed out"
        except Exception as e:
            return False, f"Failed to stop container: {e}"

    def get_status(self) -> ContainerStatusResult:
        """Get container status.

        Returns:
            ContainerStatusResult with current status.
        """
        try:
            # Check if container exists and get its status
            proc = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.Id}}|{{.State.Status}}|{{.State.Health.Status}}",
                    self.CONTAINER_NAME,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if proc.returncode != 0:
                if "No such object" in proc.stderr or "Error: No such" in proc.stderr:
                    return ContainerStatusResult(status=ContainerStatus.NOT_FOUND)
                return ContainerStatusResult(
                    status=ContainerStatus.ERROR,
                    error=proc.stderr.strip(),
                )

            output = proc.stdout.strip()
            parts = output.split("|")

            container_id = parts[0][:12] if parts else None
            state = parts[1] if len(parts) > 1 else "unknown"
            health = parts[2] if len(parts) > 2 else None

            if state == "running":
                return ContainerStatusResult(
                    status=ContainerStatus.RUNNING,
                    container_id=container_id,
                    health=health if health else "unknown",
                )
            else:
                return ContainerStatusResult(
                    status=ContainerStatus.STOPPED,
                    container_id=container_id,
                )

        except subprocess.TimeoutExpired:
            return ContainerStatusResult(
                status=ContainerStatus.ERROR,
                error="Status check timed out",
            )
        except Exception as e:
            return ContainerStatusResult(
                status=ContainerStatus.ERROR,
                error=str(e),
            )

    def wait_ready(self, timeout: int | None = None) -> tuple[bool, str]:
        """Wait for container to be ready.

        Args:
            timeout: Maximum seconds to wait (default: STARTUP_TIMEOUT).

        Returns:
            Tuple of (ready, message).
        """
        if timeout is None:
            timeout = self.STARTUP_TIMEOUT

        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_status()

            if status.status == ContainerStatus.RUNNING:
                if status.health == "healthy":
                    return True, "Container is ready"
                # If no healthcheck or still starting, try connection
                if status.health in ("starting", "unknown", None):
                    # Give it a moment then continue checking
                    time.sleep(2)
                    continue
                elif status.health == "unhealthy":
                    return False, "Container is unhealthy"

            elif status.status == ContainerStatus.NOT_FOUND:
                return False, "Container not found"

            elif status.status == ContainerStatus.ERROR:
                return False, status.error or "Unknown error"

            time.sleep(2)

        return False, f"Container not ready after {timeout} seconds"

    def is_running(self) -> bool:
        """Check if container is running.

        Returns:
            True if container is running.
        """
        status = self.get_status()
        return status.status == ContainerStatus.RUNNING

    def start_docker_desktop(self, timeout: int = 120) -> tuple[bool, str]:
        """Start Docker Desktop on Windows.

        Attempts to launch Docker Desktop and waits for the daemon to become ready.
        Only works on Windows; on other platforms, returns immediately with guidance.

        Args:
            timeout: Maximum seconds to wait for Docker daemon to become ready.

        Returns:
            Tuple of (success, message).
        """
        os_type = get_os_type()

        if os_type != OSType.WINDOWS:
            guide = DOCKER_INSTALL_GUIDES.get(os_type, {})
            return False, f"Please start Docker manually. {guide.get('title', 'Docker')}"

        # Find Docker Desktop executable
        docker_desktop_paths = [
            Path(r"C:\Program Files\Docker\Docker\Docker Desktop.exe"),
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Docker"
            / "Docker"
            / "Docker Desktop.exe",
        ]

        docker_exe = None
        for path in docker_desktop_paths:
            if path.exists():
                docker_exe = path
                break

        if not docker_exe:
            return False, (
                "Docker Desktop not found. "
                "Please install from https://www.docker.com/products/docker-desktop"
            )

        # Launch Docker Desktop in background
        try:
            logger.info(f"Starting Docker Desktop: {docker_exe}")
            # Windows-specific flags for background process
            creationflags = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "DETACHED_PROCESS")
                else 0
            )
            subprocess.Popen(
                [str(docker_exe)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except Exception as e:
            return False, f"Failed to start Docker Desktop: {e}"

        # Wait for Docker daemon to become ready
        logger.info(f"Waiting for Docker daemon (timeout: {timeout}s)...")
        start_time = time.time()
        last_log_time = start_time

        while time.time() - start_time < timeout:
            try:
                proc = subprocess.run(
                    ["docker", "info"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode == 0:
                    elapsed = int(time.time() - start_time)
                    logger.info(f"Docker daemon ready after {elapsed}s")
                    return True, f"Docker Desktop started successfully ({elapsed}s)"
            except (subprocess.TimeoutExpired, Exception):
                pass

            # Log progress every 15 seconds
            if time.time() - last_log_time >= 15:
                elapsed = int(time.time() - start_time)
                logger.info(f"Still waiting for Docker daemon... ({elapsed}s)")
                last_log_time = time.time()

            time.sleep(3)

        return False, (
            f"Docker daemon not ready after {timeout}s. "
            "Please ensure Docker Desktop is installed and try starting it manually."
        )
