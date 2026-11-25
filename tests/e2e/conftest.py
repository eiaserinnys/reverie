"""E2E test fixtures for Setup Wizard.

Provides isolated test environments for each E2E test scenario.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Generator

import pytest
import requests


@pytest.fixture(scope="function")
def temp_project_dir() -> Generator[Path, None, None]:
    """Create a temporary project directory for E2E testing.

    This directory contains:
    - .reverie/ folder for setup state
    - .env file for configuration

    Yields:
        Path to the temporary project directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        # Create .reverie folder
        (project_dir / ".reverie").mkdir(parents=True, exist_ok=True)
        yield project_dir


@pytest.fixture(scope="function")
def test_server(temp_project_dir: Path) -> Generator[str, None, None]:
    """Start a test server for E2E testing.

    Starts the MCP server with the test project directory.

    Args:
        temp_project_dir: Path to the temporary project directory.

    Yields:
        Base URL of the test server.
    """
    import subprocess
    import signal

    # Find an available port
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    # Set environment variables
    env = os.environ.copy()
    env["REVERIE_PORT"] = str(port)
    env["REVERIE_HOST"] = "127.0.0.1"

    # Start the server
    server_process = subprocess.Popen(
        [
            "python",
            "-m",
            "reverie.mcp.server",
            "--host", "127.0.0.1",
            "--port", str(port),
        ],
        env=env,
        cwd=str(temp_project_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    base_url = f"http://127.0.0.1:{port}"

    # Wait for server to be ready
    max_wait = 30
    for i in range(max_wait):
        try:
            response = requests.get(f"{base_url}/api/setup/status", timeout=1)
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    else:
        server_process.kill()
        stdout, stderr = server_process.communicate()
        raise RuntimeError(
            f"Server failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}"
        )

    yield base_url

    # Cleanup
    server_process.send_signal(signal.SIGTERM)
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()


class SetupWizardClient:
    """Client for interacting with the Setup Wizard API."""

    def __init__(self, base_url: str):
        """Initialize the client.

        Args:
            base_url: Base URL of the server.
        """
        self.base_url = base_url

    def get_status(self) -> dict:
        """Get setup status."""
        response = requests.get(f"{self.base_url}/api/setup/status")
        response.raise_for_status()
        return response.json()

    def check_sqlite(self) -> dict:
        """Check sqlite-vec availability."""
        response = requests.get(f"{self.base_url}/api/setup/check/sqlite")
        response.raise_for_status()
        return response.json()

    def check_postgres(self, config: dict) -> dict:
        """Check PostgreSQL connection."""
        response = requests.post(
            f"{self.base_url}/api/setup/check/postgres",
            json=config,
        )
        response.raise_for_status()
        return response.json()

    def check_docker(self) -> dict:
        """Check Docker availability."""
        response = requests.get(f"{self.base_url}/api/setup/check/docker")
        response.raise_for_status()
        return response.json()

    def select_sqlite_backend(self, sqlite_path: str | None = None) -> dict:
        """Select SQLite backend."""
        payload = {}
        if sqlite_path:
            payload["sqlite_path"] = sqlite_path
        response = requests.post(
            f"{self.base_url}/api/setup/backend/sqlite",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def select_postgres_backend(self) -> dict:
        """Select PostgreSQL backend."""
        response = requests.post(f"{self.base_url}/api/setup/backend/postgres")
        response.raise_for_status()
        return response.json()

    def save_database(self, config: dict) -> dict:
        """Save database configuration."""
        response = requests.post(
            f"{self.base_url}/api/setup/database",
            json=config,
        )
        response.raise_for_status()
        return response.json()

    def save_apikey(self, api_key: str) -> dict:
        """Save API key."""
        response = requests.post(
            f"{self.base_url}/api/setup/apikey",
            json={"api_key": api_key},
        )
        response.raise_for_status()
        return response.json()

    def run_migration(self) -> dict:
        """Run database migrations."""
        response = requests.post(f"{self.base_url}/api/setup/migrate")
        response.raise_for_status()
        return response.json()

    def verify_setup(self) -> dict:
        """Verify setup."""
        response = requests.get(f"{self.base_url}/api/setup/verify")
        response.raise_for_status()
        return response.json()

    def complete_setup(self) -> dict:
        """Complete setup."""
        response = requests.post(f"{self.base_url}/api/setup/complete")
        response.raise_for_status()
        return response.json()

    def start_docker_postgres(self, config: dict | None = None) -> dict:
        """Start PostgreSQL in Docker."""
        response = requests.post(
            f"{self.base_url}/api/setup/docker/start",
            json=config or {},
        )
        response.raise_for_status()
        return response.json()

    def stop_docker_postgres(self) -> dict:
        """Stop PostgreSQL Docker container."""
        response = requests.post(f"{self.base_url}/api/setup/docker/stop")
        response.raise_for_status()
        return response.json()


@pytest.fixture
def wizard_client(test_server: str) -> SetupWizardClient:
    """Create a SetupWizardClient for the test server.

    Args:
        test_server: Base URL of the test server.

    Returns:
        SetupWizardClient instance.
    """
    return SetupWizardClient(test_server)
