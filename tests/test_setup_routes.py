"""Tests for Setup API routes.

Tests the Setup API endpoints that power the Setup Wizard.
Uses direct function testing instead of full HTTP testing due to
FastMCP's architecture.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from starlette.testclient import TestClient


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def setup_state(temp_project_dir):
    """Create a real SetupState with temporary directory."""
    from reverie.config.setup_state import SetupState
    return SetupState(project_dir=temp_project_dir)


@pytest.fixture
def app_with_routes(temp_project_dir, setup_state):
    """Create a test application with setup routes."""
    from mcp.server.fastmcp import FastMCP
    from reverie.api.setup_routes import register_setup_routes
    import reverie.api.setup_routes as routes_module

    # Patch the global setup state
    routes_module._setup_state = setup_state

    mcp = FastMCP("test", host="127.0.0.1", port=8999)
    register_setup_routes(mcp)

    # Get the SSE app which is a Starlette instance
    return mcp.sse_app()


@pytest.fixture
def client(app_with_routes):
    """Create a test client."""
    return TestClient(app_with_routes)


class TestSetupStatus:
    """Tests for GET /api/setup/status."""

    def test_returns_setup_status(self, client):
        """Should return the current setup status."""
        response = client.get("/api/setup/status")

        assert response.status_code == 200
        data = response.json()
        assert "setup_complete" in data
        assert "phases" in data
        assert data["setup_complete"] is False

    def test_has_cors_headers(self, client):
        """Should include CORS headers."""
        response = client.get("/api/setup/status")

        assert "access-control-allow-origin" in response.headers


class TestPostgresCheck:
    """Tests for POST /api/setup/check/postgres."""

    @patch("reverie.api.setup_routes.psycopg2")
    def test_successful_connection(self, mock_psycopg2, client):
        """Should report success on valid connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("PostgreSQL 15.0",)
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_psycopg2.connect.return_value = mock_conn

        response = client.post(
            "/api/setup/check/postgres",
            json={
                "host": "localhost",
                "port": 5432,
                "user": "test",
                "password": "test",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_conn.close.assert_called_once()

    @patch("reverie.api.setup_routes.psycopg2")
    def test_failed_connection(self, mock_psycopg2, client):
        """Should report failure on invalid connection."""
        mock_psycopg2.connect.side_effect = Exception("Connection refused")

        response = client.post(
            "/api/setup/check/postgres",
            json={
                "host": "localhost",
                "port": 5432,
                "user": "test",
                "password": "test",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data

    @patch("reverie.api.setup_routes._get_docker_manager")
    def test_requires_user(self, mock_docker_manager, client):
        """Should require user parameter when Docker is not running."""
        from reverie.docker.docker_manager import ContainerStatus

        # Mock Docker manager to return NOT_CREATED (not running)
        mock_mgr = MagicMock()
        mock_status = MagicMock()
        mock_status.status = ContainerStatus.NOT_FOUND
        mock_mgr.get_status.return_value = mock_status
        mock_docker_manager.return_value = mock_mgr

        response = client.post(
            "/api/setup/check/postgres",
            json={
                "host": "localhost",
                "port": 5432,
                "password": "test",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data


class TestPgvectorCheck:
    """Tests for POST /api/setup/check/pgvector."""

    @patch("reverie.api.setup_routes.psycopg2")
    def test_pgvector_installed(self, mock_psycopg2, client):
        """Should detect pgvector extension."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # First call: check availability, Second: check installed, Third: get version
        mock_cursor.fetchone.side_effect = [(True,), (True,), ("0.5.0",)]
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_psycopg2.connect.return_value = mock_conn

        response = client.post(
            "/api/setup/check/pgvector",
            json={
                "host": "localhost",
                "port": 5432,
                "user": "test",
                "password": "test",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["installed"] is True

    @patch("reverie.api.setup_routes.psycopg2")
    def test_pgvector_not_available(self, mock_psycopg2, client):
        """Should report when pgvector is not available."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (False,)  # Not available
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_psycopg2.connect.return_value = mock_conn

        response = client.post(
            "/api/setup/check/pgvector",
            json={
                "host": "localhost",
                "port": 5432,
                "user": "test",
                "password": "test",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestDatabaseSave:
    """Tests for POST /api/setup/database."""

    def test_saves_database_config(self, client, setup_state):
        """Should save database configuration."""
        response = client.post(
            "/api/setup/database",
            json={
                "host": "localhost",
                "port": 5432,
                "user": "reverie",
                "password": "secret",
                "name": "reverie_db",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify config was saved
        config = setup_state.get_database_config()
        assert config["host"] == "localhost"
        assert config["user"] == "reverie"

    def test_requires_all_fields(self, client):
        """Should require all database fields."""
        response = client.post(
            "/api/setup/database",
            json={"host": "localhost"},  # Missing other fields
        )

        assert response.status_code == 400


class TestApiKeySave:
    """Tests for POST /api/setup/apikey."""

    @patch("reverie.api.setup_routes.openai")
    def test_saves_valid_api_key(self, mock_openai, client, setup_state):
        """Should save valid API key."""
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.models.list.return_value = []

        response = client.post(
            "/api/setup/apikey",
            json={"openai_api_key": "sk-test-valid-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify key was saved
        saved_key = setup_state.get_api_key()
        assert saved_key == "sk-test-valid-key"

    @patch("reverie.api.setup_routes.openai")
    def test_rejects_invalid_api_key(self, mock_openai, client):
        """Should reject invalid API key."""
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        # Create a mock AuthenticationError class
        mock_openai.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_client.models.list.side_effect = mock_openai.AuthenticationError("Invalid API key")

        response = client.post(
            "/api/setup/apikey",
            json={"openai_api_key": "invalid-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestProfileSave:
    """Tests for POST /api/setup/profile."""

    def test_saves_profile_config(self, client, setup_state):
        """Should save profile configuration."""
        response = client.post(
            "/api/setup/profile",
            json={
                "profile_name": "default",
                "sessions_path": "/path/to/sessions",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_saves_multiple_sessions_paths(self, client, setup_state):
        """Should save multiple sessions paths (multi-folder indexing)."""
        response = client.post(
            "/api/setup/profile",
            json={
                "profile_name": "multi-project",
                "sessions_paths": [
                    "/home/user/.claude/projects/project-a",
                    "/home/user/.claude/projects/project-b",
                    "/home/user/.claude/projects/project-c",
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify config was saved
        phase = setup_state.config.phases.get("profile")
        assert phase is not None
        assert "sessions_paths" in phase.config
        assert len(phase.config["sessions_paths"]) == 3

    def test_backwards_compatible_single_path(self, client, setup_state):
        """Should accept single sessions_path for backwards compatibility."""
        response = client.post(
            "/api/setup/profile",
            json={
                "profile_name": "legacy",
                "sessions_path": "/legacy/single/path",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Single path should be stored as a list
        phase = setup_state.config.phases.get("profile")
        assert phase is not None
        # Should be converted to list internally
        sessions_paths = phase.config.get("sessions_paths", [])
        assert len(sessions_paths) == 1
        assert sessions_paths[0] == "/legacy/single/path"

    def test_validates_all_paths(self, client, setup_state):
        """Should validate all paths are absolute."""
        response = client.post(
            "/api/setup/profile",
            json={
                "profile_name": "invalid",
                "sessions_paths": [
                    "/valid/absolute/path",
                    "relative/path",  # Invalid
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data

    def test_empty_paths_array_allowed(self, client, setup_state):
        """Should allow empty sessions_paths array."""
        response = client.post(
            "/api/setup/profile",
            json={
                "profile_name": "empty",
                "sessions_paths": [],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestMigrate:
    """Tests for POST /api/setup/migrate."""

    @patch("reverie.api.setup_routes.run_migrations")
    def test_runs_migration(self, mock_run, client):
        """Should run database migrations."""
        mock_run.return_value = (True, "Success")

        response = client.post("/api/setup/migrate", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestVerify:
    """Tests for GET /api/setup/verify."""

    @patch("reverie.api.setup_routes.psycopg2")
    def test_runs_verification(self, mock_psycopg2, client, setup_state):
        """Should run verification checks."""
        # Setup: save an API key so that check passes
        setup_state.save_api_key("sk-test-key")

        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (True,)  # tables exist
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_psycopg2.connect.return_value = mock_conn

        response = client.get("/api/setup/verify")

        assert response.status_code == 200
        data = response.json()
        assert "checks" in data


class TestComplete:
    """Tests for POST /api/setup/complete."""

    def test_marks_setup_complete(self, client, setup_state):
        """Should mark setup as complete."""
        response = client.post("/api/setup/complete", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify it was marked complete
        assert setup_state.is_complete() is True


class TestSqliteCheck:
    """Tests for GET /api/setup/check/sqlite."""

    def test_sqlite_vec_available(self, client):
        """Should detect sqlite-vec availability."""
        response = client.get("/api/setup/check/sqlite")

        assert response.status_code == 200
        data = response.json()
        assert "available" in data
        assert "version" in data
        # Since we have sqlite-vec installed in test env
        assert data["available"] is True

    def test_updates_phase_check(self, client, setup_state):
        """Should update prerequisites phase check."""
        response = client.get("/api/setup/check/sqlite")

        assert response.status_code == 200

        # Verify phase check was updated
        phase = setup_state.config.phases.get("prerequisites")
        assert phase is not None
        assert "sqlite_vec" in phase.checks


class TestBackendSelection:
    """Tests for backend selection APIs."""

    def test_select_sqlite_backend(self, client, setup_state):
        """Should save SQLite backend selection."""
        response = client.post(
            "/api/setup/backend/sqlite",
            json={"sqlite_path": "/custom/path/reverie.db"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify config was saved
        config = setup_state.get_backend_config()
        assert config["backend"] == "sqlite"
        assert config["sqlite_path"] == "/custom/path/reverie.db"

    def test_select_sqlite_backend_default_path(self, client, setup_state):
        """Should use default path for SQLite backend."""
        response = client.post("/api/setup/backend/sqlite", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        config = setup_state.get_backend_config()
        assert config["backend"] == "sqlite"
        assert config["sqlite_path"] == "~/.reverie/reverie.db"

    def test_select_postgres_backend(self, client, setup_state):
        """Should save PostgreSQL backend selection."""
        response = client.post("/api/setup/backend/postgres", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        config = setup_state.get_backend_config()
        assert config["backend"] == "postgres"

    def test_backend_selection_updates_phase(self, client, setup_state):
        """Backend selection should update database phase."""
        response = client.post("/api/setup/backend/sqlite", json={})

        assert response.status_code == 200

        # Phase config should have backend info
        phase = setup_state.config.phases.get("database")
        assert phase is not None
        assert phase.config.get("backend") == "sqlite"
