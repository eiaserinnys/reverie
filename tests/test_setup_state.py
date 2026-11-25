"""Tests for setup state management.

Tests the SetupState class which handles:
- Reading/writing .reverie/setup.json
- Managing .env file generation
- Tracking setup progress across phases
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from reverie.config.setup_state import (
    SetupState,
    PhaseStatus,
    SetupPhase,
    SetupConfig,
)


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def setup_state(temp_project_dir):
    """Create a SetupState instance with a temporary project directory."""
    return SetupState(project_dir=temp_project_dir)


class TestSetupStateInit:
    """Tests for SetupState initialization."""

    def test_creates_reverie_dir_if_not_exists(self, temp_project_dir):
        """SetupState should create .reverie directory if it doesn't exist."""
        state = SetupState(project_dir=temp_project_dir)
        assert (temp_project_dir / ".reverie").exists()

    def test_loads_existing_config(self, temp_project_dir):
        """SetupState should load existing setup.json."""
        reverie_dir = temp_project_dir / ".reverie"
        reverie_dir.mkdir()

        config = {
            "setup_complete": True,
            "phases": {
                "prerequisites": {"status": "complete"},
            },
            "last_updated": "2026-01-01T00:00:00",
        }
        (reverie_dir / "setup.json").write_text(json.dumps(config))

        state = SetupState(project_dir=temp_project_dir)
        assert state.config.setup_complete is True
        assert state.config.phases["prerequisites"].status == PhaseStatus.COMPLETE

    def test_creates_default_config_if_not_exists(self, temp_project_dir):
        """SetupState should create default config if setup.json doesn't exist."""
        state = SetupState(project_dir=temp_project_dir)

        assert state.config.setup_complete is False
        # Prerequisites phase was removed - now starts with database
        assert "database" in state.config.phases
        assert state.config.phases["database"].status == PhaseStatus.PENDING


class TestSetupStatePhases:
    """Tests for phase management."""

    def test_update_phase_status(self, setup_state):
        """Should update phase status and persist."""
        setup_state.update_phase("database", PhaseStatus.IN_PROGRESS)

        assert setup_state.config.phases["database"].status == PhaseStatus.IN_PROGRESS

        # Verify persistence
        reloaded = SetupState(project_dir=setup_state.project_dir)
        assert reloaded.config.phases["database"].status == PhaseStatus.IN_PROGRESS

    def test_update_phase_checks(self, setup_state):
        """Should update phase check results."""
        setup_state.update_phase_check("database", "postgres", True, "Connected")

        phase = setup_state.config.phases["database"]
        assert phase.checks["postgres"]["success"] is True
        assert phase.checks["postgres"]["message"] == "Connected"

    def test_get_phase_status(self, setup_state):
        """Should return phase status."""
        setup_state.update_phase("apikeys", PhaseStatus.COMPLETE)

        status = setup_state.get_phase_status("apikeys")
        assert status == PhaseStatus.COMPLETE

    def test_get_phase_status_unknown_phase(self, setup_state):
        """Should return PENDING for unknown phase."""
        status = setup_state.get_phase_status("unknown_phase")
        assert status == PhaseStatus.PENDING


class TestSetupStateCompletion:
    """Tests for setup completion."""

    def test_mark_complete(self, setup_state):
        """Should mark setup as complete."""
        setup_state.mark_complete()

        assert setup_state.config.setup_complete is True

        # Verify persistence
        reloaded = SetupState(project_dir=setup_state.project_dir)
        assert reloaded.config.setup_complete is True

    def test_is_complete(self, setup_state):
        """Should return completion status."""
        assert setup_state.is_complete() is False

        setup_state.mark_complete()
        assert setup_state.is_complete() is True


class TestEnvFileGeneration:
    """Tests for .env file generation."""

    def test_generate_env_file(self, temp_project_dir):
        """Should generate .env file with database settings."""
        state = SetupState(project_dir=temp_project_dir)

        state.save_database_config(
            host="localhost",
            port=5432,
            user="reverie_user",
            password="secret123",
            name="reverie_db",
        )

        env_path = temp_project_dir / ".env"
        assert env_path.exists()

        content = env_path.read_text()
        assert "REVERIE_DB_HOST=localhost" in content
        assert "REVERIE_DB_PORT=5432" in content
        assert "REVERIE_DB_USER=reverie_user" in content
        assert "REVERIE_DB_PASSWORD=secret123" in content
        assert "REVERIE_DB_NAME=reverie_db" in content

    def test_save_api_key(self, temp_project_dir):
        """Should save API key to .env file."""
        state = SetupState(project_dir=temp_project_dir)

        state.save_api_key("sk-test-key-12345")

        env_path = temp_project_dir / ".env"
        content = env_path.read_text()
        assert "REVERIE_OPENAI_API_KEY=sk-test-key-12345" in content

    def test_preserves_existing_env_values(self, temp_project_dir):
        """Should preserve existing .env values when updating."""
        env_path = temp_project_dir / ".env"
        env_path.write_text("EXISTING_VAR=value\nREVERIE_DB_HOST=old_host\n")

        state = SetupState(project_dir=temp_project_dir)
        state.save_database_config(
            host="new_host",
            port=5432,
            user="user",
            password="pass",
            name="db",
        )

        content = env_path.read_text()
        assert "EXISTING_VAR=value" in content
        assert "REVERIE_DB_HOST=new_host" in content
        assert "old_host" not in content


class TestSetupStateToDict:
    """Tests for serialization."""

    def test_to_dict_returns_full_state(self, setup_state):
        """Should return complete state as dictionary."""
        setup_state.update_phase("database", PhaseStatus.COMPLETE)
        setup_state.update_phase_check("prerequisites", "postgres", True)

        result = setup_state.to_dict()

        assert "setup_complete" in result
        assert "phases" in result
        assert "last_updated" in result
        assert result["phases"]["database"]["status"] == "complete"


class TestDatabaseConfig:
    """Tests for database configuration."""

    def test_get_database_config_from_env(self, temp_project_dir):
        """Should read database config from .env file."""
        env_path = temp_project_dir / ".env"
        env_path.write_text("""
REVERIE_DB_HOST=myhost
REVERIE_DB_PORT=5433
REVERIE_DB_USER=myuser
REVERIE_DB_PASSWORD=mypass
REVERIE_DB_NAME=mydb
""")

        state = SetupState(project_dir=temp_project_dir)
        config = state.get_database_config()

        assert config["host"] == "myhost"
        assert config["port"] == 5433
        assert config["user"] == "myuser"
        assert config["password"] == "mypass"
        assert config["name"] == "mydb"

    def test_get_database_config_defaults(self, temp_project_dir):
        """Should return defaults when .env doesn't exist."""
        state = SetupState(project_dir=temp_project_dir)
        config = state.get_database_config()

        assert config["host"] == "localhost"
        assert config["port"] == 5432
        assert config["user"] == ""
        assert config["password"] == ""
        assert config["name"] == "reverie"


class TestBackendConfig:
    """Tests for database backend configuration."""

    def test_save_backend_config_postgres(self, temp_project_dir):
        """Should save PostgreSQL backend config to .env file."""
        state = SetupState(project_dir=temp_project_dir)

        state.save_backend_config(backend="postgres")

        env_path = temp_project_dir / ".env"
        content = env_path.read_text()
        assert "REVERIE_DB_BACKEND=postgres" in content

        # Also check phase config was updated
        phase = state.config.phases.get("database")
        assert phase is not None
        assert phase.config.get("backend") == "postgres"

    def test_save_backend_config_sqlite(self, temp_project_dir):
        """Should save SQLite backend config to .env file."""
        state = SetupState(project_dir=temp_project_dir)

        state.save_backend_config(
            backend="sqlite",
            sqlite_path="/custom/path/reverie.db",
        )

        env_path = temp_project_dir / ".env"
        content = env_path.read_text()
        assert "REVERIE_DB_BACKEND=sqlite" in content
        assert "REVERIE_SQLITE_PATH=/custom/path/reverie.db" in content

        # Check phase config
        phase = state.config.phases.get("database")
        assert phase.config.get("backend") == "sqlite"
        assert phase.config.get("sqlite_path") == "/custom/path/reverie.db"

    def test_save_backend_config_sqlite_default_path(self, temp_project_dir):
        """Should use default SQLite path if not provided."""
        state = SetupState(project_dir=temp_project_dir)

        state.save_backend_config(backend="sqlite")

        env_path = temp_project_dir / ".env"
        content = env_path.read_text()
        assert "REVERIE_DB_BACKEND=sqlite" in content
        assert "REVERIE_SQLITE_PATH=~/.reverie/reverie.db" in content

    def test_get_backend_config(self, temp_project_dir):
        """Should read backend config from .env file."""
        env_path = temp_project_dir / ".env"
        env_path.write_text("""
REVERIE_DB_BACKEND=sqlite
REVERIE_SQLITE_PATH=/data/reverie.db
""")

        state = SetupState(project_dir=temp_project_dir)
        config = state.get_backend_config()

        assert config["backend"] == "sqlite"
        assert config["sqlite_path"] == "/data/reverie.db"

    def test_get_backend_config_defaults(self, temp_project_dir):
        """Should return None as default backend (not configured)."""
        state = SetupState(project_dir=temp_project_dir)
        config = state.get_backend_config()

        # Backend is None when not yet configured (forces user to select)
        assert config["backend"] is None
        assert config["sqlite_path"] == "~/.reverie/reverie.db"

    def test_save_backend_preserves_existing_env(self, temp_project_dir):
        """Should preserve existing .env values when saving backend."""
        env_path = temp_project_dir / ".env"
        env_path.write_text("REVERIE_OPENAI_API_KEY=sk-test\n")

        state = SetupState(project_dir=temp_project_dir)
        state.save_backend_config(backend="sqlite")

        content = env_path.read_text()
        assert "REVERIE_OPENAI_API_KEY=sk-test" in content
        assert "REVERIE_DB_BACKEND=sqlite" in content
