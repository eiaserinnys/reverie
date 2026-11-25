"""E2E tests for Setup Wizard scenarios.

Tests three installation scenarios:
1. PostgreSQL - User has PostgreSQL installed
2. Docker - User uses Docker for PostgreSQL
3. SQLite - User falls back to SQLite

Note: These tests are integration-level E2E tests that test the API
without a real browser. For full E2E with browser, use Playwright.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import requests

from tests.e2e.conftest import SetupWizardClient


# Skip these tests by default since they require specific environment setup
pytestmark = pytest.mark.e2e


class TestSqliteScenario:
    """Scenario 3: SQLite backend (no PostgreSQL, no Docker).

    This is the simplest scenario - user just wants to get started quickly.
    """

    def test_sqlite_full_flow(self, temp_project_dir: Path):
        """Test complete SQLite setup flow without starting server.

        This test simulates the SQLite scenario by directly testing
        the setup state and routes without starting a full server.
        """
        from reverie.config.setup_state import SetupState, PhaseStatus
        import sqlite3
        import sqlite_vec

        # Step 1: Initialize setup state
        state = SetupState(project_dir=temp_project_dir)

        # Verify initial state
        assert state.is_complete() is False
        assert state.get_phase_status("prerequisites") == PhaseStatus.PENDING

        # Step 2: Check sqlite-vec availability
        # This simulates GET /api/setup/check/sqlite
        try:
            conn = sqlite3.connect(":memory:")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            cursor = conn.execute("SELECT vec_version()")
            version = cursor.fetchone()[0]
            conn.close()
            sqlite_available = True
        except Exception:
            sqlite_available = False

        assert sqlite_available is True, "sqlite-vec should be available"

        state.update_phase_check(
            "prerequisites",
            "sqlite_vec",
            True,
            f"sqlite-vec v{version}",
        )

        # Step 3: Select SQLite backend
        # This simulates POST /api/setup/backend/sqlite
        sqlite_path = str(temp_project_dir / "reverie_test.db")
        state.save_backend_config(backend="sqlite", sqlite_path=sqlite_path)

        # Verify backend was saved
        backend_config = state.get_backend_config()
        assert backend_config["backend"] == "sqlite"
        assert backend_config["sqlite_path"] == sqlite_path

        # Step 4: Verify .env was updated
        env_path = temp_project_dir / ".env"
        assert env_path.exists()
        env_content = env_path.read_text()
        assert "REVERIE_DB_BACKEND=sqlite" in env_content
        assert f"REVERIE_SQLITE_PATH={sqlite_path}" in env_content

        # Step 5: Mark database phase complete
        state.update_phase("database", PhaseStatus.COMPLETE)
        assert state.get_phase_status("database") == PhaseStatus.COMPLETE

        # Step 6: Save a fake API key (simulates POST /api/setup/apikey)
        state.save_api_key("sk-test-fake-key")
        state.update_phase("apikeys", PhaseStatus.COMPLETE)

        # Step 7: Skip profile (use defaults)
        state.update_phase("profile", PhaseStatus.COMPLETE)

        # Step 8: Run SQLite migrations
        # Create the database and run migrations
        from reverie.db.factory import create_db

        db = create_db(
            backend="sqlite",
            db_path=sqlite_path,
            run_migrations=True,
        )

        # Verify tables were created
        cursor = db.raw_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "entities" in tables
        assert "passages" in tables
        db.close()

        state.update_phase("migration", PhaseStatus.COMPLETE)

        # Step 9: Complete setup
        state.mark_complete()
        assert state.is_complete() is True

        # Verify final state
        final_state = state.to_dict()
        assert final_state["setup_complete"] is True
        assert final_state["phases"]["database"]["status"] == "complete"


class TestPostgresScenario:
    """Scenario 1: PostgreSQL backend (user has PostgreSQL installed).

    Tests the flow when user has PostgreSQL available.
    """

    def test_postgres_full_flow(self, temp_project_dir: Path):
        """Test PostgreSQL setup flow with mocked database connection."""
        from reverie.config.setup_state import SetupState, PhaseStatus

        # Step 1: Initialize setup state
        state = SetupState(project_dir=temp_project_dir)
        assert state.is_complete() is False

        # Step 2: Select PostgreSQL backend
        state.save_backend_config(backend="postgres")

        # Verify backend was saved
        backend_config = state.get_backend_config()
        assert backend_config["backend"] == "postgres"

        # Step 3: Save database configuration
        state.save_database_config(
            host="localhost",
            port=5432,
            user="reverie_test",
            password="test_password",
            name="reverie_test",
        )

        # Verify database config was saved
        db_config = state.get_database_config()
        assert db_config["host"] == "localhost"
        assert db_config["port"] == 5432
        assert db_config["user"] == "reverie_test"

        # Verify .env was updated
        env_path = temp_project_dir / ".env"
        assert env_path.exists()
        env_content = env_path.read_text()
        assert "REVERIE_DB_BACKEND=postgres" in env_content
        assert "REVERIE_DB_HOST=localhost" in env_content
        assert "REVERIE_DB_USER=reverie_test" in env_content

        # Step 4: Mark phases complete (simulating successful tests)
        state.update_phase("prerequisites", PhaseStatus.COMPLETE)
        state.update_phase("database", PhaseStatus.COMPLETE)
        state.save_api_key("sk-test-postgres-key")
        state.update_phase("apikeys", PhaseStatus.COMPLETE)
        state.update_phase("profile", PhaseStatus.COMPLETE)
        state.update_phase("migration", PhaseStatus.COMPLETE)

        # Step 5: Complete setup
        state.mark_complete()
        assert state.is_complete() is True


class TestDockerScenario:
    """Scenario 2: Docker PostgreSQL backend.

    Tests the flow when user uses Docker for PostgreSQL.
    """

    def test_docker_flow_with_mocked_docker(self, temp_project_dir: Path):
        """Test Docker PostgreSQL setup flow with mocked Docker commands."""
        from reverie.config.setup_state import SetupState, PhaseStatus

        # Step 1: Initialize setup state
        state = SetupState(project_dir=temp_project_dir)
        assert state.is_complete() is False

        # Step 2: Check Docker availability (mocked)
        # Simulating GET /api/setup/check/docker
        state.update_phase_check(
            "prerequisites",
            "docker",
            True,
            "Docker 24.0.0 available",
        )

        # Step 3: Start Docker PostgreSQL (mocked)
        # Simulating POST /api/setup/docker/start
        state.update_phase_check(
            "prerequisites",
            "docker_postgres",
            True,
            "PostgreSQL container running",
        )

        # Step 4: Select PostgreSQL backend
        state.save_backend_config(backend="postgres")

        # Step 5: Save Docker container connection info
        state.save_database_config(
            host="localhost",
            port=5432,
            user="reverie",
            password="reverie",
            name="reverie",
        )

        # Verify configuration
        db_config = state.get_database_config()
        assert db_config["host"] == "localhost"
        assert db_config["user"] == "reverie"

        # Step 6: Mark phases complete
        state.update_phase("prerequisites", PhaseStatus.COMPLETE)
        state.update_phase("database", PhaseStatus.COMPLETE)
        state.save_api_key("sk-test-docker-key")
        state.update_phase("apikeys", PhaseStatus.COMPLETE)
        state.update_phase("profile", PhaseStatus.COMPLETE)
        state.update_phase("migration", PhaseStatus.COMPLETE)

        # Step 7: Complete setup
        state.mark_complete()
        assert state.is_complete() is True

        # Verify final state
        final_state = state.to_dict()
        assert final_state["setup_complete"] is True
        assert final_state["phases"]["prerequisites"]["checks"]["docker"]["success"] is True


class TestBackendSelection:
    """Tests for backend selection logic."""

    def test_backend_defaults_to_none(self, temp_project_dir: Path):
        """Backend should be None when not set (forces user selection)."""
        from reverie.config.setup_state import SetupState

        state = SetupState(project_dir=temp_project_dir)
        config = state.get_backend_config()

        # None indicates user must select a backend
        assert config["backend"] is None

    def test_sqlite_path_defaults(self, temp_project_dir: Path):
        """SQLite path should have a sensible default."""
        from reverie.config.setup_state import SetupState

        state = SetupState(project_dir=temp_project_dir)
        state.save_backend_config(backend="sqlite")

        config = state.get_backend_config()
        assert config["sqlite_path"] == "~/.reverie/reverie.db"

    def test_can_switch_backends(self, temp_project_dir: Path):
        """Should be able to switch between backends."""
        from reverie.config.setup_state import SetupState

        state = SetupState(project_dir=temp_project_dir)

        # Start with SQLite
        state.save_backend_config(backend="sqlite")
        assert state.get_backend_config()["backend"] == "sqlite"

        # Switch to PostgreSQL
        state.save_backend_config(backend="postgres")
        assert state.get_backend_config()["backend"] == "postgres"

        # Switch back to SQLite
        state.save_backend_config(backend="sqlite", sqlite_path="/custom/path.db")
        config = state.get_backend_config()
        assert config["backend"] == "sqlite"
        assert config["sqlite_path"] == "/custom/path.db"


class TestSqliteVecCheck:
    """Tests for sqlite-vec availability checking."""

    def test_sqlite_vec_is_available(self):
        """sqlite-vec should be available in the test environment."""
        import sqlite3
        import sqlite_vec

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        cursor = conn.execute("SELECT vec_version()")
        version = cursor.fetchone()[0]
        conn.close()

        assert version is not None
        assert len(version) > 0

    def test_sqlite_vec_vector_operations(self):
        """sqlite-vec should support vector operations."""
        import sqlite3
        import struct
        import sqlite_vec

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        # Create a simple table with vector column
        conn.execute("""
            CREATE TABLE test_vectors (
                id INTEGER PRIMARY KEY,
                embedding BLOB
            )
        """)

        # Insert a vector
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        blob1 = struct.pack(f"{len(vec1)}f", *vec1)
        blob2 = struct.pack(f"{len(vec2)}f", *vec2)

        conn.execute("INSERT INTO test_vectors (embedding) VALUES (?)", (blob1,))
        conn.execute("INSERT INTO test_vectors (embedding) VALUES (?)", (blob2,))

        # Query using cosine distance
        cursor = conn.execute("""
            SELECT id, vec_distance_cosine(embedding, ?) as distance
            FROM test_vectors
            ORDER BY distance ASC
        """, (blob1,))

        results = cursor.fetchall()
        conn.close()

        assert len(results) == 2
        # First result should be the same vector (distance 0)
        assert results[0][0] == 1  # id of vec1
        assert results[0][1] < 0.01  # distance should be ~0
