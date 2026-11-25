"""Tests for database connection module."""

import pytest

from reverie.db.connection import get_connection, health_check, close_pool


class TestDBConnection:
    """Test database connection."""

    def teardown_method(self):
        """Close pool after each test."""
        close_pool()

    def test_connection_success(self):
        """Should connect to reverie database."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                assert result[0] == 1

    def test_health_check(self):
        """Health check should return status info."""
        result = health_check()
        assert result["status"] == "ok"
        assert "PostgreSQL" in result["postgres_version"]
        # pgvector may or may not be installed yet
        assert "pgvector_version" in result
        assert isinstance(result["tables"], list)

    def test_connection_context_manager_commits(self):
        """Context manager should commit on success."""
        # Create a temp table, insert, verify in separate connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS _test_cm (id SERIAL, val TEXT)"
                )

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO _test_cm (val) VALUES ('test')")

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT val FROM _test_cm WHERE val = 'test'")
                result = cur.fetchone()
                assert result is not None
                assert result[0] == "test"
                # Cleanup
                cur.execute("DROP TABLE IF EXISTS _test_cm")

    def test_connection_context_manager_rollbacks(self):
        """Context manager should rollback on exception."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS _test_rb (id SERIAL, val TEXT)"
                )

        with pytest.raises(Exception):
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO _test_rb (val) VALUES ('should_rollback')")
                    raise Exception("Intentional error")

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT val FROM _test_rb WHERE val = 'should_rollback'"
                )
                result = cur.fetchone()
                assert result is None
                # Cleanup
                cur.execute("DROP TABLE IF EXISTS _test_rb")
