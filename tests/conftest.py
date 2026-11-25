"""Shared test fixtures for reverie tests.

CRITICAL: This module forces all tests to use a separate test database
(`reverie_test`) to prevent accidental production data loss.
The env override MUST happen before any src imports.
"""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env file from project root (for DB credentials etc.)
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# SAFETY: Force test database name BEFORE any reverie.config import.
# ---------------------------------------------------------------------------
_PRODUCTION_DB = "reverie"
_TEST_DB = "reverie_test"

os.environ["REVERIE_DB_NAME"] = _TEST_DB

# DB credentials: must be set via environment variables (no defaults for security)

# Disable API cache during tests to prevent cross-test contamination
os.environ["REVERIE_CACHE_ENABLED"] = "false"

# Set up required LLM model settings for tests
os.environ["REVERIE_OPENIE_MODEL"] = "gpt-4.1-mini"
os.environ["REVERIE_RECOGNITION_MODEL"] = "gpt-4.1-mini"
os.environ["REVERIE_PREPROCESSOR_MODEL"] = "gpt-4.1-mini"

# Set up test profiles (required by profiles.py)
_TEST_PROFILES = {
    "test": {
        "id": "test",
        "name": "Test Profile",
        "sessions_dir": "~/.claude/projects/test/",
    }
}
os.environ["REVERIE_PROFILES"] = json.dumps(_TEST_PROFILES)

# Set CORS origin for tests
os.environ["REVERIE_CORS_ORIGIN"] = "*"


@pytest.fixture(autouse=True, scope="session")
def _ensure_test_database():
    """Create the test database and run migrations if needed.

    Connects to the 'postgres' database to CREATE DATABASE (if missing),
    then runs the migration system against the test database.
    """
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    from reverie.config import get_settings

    settings = get_settings()

    # Abort if somehow pointing at production
    if settings.db_name == _PRODUCTION_DB:
        pytest.exit(
            f"SAFETY ABORT: tests would run against production DB "
            f"'{_PRODUCTION_DB}'. This should never happen.",
            returncode=1,
        )

    # Create test database if it does not exist
    admin_conn = psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        dbname="postgres",
    )
    admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (_TEST_DB,)
        )
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{_TEST_DB}"')
    admin_conn.close()

    # Enable pgvector extension in test database
    test_conn = psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        dbname=_TEST_DB,
    )
    test_conn.autocommit = True
    with test_conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    test_conn.close()

    # Run migrations on the test database
    from reverie.db.migrate import run_migrations

    run_migrations()
