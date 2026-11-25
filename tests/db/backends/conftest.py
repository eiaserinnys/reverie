"""Fixtures for dual backend database tests.

Provides both PostgreSQL and SQLite database fixtures for testing
ReverieDB implementations with the same test cases.
"""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def postgres_db():
    """Create a PostgreSQL ReverieDB instance for testing.

    Uses the test database (reverie_test) configured in the main conftest.
    """
    from reverie.db.factory import create_db
    from reverie.config import get_settings

    settings = get_settings()

    db = create_db(
        backend="postgres",
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        dbname=settings.db_name,
    )

    yield db

    db.rollback()
    db.close()


@pytest.fixture
def sqlite_db():
    """Create a SQLite ReverieDB instance for testing.

    Uses an in-memory database for speed and isolation.
    """
    from reverie.db.factory import create_db

    # Use a temporary file for SQLite
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = create_db(
            backend="sqlite",
            db_path=db_path,
            run_migrations=True,
        )

        yield db

        db.close()
    finally:
        # Clean up temporary file
        Path(db_path).unlink(missing_ok=True)


@pytest.fixture(params=["postgres", "sqlite"])
def reverie_db(request, postgres_db, sqlite_db):
    """Parametrized fixture that provides both database backends.

    Use this fixture to run the same test against both PostgreSQL and SQLite.

    Example:
        def test_insert_entity(reverie_db):
            entity_id = reverie_db.upsert_entity("test")
            assert entity_id > 0
    """
    if request.param == "postgres":
        return postgres_db
    else:
        return sqlite_db


@pytest.fixture
def sample_embedding():
    """Generate a sample 1536-dimensional embedding vector."""
    import random
    random.seed(42)
    return [random.random() for _ in range(1536)]


@pytest.fixture
def sample_embedding_2():
    """Generate another sample 1536-dimensional embedding vector."""
    import random
    random.seed(123)
    return [random.random() for _ in range(1536)]
