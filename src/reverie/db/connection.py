"""Database connection management for reverie.

Provides sync and async connection pools for PostgreSQL.
"""

import threading
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.pool

from reverie.config import get_settings


_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Get or create the connection pool.

    Thread-safe: Uses _pool_lock to prevent race conditions during
    pool initialization when multiple threads call this simultaneously.

    Note: ThreadedConnectionPool itself is thread-safe for getconn/putconn,
    but the initialization needs protection.
    """
    global _pool
    # Quick check without lock for performance
    if _pool is not None and not _pool.closed:
        return _pool

    with _pool_lock:
        # Double-check after acquiring lock
        if _pool is None or _pool.closed:
            settings = get_settings()
            _pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=15,
                host=settings.db_host,
                port=settings.db_port,
                user=settings.db_user,
                password=settings.db_password,
                dbname=settings.db_name,
            )
        return _pool


@contextmanager
def get_connection() -> Generator:
    """Get a database connection from the pool.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def health_check() -> dict:
    """Check database connectivity and return status info."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]

                # Check pgvector extension
                cur.execute(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                )
                pgvector_row = cur.fetchone()
                pgvector_version = pgvector_row[0] if pgvector_row else None

                # Check tables
                cur.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                tables = [row[0] for row in cur.fetchall()]

        return {
            "status": "ok",
            "postgres_version": version,
            "pgvector_version": pgvector_version,
            "tables": tables,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def close_pool() -> None:
    """Close the connection pool.

    Thread-safe: Uses _pool_lock to prevent race conditions.
    """
    global _pool
    with _pool_lock:
        if _pool is not None and not _pool.closed:
            _pool.closeall()
            _pool = None
