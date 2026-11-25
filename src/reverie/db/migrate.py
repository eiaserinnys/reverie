"""Database migration runner for reverie.

Usage:
    python -m reverie.db.migrate [--db-url DATABASE_URL]
"""

import logging
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# Migrations are at project root, not in src/reverie/db/
# Navigate from src/reverie/db/ up to project root (4 levels)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MIGRATIONS_DIR = PROJECT_ROOT / "migrations"

# Load .env file from project root (same location as settings.py uses)
load_dotenv(PROJECT_ROOT / ".env")


def get_db_url() -> str:
    """Get database URL from environment or argument."""
    if len(sys.argv) > 2 and sys.argv[1] == "--db-url":
        return sys.argv[2]

    url = os.environ.get("REVERIE_DATABASE_URL")
    if url:
        return url

    host = os.environ.get("REVERIE_DB_HOST")
    port = os.environ.get("REVERIE_DB_PORT")
    user = os.environ.get("REVERIE_DB_USER")
    password = os.environ.get("REVERIE_DB_PASSWORD")
    dbname = os.environ.get("REVERIE_DB_NAME")

    if not all([host, port, user, password, dbname]):
        raise ValueError(
            "REVERIE_DB_HOST, REVERIE_DB_PORT, REVERIE_DB_USER, REVERIE_DB_PASSWORD, and REVERIE_DB_NAME must be set in .env"
        )

    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def ensure_migration_table(conn) -> None:
    """Create migration tracking table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
    conn.commit()


def get_applied_migrations(conn) -> set[str]:
    """Get set of already-applied migration filenames."""
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM schema_migrations ORDER BY filename")
        return {row[0] for row in cur.fetchall()}


def get_pending_migrations(applied: set[str]) -> list[Path]:
    """Get sorted list of migration files not yet applied."""
    all_migrations = sorted(MIGRATIONS_DIR.glob("*.sql"))
    return [m for m in all_migrations if m.name not in applied]


def apply_migration(conn, migration_path: Path) -> None:
    """Apply a single migration file."""
    sql = migration_path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
        cur.execute(
            "INSERT INTO schema_migrations (filename) VALUES (%s)",
            (migration_path.name,),
        )
    conn.commit()


def run_migrations() -> None:
    """Run all pending migrations."""
    db_url = get_db_url()
    logger.info("Connecting to database...")

    conn = psycopg2.connect(db_url)
    try:
        ensure_migration_table(conn)
        applied = get_applied_migrations(conn)
        pending = get_pending_migrations(applied)

        if not pending:
            logger.info("No pending migrations.")
            return

        for migration_path in pending:
            logger.info("Applying: %s", migration_path.name)
            try:
                apply_migration(conn, migration_path)
                logger.info("  OK")
            except Exception as e:
                conn.rollback()
                logger.error("  FAILED: %s", e)
                raise
    finally:
        conn.close()

    logger.info("Done. Applied %d migration(s).", len(pending))


if __name__ == "__main__":
    run_migrations()
