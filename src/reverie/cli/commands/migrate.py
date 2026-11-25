"""Migrate command - Run database migrations."""

import sys
from pathlib import Path

import click
import psycopg2

from reverie.config.settings import get_settings


def get_migrations_dir() -> Path:
    """Get migrations directory path."""
    # migrations/ is at project root
    # This file is at src/reverie/cli/commands/migrate.py
    # Go up 5 levels: commands -> cli -> reverie -> src -> project_root
    return Path(__file__).parent.parent.parent.parent.parent / "migrations"


def get_db_url() -> str:
    """Get database URL from settings."""
    settings = get_settings()
    return (
        f"postgresql://{settings.db_user}:{settings.db_password}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )


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


def get_pending_migrations(applied: set[str], migrations_dir: Path) -> list[Path]:
    """Get sorted list of migration files not yet applied."""
    all_migrations = sorted(migrations_dir.glob("*.sql"))
    return [m for m in all_migrations if m.name not in applied]


def apply_migration(conn, migration_path: Path) -> None:
    """Apply a single migration file."""
    sql = migration_path.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
        cur.execute(
            "INSERT INTO schema_migrations (filename) VALUES (%s)",
            (migration_path.name,),
        )
    conn.commit()


def run_migrate(dry_run: bool = False) -> None:
    """Run all pending migrations."""
    migrations_dir = get_migrations_dir()

    if not migrations_dir.exists():
        click.echo(f"Error: Migrations directory not found: {migrations_dir}", err=True)
        sys.exit(1)

    db_url = get_db_url()
    settings = get_settings()
    click.echo(f"Connecting to database '{settings.db_name}'...")

    try:
        conn = psycopg2.connect(db_url)
    except psycopg2.OperationalError as e:
        click.echo(f"Error: Could not connect to database: {e}", err=True)
        sys.exit(1)

    try:
        ensure_migration_table(conn)
        applied = get_applied_migrations(conn)
        pending = get_pending_migrations(applied, migrations_dir)

        if not pending:
            click.echo("No pending migrations.")
            return

        click.echo(f"Found {len(pending)} pending migration(s):")
        for migration_path in pending:
            click.echo(f"  - {migration_path.name}")

        if dry_run:
            click.echo("\n[Dry run] No changes applied.")
            return

        click.echo()
        for migration_path in pending:
            click.echo(f"Applying: {migration_path.name}...", nl=False)
            try:
                apply_migration(conn, migration_path)
                click.echo(" OK")
            except Exception as e:
                conn.rollback()
                click.echo(f" FAILED")
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)

        click.echo(f"\nDone. Applied {len(pending)} migration(s).")

    finally:
        conn.close()
