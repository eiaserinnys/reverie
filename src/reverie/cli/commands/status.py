"""Status command - Show system status."""

import sys

import click
import psycopg2

from reverie.config.settings import get_settings


def get_db_connection():
    """Get database connection."""
    settings = get_settings()
    return psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        dbname=settings.db_name,
    )


def get_migration_status(conn) -> tuple[int, int]:
    """Get migration status: (applied_count, pending_count)."""
    from reverie.cli.commands.migrate import (
        get_applied_migrations,
        get_pending_migrations,
        get_migrations_dir,
    )

    try:
        applied = get_applied_migrations(conn)
        pending = get_pending_migrations(applied, get_migrations_dir())
        return len(applied), len(pending)
    except psycopg2.errors.UndefinedTable:
        # schema_migrations table doesn't exist
        return 0, -1


def get_entity_count(conn) -> int:
    """Get count of entities."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM entities")
            return cur.fetchone()[0]
    except psycopg2.errors.UndefinedTable:
        return -1


def get_passage_count(conn) -> int:
    """Get count of passages."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM passages")
            return cur.fetchone()[0]
    except psycopg2.errors.UndefinedTable:
        return -1


def get_session_count(conn) -> int:
    """Get count of indexed sessions."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM processed_sessions WHERE status = 'completed'")
            return cur.fetchone()[0]
    except psycopg2.errors.UndefinedTable:
        return -1


def run_status() -> None:
    """Display system status."""
    settings = get_settings()

    click.echo("Reverie Status")
    click.echo("=" * 40)

    # Database connection
    click.echo(f"\nDatabase:")
    click.echo(f"  Host: {settings.db_host}:{settings.db_port}")
    click.echo(f"  Name: {settings.db_name}")
    click.echo(f"  User: {settings.db_user}")

    try:
        conn = get_db_connection()
        click.echo(f"  Status: " + click.style("Connected", fg="green"))
    except psycopg2.OperationalError as e:
        error_msg = str(e).split("\n")[0]
        click.echo(f"  Status: " + click.style("Disconnected", fg="red"))
        click.echo(f"  Error: {error_msg}")
        return

    try:
        # Migration status
        applied, pending = get_migration_status(conn)
        click.echo(f"\nMigrations:")
        click.echo(f"  Applied: {applied}")
        if pending == -1:
            click.echo(f"  Pending: " + click.style("Unknown (no migration table)", fg="yellow"))
        elif pending > 0:
            click.echo(f"  Pending: " + click.style(str(pending), fg="yellow"))
        else:
            click.echo(f"  Pending: {pending}")

        # Data counts
        entities = get_entity_count(conn)
        passages = get_passage_count(conn)
        sessions = get_session_count(conn)

        click.echo(f"\nData:")
        if entities == -1:
            click.echo(f"  Entities: " + click.style("Table not found", fg="yellow"))
        else:
            click.echo(f"  Entities: {entities:,}")

        if passages == -1:
            click.echo(f"  Passages: " + click.style("Table not found", fg="yellow"))
        else:
            click.echo(f"  Passages: {passages:,}")

        if sessions == -1:
            click.echo(f"  Sessions: " + click.style("Table not found", fg="yellow"))
        else:
            click.echo(f"  Sessions: {sessions:,}")

    finally:
        conn.close()
