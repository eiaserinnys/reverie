"""Init-db command - Create database and run migrations."""

import sys

import click
import psycopg2
from psycopg2 import sql

from reverie.config.settings import get_settings
from reverie.cli.commands.migrate import run_migrate


def get_admin_connection():
    """Get connection to postgres database for admin operations.

    Returns connection in autocommit mode for DDL operations.
    """
    settings = get_settings()
    conn = psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        dbname="postgres",
    )
    conn.autocommit = True
    return conn


def database_exists(conn, db_name: str) -> bool:
    """Check if database exists."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,),
        )
        return cur.fetchone() is not None


def create_database(conn, db_name: str) -> None:
    """Create database."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
        )


def drop_database(conn, db_name: str) -> None:
    """Drop database."""
    with conn.cursor() as cur:
        # Terminate existing connections
        cur.execute(
            """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid()
            """,
            (db_name,),
        )
        cur.execute(
            sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name))
        )


def run_init_db(dry_run: bool = False, force: bool = False) -> None:
    """Initialize database: create if needed, then run migrations."""
    settings = get_settings()
    db_name = settings.db_name

    click.echo(f"Initializing database '{db_name}'...")
    click.echo(f"  Host: {settings.db_host}:{settings.db_port}")
    click.echo(f"  User: {settings.db_user}")

    try:
        conn = get_admin_connection()
    except psycopg2.OperationalError as e:
        click.echo(f"Error: Could not connect to PostgreSQL: {e}", err=True)
        sys.exit(1)

    try:
        exists = database_exists(conn, db_name)

        if exists and force:
            if dry_run:
                click.echo(f"\n[Dry run] Would drop existing database '{db_name}'")
                click.echo(f"[Dry run] Would create database '{db_name}'")
                click.echo(f"[Dry run] Would run migrations")
                return

            click.echo(f"\nDropping existing database '{db_name}'...")
            drop_database(conn, db_name)
            exists = False

        if exists:
            click.echo(f"\nDatabase '{db_name}' already exists.")
            if dry_run:
                click.echo("[Dry run] Would check and run pending migrations")
                return
        else:
            if dry_run:
                click.echo(f"\n[Dry run] Would create database '{db_name}'")
                click.echo(f"[Dry run] Would run migrations")
                return

            click.echo(f"\nCreating database '{db_name}'...")
            create_database(conn, db_name)
            click.echo("Database created successfully.")

    finally:
        conn.close()

    # Run migrations
    click.echo("\nRunning migrations...")
    run_migrate(dry_run=dry_run)

    click.echo(f"\nDatabase '{db_name}' is ready.")
