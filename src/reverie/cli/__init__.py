"""Reverie CLI - Command Line Interface for Reverie Memory System."""

import click

try:
    from reverie._version import __version__
except ImportError:
    # Fallback when setuptools-scm hasn't generated _version.py
    __version__ = "0.0.0-dev"


@click.group()
@click.version_option(version=__version__, prog_name="reverie")
def cli():
    """Reverie - Memory retrieval system for Claude Code sessions.

    A semantic memory system that indexes Claude Code conversations
    and retrieves relevant context using the MemoryLink v2 algorithm.
    """
    pass


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")
def migrate(dry_run: bool):
    """Run database migrations.

    Applies pending SQL migrations to the configured database.
    Requires the database to already exist.
    """
    from reverie.cli.commands.migrate import run_migrate
    run_migrate(dry_run=dry_run)


@cli.command("init-db")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")
@click.option("--force", is_flag=True, help="Drop and recreate if database exists")
def init_db(dry_run: bool, force: bool):
    """Initialize the database.

    Creates the database if it doesn't exist, then runs all migrations.
    Use this for first-time setup or to reset the database with --force.
    """
    from reverie.cli.commands.init_db import run_init_db
    run_init_db(dry_run=dry_run, force=force)


@cli.command()
def status():
    """Show system status.

    Displays database connection status, migration state,
    indexed sessions count, and other system information.
    """
    from reverie.cli.commands.status import run_status
    run_status()


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
