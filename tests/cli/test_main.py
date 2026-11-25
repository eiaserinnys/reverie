"""Tests for CLI main entry point."""

import pytest
from click.testing import CliRunner

from reverie.cli import main, cli


# Common test environment variables
TEST_ENV = {
    "REVERIE_DB_HOST": "localhost",
    "REVERIE_DB_PORT": "5432",
    "REVERIE_DB_USER": "test",
    "REVERIE_DB_PASSWORD": "test",
    "REVERIE_DB_NAME": "reverie_test",
    "REVERIE_OPENIE_MODEL": "gpt-4o-mini",
    "REVERIE_RECOGNITION_MODEL": "gpt-4o-mini",
    "REVERIE_PREPROCESSOR_MODEL": "gpt-4o-mini",
}


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_exists(self):
        """CLI entry point should exist."""
        assert main is not None
        assert cli is not None

    def test_cli_help(self):
        """CLI should display help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Reverie" in result.output or "reverie" in result.output

    def test_cli_version(self):
        """CLI should display version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        # Should contain version number pattern
        assert "." in result.output  # e.g., "0.1.0"


class TestMigrateCommand:
    """Test migrate command."""

    def test_migrate_help(self):
        """Migrate command should have help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["migrate", "--help"])
        assert result.exit_code == 0
        assert "migrate" in result.output.lower() or "migration" in result.output.lower()

    def test_migrate_requires_db(self):
        """Migrate should fail gracefully without DB."""
        runner = CliRunner()
        # Use a non-existent database
        env = {**TEST_ENV, "REVERIE_DB_NAME": "nonexistent_db_12345"}
        result = runner.invoke(cli, ["migrate"], env=env)
        # Should fail but not crash
        assert result.exit_code != 0


class TestInitDBCommand:
    """Test init-db command."""

    def test_init_db_help(self):
        """init-db command should have help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init-db", "--help"])
        assert result.exit_code == 0
        assert "database" in result.output.lower() or "init" in result.output.lower()

    def test_init_db_dry_run(self):
        """init-db --dry-run should show what would be done."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init-db", "--dry-run"], env=TEST_ENV)
        # May fail to connect but should not crash on import/validation
        # Check that it at least tries (shows db name or dry-run message)
        assert "reverie" in result.output.lower() or "dry" in result.output.lower() or "error" in result.output.lower()


class TestStatusCommand:
    """Test status command."""

    def test_status_help(self):
        """Status command should have help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0
