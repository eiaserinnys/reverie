"""Tests for Docker compose template generation."""

import tempfile
from pathlib import Path

import pytest

from reverie.docker.compose_template import (
    COMPOSE_TEMPLATE,
    compose_exists,
    generate_compose_file,
    get_compose_path,
    get_local_dir,
)


class TestGetLocalDir:
    """Tests for get_local_dir function."""

    def test_with_project_dir(self):
        """Should return .local in provided project dir."""
        project_dir = Path("/tmp/test-project")
        result = get_local_dir(project_dir)
        assert result == Path("/tmp/test-project/.local")

    def test_with_string_path(self):
        """Should accept string path."""
        result = get_local_dir("/tmp/test-project")
        assert result == Path("/tmp/test-project/.local")

    def test_without_project_dir(self):
        """Should use default path when no project_dir given."""
        result = get_local_dir(None)
        # Should be relative to compose_template.py
        assert result.name == ".local"


class TestGenerateComposeFile:
    """Tests for generate_compose_file function."""

    def test_creates_directory_and_file(self):
        """Should create .local directory and docker-compose.yml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compose_path = generate_compose_file(project_dir=tmpdir)

            assert compose_path.exists()
            assert compose_path.name == "docker-compose.yml"
            assert compose_path.parent.name == ".local"

    def test_creates_gitignore(self):
        """Should create .gitignore in .local directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_compose_file(project_dir=tmpdir)

            gitignore = Path(tmpdir) / ".local" / ".gitignore"
            assert gitignore.exists()
            assert "postgres-data/" in gitignore.read_text()

    def test_custom_postgres_settings(self):
        """Should use custom PostgreSQL settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compose_path = generate_compose_file(
                project_dir=tmpdir,
                postgres_user="testuser",
                postgres_password="testpass",
                postgres_db="testdb",
                postgres_port=5433,
            )

            content = compose_path.read_text()
            assert "POSTGRES_USER: testuser" in content
            assert "POSTGRES_PASSWORD: testpass" in content
            assert "POSTGRES_DB: testdb" in content
            assert '"5433:5432"' in content

    def test_default_postgres_settings(self):
        """Should use default PostgreSQL settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compose_path = generate_compose_file(project_dir=tmpdir)

            content = compose_path.read_text()
            assert "POSTGRES_USER: reverie" in content
            assert "POSTGRES_PASSWORD: reverie" in content
            assert "POSTGRES_DB: reverie" in content
            assert '"5432:5432"' in content


class TestGetComposePath:
    """Tests for get_compose_path function."""

    def test_returns_expected_path(self):
        """Should return path to docker-compose.yml."""
        result = get_compose_path("/tmp/test-project")
        assert result == Path("/tmp/test-project/.local/docker-compose.yml")


class TestComposeExists:
    """Tests for compose_exists function."""

    def test_returns_false_when_not_exists(self):
        """Should return False when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert compose_exists(tmpdir) is False

    def test_returns_true_when_exists(self):
        """Should return True when file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_compose_file(project_dir=tmpdir)
            assert compose_exists(tmpdir) is True


class TestComposeTemplate:
    """Tests for the compose template content."""

    def test_uses_pgvector_image(self):
        """Should use pgvector/pgvector:pg16 image."""
        assert "pgvector/pgvector:pg16" in COMPOSE_TEMPLATE

    def test_has_healthcheck(self):
        """Should include healthcheck configuration."""
        assert "healthcheck:" in COMPOSE_TEMPLATE
        assert "pg_isready" in COMPOSE_TEMPLATE

    def test_has_volume_mount(self):
        """Should mount postgres-data volume."""
        assert "./postgres-data:/var/lib/postgresql/data" in COMPOSE_TEMPLATE

    def test_has_restart_policy(self):
        """Should have restart unless-stopped policy."""
        assert "restart: unless-stopped" in COMPOSE_TEMPLATE
