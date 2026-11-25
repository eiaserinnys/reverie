"""Tests for configuration module."""

import pytest
from pydantic import ValidationError

from reverie.config import Settings, get_settings


class TestSettings:
    """Test Settings class."""

    def test_default_values(self, monkeypatch):
        """Optional settings should have correct defaults.

        Note: All DB-related fields and cors_origin are now required
        and must be explicitly set via .env file.
        """
        monkeypatch.delenv("REVERIE_DB_NAME", raising=False)
        settings = Settings(
            _env_file=None,  # Don't load .env in tests
            db_host="localhost",  # Required field
            db_port=5432,  # Required field
            db_user="testuser",  # Required field
            db_password="testpass",  # Required field
            db_name="reverie",  # Required field
            cors_origin="*",  # Required field
        )
        # Verify required fields are set as provided
        assert settings.db_host == "localhost"
        assert settings.db_port == 5432
        assert settings.db_user == "testuser"
        assert settings.db_name == "reverie"
        assert settings.cors_origin == "*"
        # Verify optional fields have defaults
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.embedding_dimensions == 1536
        assert settings.chunk_token_limit == 2000
        assert settings.synonym_threshold == 0.8
        assert settings.ppr_damping == 0.85
        assert settings.ppr_top_k == 10
        assert settings.recall_max_passages == 5

    def test_database_url(self):
        """Database URL should be constructed correctly."""
        settings = Settings(
            _env_file=None,
            db_host="testhost",
            db_port=5433,
            db_user="testuser",
            db_password="testpass",
            db_name="testdb",
            cors_origin="*",  # Required field
        )
        assert settings.database_url == "postgresql://testuser:testpass@testhost:5433/testdb"

    def test_env_override(self, monkeypatch):
        """Environment variables should override defaults."""
        monkeypatch.setenv("REVERIE_DB_HOST", "customhost")
        monkeypatch.setenv("REVERIE_DB_PORT", "5433")
        monkeypatch.setenv("REVERIE_CHUNK_TOKEN_LIMIT", "3000")

        settings = Settings(
            _env_file=None,
            db_user="testuser",  # Required field
            db_password="testpass",  # Required field
            db_name="testdb",  # Required field
            cors_origin="*",  # Required field
        )
        assert settings.db_host == "customhost"
        assert settings.db_port == 5433
        assert settings.chunk_token_limit == 3000

    def test_get_settings(self):
        """get_settings() should return a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_setup_mode_allows_defaults(self):
        """In setup mode, DB fields should have defaults to allow bootstrap.

        The Settings class uses dynamic defaults based on _SETUP_COMPLETE flag.
        When setup is not complete, fields have defaults to allow the server
        to start and show the Setup Wizard. After setup, the model_validator
        enforces that required fields are properly configured.
        """
        # In test environment, setup is typically not complete
        # so fields should have defaults (not required)
        from reverie.config.settings import _SETUP_COMPLETE

        fields = Settings.model_fields

        if _SETUP_COMPLETE:
            # After setup: fields should be required (no defaults)
            assert fields["db_host"].is_required(), "db_host should be required after setup"
            assert fields["db_user"].is_required(), "db_user should be required after setup"
            assert fields["db_password"].is_required(), "db_password should be required after setup"
        else:
            # During setup: fields have defaults to allow bootstrap
            assert not fields["db_host"].is_required(), "db_host should have default in setup mode"
            assert not fields["db_user"].is_required(), "db_user should have default in setup mode"
            # Port and name always have sensible defaults
            assert not fields["db_port"].is_required(), "db_port should have default"
            assert not fields["db_name"].is_required(), "db_name should have default"
