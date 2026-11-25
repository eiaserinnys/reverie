"""Tests for profile configuration."""

import json
import os
from pathlib import Path

import pytest

from reverie.config.profiles import (
    ProfileConfig,
    clear_profile_cache,
    get_default_profile,
    get_profile,
    get_profiles,
)


# Test profile data
TEST_PROFILES = {
    "main": {
        "id": "main",
        "name": "Main Agent",
        "sessions_dir": "~/.claude/projects/main/",
    },
    "secondary": {
        "id": "secondary",
        "name": "Secondary Agent",
        "sessions_dir": "~/.claude/projects/secondary/",
    },
}


class TestProfileConfig:
    """Tests for ProfileConfig dataclass."""

    def test_create_valid_profile(self):
        """Test creating a valid ProfileConfig."""
        profile = ProfileConfig(
            id="test",
            name="Test Profile",
            sessions_dir="~/.claude/test/",
        )
        assert profile.id == "test"
        assert profile.name == "Test Profile"
        assert profile.sessions_dir == "~/.claude/test/"

    def test_sessions_path_expands_tilde(self):
        """Test that sessions_path expands ~ to home directory."""
        profile = ProfileConfig(
            id="test",
            name="Test",
            sessions_dir="~/some/path/",
        )
        assert profile.sessions_path == Path.home() / "some" / "path"

    def test_sessions_path_absolute(self):
        """Test that sessions_path works with absolute paths."""
        profile = ProfileConfig(
            id="test",
            name="Test",
            sessions_dir="/absolute/path/",
        )
        assert profile.sessions_path == Path("/absolute/path/")

    def test_empty_id_raises_error(self):
        """Test that empty id raises ValueError."""
        with pytest.raises(ValueError, match="Profile id cannot be empty"):
            ProfileConfig(id="", name="Test", sessions_dir="/path/")

    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Profile name cannot be empty"):
            ProfileConfig(id="test", name="", sessions_dir="/path/")

    def test_empty_sessions_dir_raises_error(self):
        """Test that empty sessions_dir raises ValueError."""
        with pytest.raises(ValueError, match="Profile sessions_dir cannot be empty"):
            ProfileConfig(id="test", name="Test", sessions_dir="")


class TestGetProfiles:
    """Tests for get_profiles function."""

    def setup_method(self):
        """Clear cache and set env before each test."""
        clear_profile_cache()
        # Clean up any existing env var
        if "REVERIE_PROFILES" in os.environ:
            del os.environ["REVERIE_PROFILES"]

    def teardown_method(self):
        """Clean up after each test."""
        clear_profile_cache()
        if "REVERIE_PROFILES" in os.environ:
            del os.environ["REVERIE_PROFILES"]

    def test_env_profiles_loaded(self):
        """Test that REVERIE_PROFILES env is loaded correctly."""
        os.environ["REVERIE_PROFILES"] = json.dumps(TEST_PROFILES)
        clear_profile_cache()

        profiles = get_profiles()
        assert "main" in profiles
        assert "secondary" in profiles
        assert profiles["main"].name == "Main Agent"
        assert profiles["secondary"].name == "Secondary Agent"

    def test_missing_env_returns_empty_dict(self):
        """Test that missing REVERIE_PROFILES returns empty dict (setup mode)."""
        # Ensure env is not set
        if "REVERIE_PROFILES" in os.environ:
            del os.environ["REVERIE_PROFILES"]
        clear_profile_cache()

        # In setup mode, returns empty dict instead of raising error
        profiles = get_profiles()
        assert profiles == {}

    def test_empty_profiles_returns_empty_dict(self):
        """Test that empty profiles returns empty dict."""
        os.environ["REVERIE_PROFILES"] = json.dumps({})
        clear_profile_cache()

        profiles = get_profiles()
        assert profiles == {}

    def test_invalid_json_returns_empty_dict(self):
        """Test that invalid JSON returns empty dict (graceful handling)."""
        os.environ["REVERIE_PROFILES"] = "not valid json"
        clear_profile_cache()

        # Gracefully returns empty dict and logs error
        profiles = get_profiles()
        assert profiles == {}

    def test_missing_required_field_returns_empty_dict(self):
        """Test that missing required fields returns empty dict."""
        os.environ["REVERIE_PROFILES"] = json.dumps(
            {"bad": {"id": "bad"}}  # Missing name and sessions_dir
        )
        clear_profile_cache()

        # Gracefully returns empty dict and logs error
        profiles = get_profiles()
        assert profiles == {}


class TestGetProfile:
    """Tests for get_profile function."""

    def setup_method(self):
        """Clear cache and set env before each test."""
        clear_profile_cache()
        os.environ["REVERIE_PROFILES"] = json.dumps(TEST_PROFILES)

    def teardown_method(self):
        """Clean up after each test."""
        clear_profile_cache()
        if "REVERIE_PROFILES" in os.environ:
            del os.environ["REVERIE_PROFILES"]

    def test_get_existing_profile(self):
        """Test getting an existing profile."""
        profile = get_profile("main")
        assert profile is not None
        assert profile.id == "main"
        assert profile.name == "Main Agent"

    def test_get_nonexistent_profile(self):
        """Test getting a non-existent profile returns None."""
        profile = get_profile("nonexistent")
        assert profile is None


class TestMultiFolderSupport:
    """Tests for multi-folder indexing support."""

    def test_create_profile_with_multiple_dirs(self):
        """Test creating ProfileConfig with multiple sessions directories."""
        profile = ProfileConfig(
            id="test",
            name="Test Profile",
            sessions_dirs=["~/.claude/project1/", "~/.claude/project2/"],
        )
        assert profile.id == "test"
        assert profile.sessions_dirs == ["~/.claude/project1/", "~/.claude/project2/"]

    def test_sessions_paths_returns_list(self):
        """Test that sessions_paths returns a list of Path objects."""
        profile = ProfileConfig(
            id="test",
            name="Test",
            sessions_dirs=["~/path1/", "/absolute/path2/"],
        )
        paths = profile.sessions_paths
        assert isinstance(paths, list)
        assert len(paths) == 2
        assert paths[0] == Path.home() / "path1"
        assert paths[1] == Path("/absolute/path2/")

    def test_empty_sessions_dirs_raises_error(self):
        """Test that empty sessions_dirs raises ValueError."""
        with pytest.raises(ValueError, match="sessions_dirs cannot be empty"):
            ProfileConfig(id="test", name="Test", sessions_dirs=[])

    def test_backward_compat_sessions_dir_converted_to_list(self):
        """Test backward compatibility: sessions_dir string converted to sessions_dirs list."""
        profile = ProfileConfig(
            id="test",
            name="Test",
            sessions_dir="~/.claude/legacy/",
        )
        # Should have sessions_dirs as a list with single element
        assert profile.sessions_dirs == ["~/.claude/legacy/"]
        assert len(profile.sessions_paths) == 1

    def test_backward_compat_sessions_path_still_works(self):
        """Test backward compatibility: sessions_path property still works for single dir."""
        profile = ProfileConfig(
            id="test",
            name="Test",
            sessions_dir="~/.claude/legacy/",
        )
        # Legacy property should return the first path
        assert profile.sessions_path == Path.home() / ".claude" / "legacy"

    def test_duplicate_paths_raises_error(self):
        """Test that duplicate paths in sessions_dirs raises ValueError."""
        with pytest.raises(ValueError, match="sessions_dirs contains duplicates"):
            ProfileConfig(
                id="test",
                name="Test",
                sessions_dirs=["/same/path/", "/same/path/"],
            )

    def test_both_sessions_dir_and_sessions_dirs_normalizes(self):
        """Test behavior when both old and new format are provided.

        sessions_dirs takes precedence and sessions_dir is normalized
        to match the first element of sessions_dirs.
        """
        profile = ProfileConfig(
            id="test",
            name="Test",
            sessions_dir="~/.claude/old/",  # Should be overwritten
            sessions_dirs=["~/.claude/new1/", "~/.claude/new2/"],
        )
        assert profile.sessions_dirs == ["~/.claude/new1/", "~/.claude/new2/"]
        # sessions_dir should be normalized to first element
        assert profile.sessions_dir == "~/.claude/new1/"


class TestMultiFolderEnvParsing:
    """Tests for multi-folder parsing from environment variable."""

    def setup_method(self):
        """Clear cache and set env before each test."""
        clear_profile_cache()
        if "REVERIE_PROFILES" in os.environ:
            del os.environ["REVERIE_PROFILES"]

    def teardown_method(self):
        """Clean up after each test."""
        clear_profile_cache()
        if "REVERIE_PROFILES" in os.environ:
            del os.environ["REVERIE_PROFILES"]

    def test_env_with_sessions_dirs_array(self):
        """Test loading profiles with sessions_dirs array from env."""
        profiles_data = {
            "multi": {
                "id": "multi",
                "name": "Multi Folder Agent",
                "sessions_dirs": [
                    "~/.claude/project1/",
                    "~/.claude/project2/",
                ],
            }
        }
        os.environ["REVERIE_PROFILES"] = json.dumps(profiles_data)
        clear_profile_cache()

        profiles = get_profiles()
        assert "multi" in profiles
        assert profiles["multi"].sessions_dirs == [
            "~/.claude/project1/",
            "~/.claude/project2/",
        ]

    def test_env_backward_compat_sessions_dir_string(self):
        """Test backward compatibility: sessions_dir string in env is converted."""
        profiles_data = {
            "legacy": {
                "id": "legacy",
                "name": "Legacy Agent",
                "sessions_dir": "~/.claude/legacy/",  # Old format
            }
        }
        os.environ["REVERIE_PROFILES"] = json.dumps(profiles_data)
        clear_profile_cache()

        profiles = get_profiles()
        assert "legacy" in profiles
        # Should be converted to list
        assert profiles["legacy"].sessions_dirs == ["~/.claude/legacy/"]


class TestGetDefaultProfile:
    """Tests for get_default_profile function."""

    def setup_method(self):
        """Clear cache and set env before each test."""
        clear_profile_cache()
        if "REVERIE_PROFILES" in os.environ:
            del os.environ["REVERIE_PROFILES"]

    def teardown_method(self):
        """Clean up after each test."""
        clear_profile_cache()
        if "REVERIE_PROFILES" in os.environ:
            del os.environ["REVERIE_PROFILES"]

    def test_default_returns_first_profile(self):
        """Test that default profile is the first configured profile."""
        os.environ["REVERIE_PROFILES"] = json.dumps(TEST_PROFILES)
        clear_profile_cache()

        profile = get_default_profile()
        # First profile in dict
        assert profile.id in ("main", "secondary")

    def test_single_profile_is_default(self):
        """Test that single profile becomes default."""
        os.environ["REVERIE_PROFILES"] = json.dumps(
            {
                "only": {
                    "id": "only",
                    "name": "Only Profile",
                    "sessions_dir": "/path/",
                }
            }
        )
        clear_profile_cache()

        profile = get_default_profile()
        assert profile.id == "only"

    def test_raises_when_no_profiles_configured(self):
        """Test that ValueError is raised when no profiles are configured."""
        # Ensure env is not set
        if "REVERIE_PROFILES" in os.environ:
            del os.environ["REVERIE_PROFILES"]
        clear_profile_cache()

        # In setup mode, get_profiles returns empty dict, so get_default_profile raises
        with pytest.raises(ValueError, match="No profiles configured"):
            get_default_profile()
