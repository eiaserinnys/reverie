"""Profile configuration for incremental indexing.

Manages multiple agent profiles with their session directories.
Profiles MUST be configured via REVERIE_PROFILES environment variable.
"""

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class ProfileConfig:
    """Configuration for an agent profile.

    Attributes:
        id: Unique profile identifier.
        name: Display name for the profile.
        sessions_dirs: List of paths to Claude Code session JSONL directories.
        sessions_dir: (Deprecated) Single path for backward compatibility.
    """

    id: str
    name: str
    sessions_dirs: list[str] | None = None
    sessions_dir: str | None = None  # Deprecated, for backward compatibility

    @property
    def sessions_paths(self) -> list[Path]:
        """Get resolved sessions directory paths."""
        if self.sessions_dirs is None:
            return []
        return [Path(os.path.expanduser(d)) for d in self.sessions_dirs]

    @property
    def sessions_path(self) -> Path:
        """Get resolved sessions directory path (first one for backward compatibility)."""
        paths = self.sessions_paths
        if not paths:
            raise ValueError("No sessions directories configured")
        return paths[0]

    def __post_init__(self):
        """Validate and normalize profile configuration."""
        if not self.id:
            raise ValueError("Profile id cannot be empty")
        if not self.name:
            raise ValueError("Profile name cannot be empty")

        # Handle backward compatibility: convert sessions_dir to sessions_dirs
        if self.sessions_dirs is None and self.sessions_dir is not None:
            # Legacy format: single sessions_dir string
            if not self.sessions_dir:
                raise ValueError("Profile sessions_dir cannot be empty")
            self.sessions_dirs = [self.sessions_dir]
        elif self.sessions_dirs is not None:
            # New format: sessions_dirs list
            if not self.sessions_dirs:
                raise ValueError("Profile sessions_dirs cannot be empty")
            # Check for duplicates
            if len(set(self.sessions_dirs)) != len(self.sessions_dirs):
                raise ValueError("Profile sessions_dirs contains duplicates")
            # Normalize: sessions_dir should match first element of sessions_dirs
            self.sessions_dir = self.sessions_dirs[0]
        else:
            # Neither provided
            raise ValueError("Profile sessions_dirs cannot be empty")


def _load_profiles_from_env() -> dict[str, ProfileConfig]:
    """Load profiles from REVERIE_PROFILES environment variable.

    Expected format (new - multi-folder):
    ```json
    {
        "default": {
            "id": "default",
            "name": "Default Agent",
            "sessions_dirs": ["~/.claude/project1/", "~/.claude/project2/"]
        }
    }
    ```

    Legacy format (still supported):
    ```json
    {
        "default": {
            "id": "default",
            "name": "Default Agent",
            "sessions_dir": "~/.claude/projects/"
        }
    }
    ```

    Returns:
        Dict of ProfileConfig. Empty dict if not configured (setup mode).

    Note:
        Returns empty dict instead of raising error when REVERIE_PROFILES
        is not set, allowing the server to bootstrap for Setup Wizard.
    """
    profiles_json = os.environ.get("REVERIE_PROFILES")
    if not profiles_json:
        logger.info("REVERIE_PROFILES not set - running in setup mode (no indexing)")
        return {}

    try:
        profiles_data = json.loads(profiles_json)
        if not profiles_data:
            logger.warning("REVERIE_PROFILES is empty - no profiles configured")
            return {}

        profiles = {}
        for profile_id, config in profiles_data.items():
            # Support both new (sessions_dirs) and legacy (sessions_dir) formats
            sessions_dirs = config.get("sessions_dirs")
            sessions_dir = config.get("sessions_dir")

            if sessions_dirs is None and sessions_dir is None:
                raise KeyError("sessions_dirs or sessions_dir")

            profiles[profile_id] = ProfileConfig(
                id=config.get("id", profile_id),
                name=config["name"],
                sessions_dirs=sessions_dirs,
                sessions_dir=sessions_dir,
            )
        logger.info("Loaded %d profiles from REVERIE_PROFILES env", len(profiles))
        return profiles
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in REVERIE_PROFILES: {e}")
        return {}
    except KeyError as e:
        logger.error(f"Missing required field in REVERIE_PROFILES: {e}")
        return {}


@lru_cache(maxsize=1)
def get_profiles() -> dict[str, ProfileConfig]:
    """Get all available profiles.

    Loads from REVERIE_PROFILES environment variable.

    Returns:
        Dict mapping profile_id to ProfileConfig.

    Raises:
        ValueError: If REVERIE_PROFILES is not configured.
    """
    return _load_profiles_from_env()


def get_profile(profile_id: str) -> ProfileConfig | None:
    """Get a specific profile by ID.

    Args:
        profile_id: Profile identifier.

    Returns:
        ProfileConfig if found, None otherwise.
    """
    return get_profiles().get(profile_id)


def get_default_profile() -> ProfileConfig:
    """Get the default profile (first configured profile).

    Returns:
        Default ProfileConfig.

    Raises:
        ValueError: If no profiles are configured.
    """
    profiles = get_profiles()
    if not profiles:
        raise ValueError("No profiles configured in REVERIE_PROFILES")
    # Return first profile as default
    return next(iter(profiles.values()))


def clear_profile_cache() -> None:
    """Clear the profile cache (for testing)."""
    get_profiles.cache_clear()
