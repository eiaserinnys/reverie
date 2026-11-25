"""Configuration module for reverie.

Contains settings and profile management.
"""

# Settings
from reverie.config.settings import Settings, get_settings

# Profile management
from reverie.config.profiles import (
    ProfileConfig,
    get_profile,
    get_profiles,
    get_default_profile,
)

__all__ = [
    "Settings",
    "get_settings",
    "ProfileConfig",
    "get_profile",
    "get_profiles",
    "get_default_profile",
]
