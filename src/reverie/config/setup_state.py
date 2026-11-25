"""Setup state management for Reverie.

Handles:
- Reading/writing .reverie/setup.json (setup progress)
- Managing .env file generation (configuration)
- Tracking setup phases (database, apikeys, profile, migration, verification)
"""

import json
import logging
import os
import stat
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PhaseStatus(str, Enum):
    """Status of a setup phase."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


class SetupPhase(BaseModel):
    """A single setup phase."""

    status: PhaseStatus = PhaseStatus.PENDING
    checks: dict[str, dict[str, Any]] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class SetupConfig(BaseModel):
    """Complete setup configuration."""

    setup_complete: bool = False
    phases: dict[str, SetupPhase] = Field(default_factory=dict)
    last_updated: str = ""

    @classmethod
    def default(cls) -> "SetupConfig":
        """Create default setup config with all phases."""
        return cls(
            setup_complete=False,
            phases={
                "greeting": SetupPhase(),
                "database": SetupPhase(),
                "apikeys": SetupPhase(),
                "profile": SetupPhase(),
                "migration": SetupPhase(),
                "verification": SetupPhase(),
            },
            last_updated=datetime.utcnow().isoformat(),
        )


class SetupState:
    """Manages setup state and .env file."""

    def __init__(self, project_dir: Path | str | None = None):
        """Initialize setup state.

        Args:
            project_dir: Path to the project directory. If None, uses the
                         reverie source package directory.
        """
        if project_dir is None:
            # Default to package root (src/reverie/../../..)
            project_dir = Path(__file__).parent.parent.parent.parent

        self.project_dir = Path(project_dir)
        self.reverie_dir = self.project_dir / ".reverie"
        self.setup_json_path = self.reverie_dir / "setup.json"
        self.env_path = self.project_dir / ".env"

        # Ensure .reverie directory exists
        self.reverie_dir.mkdir(parents=True, exist_ok=True)

        # Load or create config
        self.config = self._load_config()

    def _load_config(self) -> SetupConfig:
        """Load config from setup.json or create default."""
        if self.setup_json_path.exists():
            try:
                data = json.loads(self.setup_json_path.read_text())
                # Parse phases with proper type handling
                phases = {}
                for name, phase_data in data.get("phases", {}).items():
                    if isinstance(phase_data, dict):
                        status_str = phase_data.get("status", "pending")
                        phases[name] = SetupPhase(
                            status=PhaseStatus(status_str),
                            checks=phase_data.get("checks", {}),
                            config=phase_data.get("config", {}),
                        )
                    else:
                        phases[name] = SetupPhase()

                return SetupConfig(
                    setup_complete=data.get("setup_complete", False),
                    phases=phases,
                    last_updated=data.get("last_updated", ""),
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse setup.json, using defaults: {e}")

        return SetupConfig.default()

    def _save_config(self) -> None:
        """Save config to setup.json."""
        self.config.last_updated = datetime.utcnow().isoformat()

        # Convert to dict for JSON serialization
        data = {
            "setup_complete": self.config.setup_complete,
            "phases": {},
            "last_updated": self.config.last_updated,
        }

        for name, phase in self.config.phases.items():
            data["phases"][name] = {
                "status": phase.status.value,
                "checks": phase.checks,
                "config": phase.config,
            }

        self.setup_json_path.write_text(json.dumps(data, indent=2))

    def update_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Update a phase's status.

        Args:
            phase_name: Name of the phase.
            status: New status.
        """
        if phase_name not in self.config.phases:
            self.config.phases[phase_name] = SetupPhase()

        self.config.phases[phase_name].status = status
        self._save_config()

    def update_phase_check(
        self,
        phase_name: str,
        check_name: str,
        success: bool,
        message: str = "",
    ) -> None:
        """Update a check result within a phase.

        Args:
            phase_name: Name of the phase.
            check_name: Name of the check.
            success: Whether the check passed.
            message: Optional message.
        """
        if phase_name not in self.config.phases:
            self.config.phases[phase_name] = SetupPhase()

        self.config.phases[phase_name].checks[check_name] = {
            "success": success,
            "message": message,
            "checked_at": datetime.utcnow().isoformat(),
        }
        self._save_config()

    def get_phase_status(self, phase_name: str) -> PhaseStatus:
        """Get a phase's status.

        Args:
            phase_name: Name of the phase.

        Returns:
            The phase status, or PENDING if unknown.
        """
        if phase_name not in self.config.phases:
            return PhaseStatus.PENDING
        return self.config.phases[phase_name].status

    def mark_complete(self) -> None:
        """Mark setup as complete.

        Also ensures CORS_ORIGIN has a value in .env file.
        All model settings should be configured via Setup Wizard's API Keys phase.
        """
        env_vars = self._read_env_file()

        # CORS origin - default to allow all
        if "REVERIE_CORS_ORIGIN" not in env_vars:
            env_vars["REVERIE_CORS_ORIGIN"] = "*"

        self._write_env_file(env_vars)

        self.config.setup_complete = True
        self._save_config()

    def is_complete(self) -> bool:
        """Check if setup is complete.

        Returns:
            True if setup is complete.
        """
        return self.config.setup_complete

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation of the setup state.
        """
        return {
            "setup_complete": self.config.setup_complete,
            "phases": {
                name: {
                    "status": phase.status.value,
                    "checks": phase.checks,
                    "config": phase.config,
                }
                for name, phase in self.config.phases.items()
            },
            "last_updated": self.config.last_updated,
        }

    def _read_env_file(self) -> dict[str, str]:
        """Read existing .env file into a dictionary.

        Returns:
            Dictionary of environment variables.
        """
        env_vars = {}
        if self.env_path.exists():
            content = self.env_path.read_text()
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env_vars[key.strip()] = value.strip()
        return env_vars

    def _write_env_file(self, env_vars: dict[str, str]) -> None:
        """Write environment variables to .env file.

        Security: Sets file permissions to 600 (owner read/write only)
        to protect sensitive credentials.

        Args:
            env_vars: Dictionary of environment variables.
        """
        lines = []
        for key, value in env_vars.items():
            lines.append(f"{key}={value}")
        self.env_path.write_text("\n".join(lines) + "\n")
        # Set permissions to 600 (owner read/write only) for security
        self.env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def save_database_config(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        name: str,
    ) -> None:
        """Save database configuration to .env file.

        Args:
            host: Database host.
            port: Database port.
            user: Database user.
            password: Database password.
            name: Database name.
        """
        env_vars = self._read_env_file()

        env_vars["REVERIE_DB_HOST"] = host
        env_vars["REVERIE_DB_PORT"] = str(port)
        env_vars["REVERIE_DB_USER"] = user
        env_vars["REVERIE_DB_PASSWORD"] = password
        env_vars["REVERIE_DB_NAME"] = name

        self._write_env_file(env_vars)

        # Also update phase config
        if "database" not in self.config.phases:
            self.config.phases["database"] = SetupPhase()

        self.config.phases["database"].config = {
            "host": host,
            "port": port,
            "user": user,
            "name": name,
            # Don't store password in setup.json
        }
        self._save_config()

    def save_api_key(
        self,
        api_key: str,
        embedding_model: str | None = None,
        indexing_ner_model: str | None = None,
        query_ner_model: str | None = None,
        preprocessor_model: str | None = None,
        enrich_stage1_model: str | None = None,
        enrich_stage2_model: str | None = None,
    ) -> None:
        """Save OpenAI API key and model settings to .env file.

        Args:
            api_key: OpenAI API key.
            embedding_model: Embedding model name.
            indexing_ner_model: Indexing NER model name.
            query_ner_model: Query NER model name.
            preprocessor_model: Preprocessor model name.
            enrich_stage1_model: Enrich stage 1 model name.
            enrich_stage2_model: Enrich stage 2 model name.
        """
        env_vars = self._read_env_file()
        env_vars["REVERIE_OPENAI_API_KEY"] = api_key

        # Save model settings if provided
        if embedding_model:
            env_vars["REVERIE_EMBEDDING_MODEL"] = embedding_model
        if indexing_ner_model:
            env_vars["REVERIE_OPENIE_MODEL"] = indexing_ner_model
        if query_ner_model:
            env_vars["REVERIE_QUERY_NER_MODEL"] = query_ner_model
        if preprocessor_model:
            env_vars["REVERIE_PREPROCESSOR_MODEL"] = preprocessor_model
        if enrich_stage1_model:
            env_vars["REVERIE_ENRICH_STAGE1_MODEL"] = enrich_stage1_model
        if enrich_stage2_model:
            env_vars["REVERIE_ENRICH_STAGE2_MODEL"] = enrich_stage2_model

        self._write_env_file(env_vars)

        # Update phase status
        if "apikeys" not in self.config.phases:
            self.config.phases["apikeys"] = SetupPhase()

        self.config.phases["apikeys"].config["validated"] = True
        self._save_config()

    def save_profile_config(
        self,
        profile_name: str,
        sessions_paths: list[str] | None = None,
        sessions_path: str | None = None,  # Deprecated, for backwards compatibility
    ) -> None:
        """Save profile configuration.

        Args:
            profile_name: Name of the profile.
            sessions_paths: List of paths to session files (multi-folder indexing).
            sessions_path: Single path (deprecated, use sessions_paths instead).
        """
        # Handle backwards compatibility
        paths: list[str] = []
        if sessions_paths is not None:
            paths = sessions_paths
        elif sessions_path is not None and sessions_path:
            paths = [sessions_path]

        env_vars = self._read_env_file()

        # Store as comma-separated list for multi-folder support
        # Empty list results in empty string
        env_vars["REVERIE_SESSION_DIRS"] = ",".join(paths)

        # Keep legacy REVERIE_SESSION_DIR for backwards compatibility
        # Use first path if available
        env_vars["REVERIE_SESSION_DIR"] = paths[0] if paths else ""

        # Build REVERIE_PROFILES JSON for IndexScheduler
        # profiles.py reads this to determine which directories to watch
        profile_id = profile_name.lower().replace(" ", "_")
        profiles = {
            profile_id: {
                "id": profile_id,
                "name": profile_name,
                "sessions_dirs": paths,
            }
        }
        env_vars["REVERIE_PROFILES"] = json.dumps(profiles)

        self._write_env_file(env_vars)

        # Update phase config
        if "profile" not in self.config.phases:
            self.config.phases["profile"] = SetupPhase()

        self.config.phases["profile"].config = {
            "profile_name": profile_name,
            "sessions_paths": paths,
            # Keep sessions_path for backwards compatibility
            "sessions_path": paths[0] if paths else "",
        }
        self._save_config()

    def get_database_config(self) -> dict:
        """Get database configuration from .env file.

        Returns:
            Dictionary with database configuration.
        """
        env_vars = self._read_env_file()

        port_str = env_vars.get("REVERIE_DB_PORT", "5432")
        try:
            port = int(port_str)
        except ValueError:
            port = 5432

        return {
            "host": env_vars.get("REVERIE_DB_HOST", "localhost"),
            "port": port,
            "user": env_vars.get("REVERIE_DB_USER", ""),
            "password": env_vars.get("REVERIE_DB_PASSWORD", ""),
            "name": env_vars.get("REVERIE_DB_NAME", "reverie"),
        }

    def get_api_key(self) -> str:
        """Get OpenAI API key from .env file.

        Returns:
            API key or empty string if not set.
        """
        env_vars = self._read_env_file()
        return env_vars.get("REVERIE_OPENAI_API_KEY", "")

    def save_backend_config(
        self,
        backend: str,
        sqlite_path: str | None = None,
    ) -> None:
        """Save database backend configuration to .env file.

        Args:
            backend: Database backend type ("postgres" or "sqlite").
            sqlite_path: Path to SQLite database file (for sqlite backend).
        """
        env_vars = self._read_env_file()

        env_vars["REVERIE_DB_BACKEND"] = backend

        if backend == "sqlite":
            # Use provided path or default
            path = sqlite_path if sqlite_path else "~/.reverie/reverie.db"
            env_vars["REVERIE_SQLITE_PATH"] = path

        self._write_env_file(env_vars)

        # Update phase config
        if "database" not in self.config.phases:
            self.config.phases["database"] = SetupPhase()

        self.config.phases["database"].config["backend"] = backend
        if backend == "sqlite":
            self.config.phases["database"].config["sqlite_path"] = (
                sqlite_path if sqlite_path else "~/.reverie/reverie.db"
            )
        self._save_config()

    def get_backend_config(self) -> dict:
        """Get database backend configuration from .env file.

        Returns:
            Dictionary with backend configuration.
        """
        env_vars = self._read_env_file()

        return {
            "backend": env_vars.get("REVERIE_DB_BACKEND"),  # None if not configured
            "sqlite_path": env_vars.get("REVERIE_SQLITE_PATH", "~/.reverie/reverie.db"),
        }
