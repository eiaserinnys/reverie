"""Configuration for Reverie.

Loads settings from environment variables or .env file.
Supports "setup mode" where required fields have defaults for initial setup wizard.
"""

import json
import logging
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def _is_setup_complete() -> bool:
    """Check if setup is complete by reading .reverie/setup.json.

    Returns:
        True if setup_complete is True in setup.json, False otherwise.
    """
    setup_json_path = Path(__file__).parent.parent.parent.parent / ".reverie" / "setup.json"
    if not setup_json_path.exists():
        return False

    try:
        data = json.loads(setup_json_path.read_text())
        return data.get("setup_complete", False)
    except (json.JSONDecodeError, OSError):
        return False


# Determine setup mode at module load time
# In setup mode, required fields get defaults to allow server bootstrap
_SETUP_COMPLETE = _is_setup_complete()


class Settings(BaseSettings):
    """Reverie configuration.

    In setup mode (before setup completion), required fields have defaults
    to allow the server to bootstrap and reach the Setup Wizard.
    After setup is complete, these fields are validated as required.
    """

    model_config = SettingsConfigDict(
        env_prefix="REVERIE_",
        # .env file is at project root (three levels up from config/settings.py)
        env_file=str(Path(__file__).parent.parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database - defaults for setup mode, required after setup
    db_host: str = Field(
        default="" if not _SETUP_COMPLETE else ...,
        description="DB host (required after setup, set REVERIE_DB_HOST)"
    )
    db_port: int = Field(
        default=5432 if not _SETUP_COMPLETE else ...,
        description="DB port (required after setup, set REVERIE_DB_PORT)"
    )
    db_user: str = Field(
        default="" if not _SETUP_COMPLETE else ...,
        description="DB username (required after setup, set REVERIE_DB_USER)"
    )
    db_password: str = Field(
        default="" if not _SETUP_COMPLETE else ...,
        description="DB password (required after setup, set REVERIE_DB_PASSWORD)"
    )
    db_name: str = Field(
        default="reverie" if not _SETUP_COMPLETE else ...,
        description="DB name (required after setup, set REVERIE_DB_NAME)"
    )

    # CORS - allow all in setup mode
    cors_origin: str = Field(
        default="*" if not _SETUP_COMPLETE else ...,
        description="CORS origin (required after setup, set REVERIE_CORS_ORIGIN)"
    )

    # OpenAI
    openai_api_key: str = ""

    # Database backend selection ("postgres" or "sqlite")
    db_backend: str = "postgres"

    # SQLite settings (used when db_backend = "sqlite")
    sqlite_path: str = "~/.reverie/reverie.db"

    # Session JSONL (set via REVERIE_SESSION_DIR)
    session_dir: str = ""

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 16

    # OpenIE (NER + triple extraction) - defaults for setup mode
    openie_model: str = Field(
        default="gpt-5-mini" if not _SETUP_COMPLETE else ...,
        description="OpenIE model (required after setup, set REVERIE_OPENIE_MODEL)"
    )
    openie_temperature: float = 0.0
    openie_max_tokens: int = 4096

    # Query NER (lightweight extraction for retrieval)
    query_ner_model: str = Field(
        default="gpt-5-mini",
        description="Query NER model (set REVERIE_QUERY_NER_MODEL)"
    )

    # Enrich models (for retrieval enrichment)
    enrich_stage1_model: str = Field(
        default="gpt-5-mini",
        description="Enrich stage 1 model - parallel extraction (set REVERIE_ENRICH_STAGE1_MODEL)"
    )
    enrich_stage2_model: str = Field(
        default="gpt-5.2",
        description="Enrich stage 2 model - re-ranking (set REVERIE_ENRICH_STAGE2_MODEL)"
    )

    # API Cache
    cache_dir: str = str(Path(__file__).parent.parent.parent / ".cache")
    cache_enabled: bool = True

    # Chunking
    chunk_token_limit: int = 2000
    tool_result_max_lines: int = 20

    # Entity filtering
    entity_min_length: int = 3
    entity_max_length: int = 80

    # Entity similarity matching (embedding-based dedup in upsert)
    entity_similarity_threshold: float = 0.92

    # Graph
    synonym_threshold: float = 0.8
    ppr_damping: float = 0.85
    ppr_top_k: int = 10
    graph_ttl_seconds: int = 3600  # Graph cache TTL (1 hour)

    # Retrieval settings
    v2_tau: float = 0.5  # Entity activation threshold
    v2_delta: float = 0.1  # Pruning threshold
    v2_damping: float = 0.85  # PPR damping factor
    v2_max_iter: int = 2  # Semantic propagation iterations
    v2_adjacency_window: int = 1  # Passage adjacency window (turns)
    v2_beta: float = 1.0  # PPR/KNN hybrid weight (1.0 = PPR only, 0.0 = KNN only)
    v2_score_threshold: float = 0.001  # Minimum PPR score to include node in visualization
    v2_use_server_side_ppr: bool = False  # Use PostgreSQL server-side PPR function

    # Energy BFS settings (unified graph traversal)
    energy_bfs_alpha: float = 0.5  # Structural cost weight (0-1)
    energy_bfs_beta: float = 0.5  # Semantic cost weight (0-1)
    energy_bfs_initial: float = 10.0  # Initial energy budget
    energy_bfs_max_energy: float = 50.0  # Max energy for adaptive search
    energy_bfs_multi_seed_bonus: float = 1.5  # Bonus for multi-seed convergence

    # Fact novelty gate
    fact_novelty_threshold: float = 0.85
    fact_novelty_top_k: int = 10

    # Session preprocessor (turn-level filtering before OpenIE) - defaults for setup mode
    preprocessor_enabled: bool = False
    preprocessor_model: str = Field(
        default="gpt-5-mini" if not _SETUP_COMPLETE else ...,
        description="Preprocessor model (required after setup, set REVERIE_PREPROCESSOR_MODEL)"
    )
    preprocessor_temperature: float = 0.0
    preprocessor_max_tokens: int = 2048

    # Incremental indexing
    index_workers: int = 10  # Number of parallel indexing workers
    index_debounce_seconds: float = 2.0  # File watcher debounce delay

    # Retrieval
    fact_knn_top_k: int = 50
    recall_max_passages: int = 5
    recall_max_tokens: int = 4000

    # Logging
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_required_fields_after_setup(self) -> "Settings":
        """Validate required fields after setup is complete."""
        if _SETUP_COMPLETE:
            missing = []
            if not self.db_host:
                missing.append("REVERIE_DB_HOST")
            if not self.db_user:
                missing.append("REVERIE_DB_USER")
            if not self.db_password:
                missing.append("REVERIE_DB_PASSWORD")
            if not self.cors_origin or self.cors_origin == "*":
                # Allow * in production if explicitly set
                pass

            if missing:
                raise ValueError(
                    f"Setup is complete but missing required settings: {', '.join(missing)}. "
                    "Please configure these in your .env file."
                )
        return self

    @classmethod
    def is_setup_mode(cls) -> bool:
        """Check if currently in setup mode (setup not complete)."""
        return not _SETUP_COMPLETE

    @property
    def database_url(self) -> str:
        """PostgreSQL connection URL."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def async_database_url(self) -> str:
        """Async PostgreSQL connection URL (for asyncpg)."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


def get_settings() -> Settings:
    """Get settings singleton."""
    return Settings()
