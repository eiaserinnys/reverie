"""Database backend factory.

Creates the appropriate ReverieDB implementation based on configuration.
Supports PostgreSQL and SQLite backends.
"""

import logging
from pathlib import Path

from reverie.db.reverie_db import ReverieDB

logger = logging.getLogger(__name__)


# Default SQLite database path
DEFAULT_SQLITE_PATH = Path.home() / ".reverie" / "reverie.db"


def create_db(backend: str = "postgres", **kwargs) -> ReverieDB:
    """Create a ReverieDB instance for the specified backend.

    Args:
        backend: Database backend type ("postgres" or "sqlite").
        **kwargs: Backend-specific connection parameters.
            For postgres: conn, host, port, user, password, dbname
            For sqlite: db_path (path to .db file, default ~/.reverie/reverie.db)

    Returns:
        ReverieDB implementation instance.

    Raises:
        ValueError: If backend type is not supported.
    """
    if backend == "postgres":
        return _create_postgres_db(**kwargs)
    elif backend == "sqlite":
        return _create_sqlite_db(**kwargs)
    else:
        raise ValueError(f"Unsupported database backend: {backend}")


def _create_postgres_db(**kwargs) -> ReverieDB:
    """Create PostgreSQL database instance.

    Args:
        **kwargs: Connection parameters. Can include:
            - conn: Existing psycopg2 connection (use this if provided)
            - host, port, user, password, dbname: Connection parameters
            - If none provided, uses settings from config

    Returns:
        PostgresDB instance.
    """
    from reverie.db.postgres_db import PostgresDB

    # Use provided connection if available
    if "conn" in kwargs:
        return PostgresDB(kwargs["conn"])

    # Create new connection
    import psycopg2

    # If specific params provided, use them
    if any(k in kwargs for k in ("host", "port", "user", "password", "dbname")):
        conn = psycopg2.connect(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 5432),
            user=kwargs.get("user", "postgres"),
            password=kwargs.get("password", ""),
            dbname=kwargs.get("dbname", "reverie"),
        )
    else:
        # Use settings from config
        from reverie.config import get_settings

        settings = get_settings()
        conn = psycopg2.connect(
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            dbname=settings.db_name,
        )

    return PostgresDB(conn)


def _create_sqlite_db(**kwargs) -> ReverieDB:
    """Create SQLite database instance.

    Args:
        **kwargs: Connection parameters. Can include:
            - db_path: Path to SQLite database file (default: ~/.reverie/reverie.db)
            - run_migrations: Whether to run migrations (default: True)
            - If none provided, uses settings from config

    Returns:
        SqliteDB instance.
    """
    from reverie.db.sqlite_db import SqliteDB

    # Get database path
    if "db_path" in kwargs:
        db_path = Path(kwargs["db_path"])
    else:
        try:
            from reverie.config import get_settings
            settings = get_settings()
            db_path = Path(getattr(settings, "sqlite_path", str(DEFAULT_SQLITE_PATH)))
        except Exception:
            db_path = DEFAULT_SQLITE_PATH

    # Expand user path and ensure directory exists
    db_path = db_path.expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create database connection
    db = SqliteDB(db_path)

    # Run migrations if requested (default: True)
    if kwargs.get("run_migrations", True):
        _run_sqlite_migrations(db)

    return db


def _run_sqlite_migrations(db: "SqliteDB") -> None:
    """Run SQLite migrations on the database.

    Reads migration files from migrations/sqlite/ and applies them in order.
    """
    from pathlib import Path

    # Find migrations directory
    # Navigate from src/reverie/db/ up to project root
    project_root = Path(__file__).parent.parent.parent.parent
    migrations_dir = project_root / "migrations" / "sqlite"

    if not migrations_dir.exists():
        logger.warning("SQLite migrations directory not found: %s", migrations_dir)
        return

    # Get applied migrations
    try:
        cursor = db.raw_connection.execute(
            "SELECT filename FROM schema_migrations ORDER BY filename"
        )
        applied = {row[0] for row in cursor.fetchall()}
    except Exception:
        # Table doesn't exist yet, will be created by first migration
        applied = set()

    # Get pending migrations
    all_migrations = sorted(migrations_dir.glob("*.sql"))
    pending = [m for m in all_migrations if m.name not in applied]

    if not pending:
        logger.debug("No pending SQLite migrations")
        return

    # Apply migrations
    for migration_path in pending:
        logger.info("Applying SQLite migration: %s", migration_path.name)
        try:
            sql = migration_path.read_text(encoding="utf-8")
            # SQLite can't execute multiple statements with executescript if
            # we want to track individual migrations, so we execute the whole
            # file as a script and track it
            db.raw_connection.executescript(sql)
            db.raw_connection.execute(
                "INSERT INTO schema_migrations (filename) VALUES (?)",
                (migration_path.name,),
            )
            db.commit()
            logger.info("  Applied: %s", migration_path.name)
        except Exception as e:
            db.rollback()
            logger.error("  Failed: %s - %s", migration_path.name, e)
            raise

    logger.info("Applied %d SQLite migration(s)", len(pending))


def get_backend_from_settings() -> str:
    """Get database backend type from settings.

    Returns:
        Backend type string ("postgres" or "sqlite").
    """
    try:
        from reverie.config import get_settings
        settings = get_settings()
        # Default to postgres if db_backend not set
        return getattr(settings, "db_backend", "postgres")
    except Exception:
        return "postgres"


def create_db_from_settings() -> ReverieDB:
    """Create ReverieDB instance using settings from config.

    Convenience function that reads backend type from settings
    and creates the appropriate database instance.

    Returns:
        ReverieDB implementation instance.
    """
    backend = get_backend_from_settings()
    return create_db(backend=backend)
