"""MCP server for Reverie semantic search.

Exposes tools via the Model Context Protocol:
- semantic_search: Query the knowledge graph for relevant passages
- recall_enrich: Enrich a prompt with relevant long-term memory context
- memory_status: Get indexing statistics

Uses MemoryLink v2 pipeline (LinearRAG-based bipartite graph).
Includes IndexScheduler for incremental indexing on startup.
Also serves the Memory Explorer dashboard at root path.
"""

import asyncio
import logging
import os
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Lazy-initialized singletons with thread-safe initialization
_pipeline = None  # MemoryLinkV2Pipeline
_conn = None
_db = None  # ReverieDB instance
_shared_components = None  # (cache, embedding_client)
_scheduler = None  # IndexScheduler

# Lock for thread-safe initialization of singletons
# Use RLock (reentrant lock) because _get_pipeline() calls _get_connection()
# and _get_shared_components() while holding the lock
_init_lock = threading.RLock()


def _get_db():
    """Get or create a ReverieDB instance for the MCP server.

    Uses the factory to create the appropriate database backend.
    The ReverieDB wraps a dedicated connection (not from pool) to avoid
    pool exhaustion - the MCP server holds a single long-lived connection
    for the lifetime of the process.

    Thread-safe: Uses _init_lock to prevent race conditions during
    initialization when multiple requests arrive simultaneously.

    Returns:
        ReverieDB implementation instance.
    """
    global _db
    # Quick check without lock for performance
    if _db is not None and not _db.closed:
        return _db

    with _init_lock:
        # Double-check after acquiring lock (another thread may have initialized)
        if _db is None or _db.closed:
            from reverie.db.factory import create_db, get_backend_from_settings
            from reverie.config import get_settings

            settings = get_settings()
            backend = get_backend_from_settings()

            try:
                _db = create_db(
                    backend=backend,
                    host=settings.db_host,
                    port=settings.db_port,
                    user=settings.db_user,
                    password=settings.db_password,
                    dbname=settings.db_name,
                )
                # Set autocommit on the underlying connection for PostgreSQL
                if hasattr(_db, 'raw_connection'):
                    _db.raw_connection.autocommit = True
                logger.info(f"ReverieDB initialized (backend={backend})")
            except Exception as e:
                # Mask sensitive info in error message
                logger.error(
                    "Database connection failed: host=%s port=%s dbname=%s user=%s",
                    settings.db_host,
                    settings.db_port,
                    settings.db_name,
                    settings.db_user,
                )
                raise RuntimeError(
                    f"Failed to connect to database at {settings.db_host}:{settings.db_port}"
                ) from e
        return _db


def _get_connection():
    """Get or create a dedicated database connection for the MCP server.

    DEPRECATED: Use _get_db() instead. This function is kept for backward
    compatibility during migration. Returns the raw connection from ReverieDB.

    Thread-safe: Uses _init_lock to prevent race conditions during
    initialization when multiple requests arrive simultaneously.
    """
    db = _get_db()
    # Return raw connection for backward compatibility
    if hasattr(db, 'raw_connection'):
        return db.raw_connection
    # Fallback for non-PostgreSQL backends (future)
    raise RuntimeError("raw_connection not available for this backend")


def _get_shared_components():
    """Get or create shared components (cache, embedding).

    Thread-safe: Uses _init_lock to prevent race conditions during
    initialization when multiple requests arrive simultaneously.
    """
    global _shared_components
    # Quick check without lock for performance
    if _shared_components is not None:
        return _shared_components

    with _init_lock:
        # Double-check after acquiring lock
        if _shared_components is None:
            from reverie.config import get_settings
            from reverie.cache.api_cache import ApiCache
            from reverie.indexer.embedding import EmbeddingClient

            settings = get_settings()

            cache = ApiCache(
                cache_dir=settings.cache_dir,
                enabled=settings.cache_enabled,
            )
            embedding_client = EmbeddingClient(
                api_key=settings.openai_api_key,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
                cache=cache,
            )
            _shared_components = (cache, embedding_client)

        return _shared_components


def _get_pipeline():
    """Get or create the MemoryLink v2 retrieval pipeline.

    Thread-safe: Uses _init_lock to prevent race conditions during
    pipeline initialization when multiple requests arrive simultaneously.

    Returns:
        MemoryLinkV2Pipeline instance.
    """
    global _pipeline

    # Quick check without lock for performance
    if _pipeline is not None:
        return _pipeline

    with _init_lock:
        # Double-check after acquiring lock
        if _pipeline is None:
            from reverie.config import get_settings
            from reverie.retriever.pipeline_v2 import MemoryLinkV2Pipeline

            settings = get_settings()
            db = _get_db()
            _, embedding_client = _get_shared_components()

            _pipeline = MemoryLinkV2Pipeline(
                db=db,
                embedding_client=embedding_client,
                tau=settings.v2_tau,
                delta=settings.v2_delta,
                damping=settings.v2_damping,
                max_iter=settings.v2_max_iter,
                beta=settings.v2_beta,
                graph_ttl_seconds=settings.graph_ttl_seconds,
                use_server_side_ppr=settings.v2_use_server_side_ppr,
            )

        return _pipeline


def _get_scheduler():
    """Get or create the IndexScheduler.

    Thread-safe: Uses _init_lock to prevent race conditions during
    scheduler initialization.

    Returns:
        IndexScheduler instance (not yet started).
    """
    global _scheduler

    # Quick check without lock for performance
    if _scheduler is not None:
        return _scheduler

    with _init_lock:
        # Double-check after acquiring lock
        if _scheduler is None:
            from reverie.config import get_settings
            from reverie.indexer.scheduler import IndexScheduler

            settings = get_settings()

            _scheduler = IndexScheduler(
                num_workers=settings.index_workers,
                debounce_seconds=settings.index_debounce_seconds,
            )
            logger.info("IndexScheduler initialized")

        return _scheduler


def _can_start_scheduler() -> tuple[bool, str]:
    """Check if scheduler can be started.

    Returns:
        Tuple of (can_start, reason).
    """
    # Check if setup is complete
    from reverie.config.setup_state import SetupState
    setup_state = SetupState()
    if not setup_state.is_complete():
        return False, "Setup not complete"

    # Check if profiles are configured
    try:
        from reverie.config import get_profiles
        profiles = get_profiles()
        if not profiles:
            return False, "No profiles configured (REVERIE_PROFILES empty or not set)"
    except Exception as e:
        return False, f"Failed to load profiles: {e}"

    return True, "Ready"


async def _start_scheduler():
    """Start the IndexScheduler (runs startup scan).

    This should be called when the MCP server starts.
    Only starts if setup is complete and profiles are configured.
    """
    can_start, reason = _can_start_scheduler()
    if not can_start:
        logger.info(f"IndexScheduler not starting: {reason}")
        return

    try:
        scheduler = _get_scheduler()
        if not scheduler.is_running:
            await scheduler.start(run_startup_scan=True)
            logger.info("IndexScheduler started with startup scan")
    except Exception as e:
        logger.error(f"Failed to start IndexScheduler: {e}", exc_info=True)


async def _stop_scheduler():
    """Stop the IndexScheduler gracefully."""
    global _scheduler
    if _scheduler is not None and _scheduler.is_running:
        try:
            await _scheduler.stop()
            logger.info("IndexScheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping IndexScheduler: {e}", exc_info=True)


def _get_process_memory_mb() -> float:
    """Get current process memory usage in MB (RSS)."""
    try:
        # Read from /proc/self/statm (Linux)
        with open("/proc/self/statm", "r") as f:
            parts = f.read().split()
            # Second value is RSS in pages
            rss_pages = int(parts[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (rss_pages * page_size) / (1024 * 1024)
    except (FileNotFoundError, IndexError, ValueError):
        # Fallback for non-Linux systems
        try:
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss is in KB on Linux, bytes on macOS
            return rusage.ru_maxrss / 1024  # Assume KB
        except Exception:
            return 0.0


def _get_stats() -> dict:
    """Get memory index statistics from DB."""
    db = _get_db()
    stats = db.get_stats()

    # Add process memory usage
    stats["process_memory_mb"] = round(_get_process_memory_mb(), 1)

    return stats


# Global indexer state for status reporting
_indexer_state: dict = {
    "running": False,
    "current_session": None,
    "current_turn": None,
    "queue_size": 0,
    "workers": 0,
}


def set_indexer_state(
    running: bool = None,
    current_session: str = None,
    current_turn: int = None,
    queue_size: int = None,
    workers: int = None,
) -> None:
    """Update indexer state for status reporting."""
    if running is not None:
        _indexer_state["running"] = running
    if current_session is not None:
        _indexer_state["current_session"] = current_session
    if current_turn is not None:
        _indexer_state["current_turn"] = current_turn
    if queue_size is not None:
        _indexer_state["queue_size"] = queue_size
    if workers is not None:
        _indexer_state["workers"] = workers


def _register_index_status_route(mcp: FastMCP) -> None:
    """Register the /api/index/status endpoint."""
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    def _cors_headers() -> dict:
        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }

    @mcp.custom_route("/api/index/status", methods=["GET"])
    async def index_status(request: Request):
        """Return indexer status from scheduler."""
        global _scheduler

        # Get live status from scheduler if available
        if _scheduler is not None and _scheduler.is_running:
            stats = _scheduler.stats
            pool_stats = stats.get("pool", {})
            queue_stats = stats.get("queue", {})

            return JSONResponse({
                "running": pool_stats.get("active_workers", 0) > 0 or queue_stats.get("size", 0) > 0,
                "current_session": _indexer_state.get("current_session"),
                "current_turn": _indexer_state.get("current_turn"),
                "queue_size": queue_stats.get("size", 0),
                "workers": pool_stats.get("num_workers", 0),
                "active_sessions": pool_stats.get("active_sessions", []),
            }, headers=_cors_headers())

        # Fallback to cached state
        return JSONResponse(_indexer_state, headers=_cors_headers())

    @mcp.custom_route("/api/index/{rest:path}", methods=["OPTIONS"])
    async def index_options(request: Request):
        """Handle CORS preflight for index API routes."""
        from starlette.responses import Response
        return Response(status_code=204, headers=_cors_headers())

    logger.info("Index status route registered")


def _mount_dashboard(mcp: FastMCP) -> None:
    """Mount the Memory Explorer dashboard static files.

    Looks for built dashboard files in dashboard_dist/ directory
    (relative to this package). Falls back gracefully if not found,
    providing a helpful message at /reverie for users.
    """
    from starlette.responses import FileResponse, Response, HTMLResponse
    from starlette.requests import Request

    # Dashboard files location: src/reverie/dashboard_dist/
    package_dir = Path(__file__).parent.parent
    dashboard_dir = package_dir / "dashboard_dist"

    if not dashboard_dir.exists():
        logger.warning(
            f"Dashboard directory not found at {dashboard_dir}. "
            "Dashboard will not be available. Build with: ./scripts/build_dashboard.sh"
        )
        # Register a fallback route that shows a helpful message
        @mcp.custom_route("/reverie", methods=["GET"])
        @mcp.custom_route("/reverie/{path:path}", methods=["GET"])
        async def dashboard_not_built(request: Request) -> Response:
            """Show helpful message when dashboard is not built."""
            return HTMLResponse(
                content="""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Reverie Dashboard - Not Built</title>
                    <style>
                        body { font-family: system-ui, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
                        h1 { color: #e74c3c; }
                        code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
                        pre { background: #f4f4f4; padding: 15px; border-radius: 8px; overflow-x: auto; }
                        .note { background: #fff3cd; padding: 15px; border-radius: 8px; margin-top: 20px; }
                    </style>
                </head>
                <body>
                    <h1>Dashboard Not Built</h1>
                    <p>The Reverie dashboard has not been built yet.</p>
                    <p>To build the dashboard, run:</p>
                    <pre>./scripts/build_dashboard.sh</pre>
                    <p>Or manually:</p>
                    <pre>cd dashboard
npm install
npm run build
cp -r dist ../src/reverie/dashboard_dist</pre>
                    <div class="note">
                        <strong>Note:</strong> The Setup Wizard API is still available at
                        <code>/api/setup/status</code>. You can use any HTTP client to interact with it.
                    </div>
                </body>
                </html>
                """,
                status_code=503,
            )
        return

    index_file = dashboard_dir / "index.html"
    if not index_file.exists():
        logger.warning(
            f"Dashboard index.html not found at {index_file}. "
            "Dashboard will not be available."
        )
        return

    logger.info(f"Mounting dashboard from {dashboard_dir}")

    @mcp.custom_route("/reverie/{path:path}", methods=["GET"])
    async def serve_dashboard(request: Request) -> Response:
        """Serve dashboard static files with SPA fallback."""
        path = request.path_params.get("path", "")

        # Try to serve the requested file
        if path:
            file_path = dashboard_dir / path
            if file_path.exists() and file_path.is_file():
                # Determine content type
                suffix = file_path.suffix.lower()
                content_types = {
                    ".html": "text/html",
                    ".js": "application/javascript",
                    ".css": "text/css",
                    ".json": "application/json",
                    ".svg": "image/svg+xml",
                    ".png": "image/png",
                    ".ico": "image/x-icon",
                    ".woff": "font/woff",
                    ".woff2": "font/woff2",
                }
                media_type = content_types.get(suffix, "application/octet-stream")
                return FileResponse(file_path, media_type=media_type)

        # SPA fallback: serve index.html for unmatched routes
        return FileResponse(index_file, media_type="text/html")

    @mcp.custom_route("/reverie", methods=["GET"])
    async def serve_dashboard_root(request: Request) -> Response:
        """Serve dashboard index at /reverie (without trailing slash)."""
        return FileResponse(index_file, media_type="text/html")

    # Also serve assets from /reverie/assets/
    assets_dir = dashboard_dir / "assets"
    if assets_dir.exists():
        @mcp.custom_route("/reverie/assets/{path:path}", methods=["GET"])
        async def serve_assets(request: Request) -> Response:
            """Serve dashboard assets."""
            path = request.path_params.get("path", "")
            file_path = assets_dir / path
            if file_path.exists() and file_path.is_file():
                suffix = file_path.suffix.lower()
                content_types = {
                    ".js": "application/javascript",
                    ".css": "text/css",
                    ".json": "application/json",
                    ".svg": "image/svg+xml",
                    ".png": "image/png",
                    ".woff": "font/woff",
                    ".woff2": "font/woff2",
                }
                media_type = content_types.get(suffix, "application/octet-stream")
                return FileResponse(file_path, media_type=media_type)
            return Response(status_code=404)


def create_server(host: str = "127.0.0.1", port: int = 8004) -> FastMCP:
    """Create and configure the MCP server with tools."""
    mcp = FastMCP("reverie", host=host, port=port)

    # Register Setup Wizard API routes (must be before other routes for /api/setup/*)
    from reverie.api.setup_routes import register_setup_routes
    register_setup_routes(mcp)

    # Register REST API routes for Memory Explorer dashboard
    from reverie.api.routes import register_routes

    register_routes(
        mcp,
        get_db=_get_db,
        get_stats=_get_stats,
        get_pipeline=_get_pipeline,
    )

    # Register simple index status endpoint
    _register_index_status_route(mcp)

    # Serve dashboard static files
    _mount_dashboard(mcp)

    @mcp.tool()
    async def semantic_search(query: str, top_k: int = 5) -> str:
        """Search long-term memory for relevant passages.

        Uses MemoryLink v2 bipartite graph with Semantic Propagation + PPR
        to find contextually relevant information from past sessions.

        Args:
            query: Natural language search query.
            top_k: Maximum number of passages to return (default: 5).

        Returns:
            Formatted search results with passage text, score, and source.
        """
        try:
            pipeline = _get_pipeline()
            results = pipeline.retrieve(query, top_k=top_k)

            if not results:
                return "No results found for the given query."

            lines = [f"Found {len(results)} results:\n"]
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] (score: {r.score:.2f}, source: {r.source})\n"
                    f"{r.text}\n"
                )
            return "\n".join(lines)

        except Exception as e:
            logger.exception("semantic_search error")
            return f"Error during search: {e}"

    @mcp.tool()
    async def recall_enrich(
        prompt: str,
        max_tokens: int = 4000,
        max_passages: int = 5,
    ) -> str:
        """Enrich a prompt with relevant long-term memory context.

        Retrieves passages from the knowledge graph and prepends
        them as a context block before the original prompt. If no relevant
        passages are found or an error occurs, returns the original prompt
        unmodified.

        Args:
            prompt: The original prompt to enrich.
            max_tokens: Token budget for the memory context block (default: 4000).
            max_passages: Maximum passages to include (default: 5).

        Returns:
            The enriched prompt with memory context, or original prompt on failure.
        """
        try:
            from reverie.recall.service import RecallService

            pipeline = _get_pipeline()
            service = RecallService(
                pipeline=pipeline,
                max_tokens=max_tokens,
                max_passages=max_passages,
            )
            return service.enrich_prompt(prompt)

        except Exception as e:
            logger.exception("recall_enrich error")
            return prompt

    @mcp.tool()
    async def memory_status() -> str:
        """Get Reverie indexing statistics.

        Returns counts of entities, passages, edges, indexed sessions,
        last indexing timestamp, process memory usage, and scheduler status.
        """
        try:
            stats = _get_stats()

            lines = [
                "Reverie Status (v2):",
                f"  Entities:         {stats['entities']}",
                f"  Passages:         {stats['passages']}",
                f"  Entity-Passages:  {stats['entity_passages']}",
                f"  Passage-Adjacency: {stats['passage_adjacency']}",
                f"  Sessions:         {stats['sessions']}",
                f"  Last indexed:     {stats['last_indexed']}",
                "",
                "Process Memory:",
                f"  RSS: {stats['process_memory_mb']} MB",
            ]

            # Add scheduler stats if available
            if _scheduler is not None:
                sched_stats = _scheduler.stats
                lines.extend([
                    "",
                    "Index Scheduler:",
                    f"  Running: {sched_stats['running']}",
                    f"  Queue pending: {sched_stats['queue'].get('pending', 0)}",
                    f"  Queue completed: {sched_stats['queue'].get('completed', 0)}",
                    f"  Queue failed: {sched_stats['queue'].get('failed', 0)}",
                    f"  Workers active: {sched_stats['pool'].get('active', 0)}",
                ])

            return "\n".join(lines)
        except Exception as e:
            logger.exception("memory_status error")
            return f"Error retrieving status: {e}"

    @mcp.tool()
    async def invalidate_graph() -> str:
        """Invalidate the cached bipartite graph.

        Forces the graph to be rebuilt on the next retrieval query.
        Use this after indexing new data to ensure the graph reflects
        the latest database state.

        Returns:
            Confirmation message.
        """
        try:
            pipeline = _get_pipeline()
            pipeline.invalidate_graph()
            logger.info("Invalidated graph cache")

            return "Graph cache invalidated. Next query will rebuild the graph."
        except Exception as e:
            logger.exception("invalidate_graph error")
            return f"Error invalidating graph: {e}"

    return mcp


async def _run_with_scheduler(server, transport: str, mount_path: str | None = None):
    """Run server with IndexScheduler lifecycle management.

    Starts IndexScheduler before server, stops it on shutdown.
    """
    try:
        await _start_scheduler()

        if transport == "sse":
            await server.run_sse_async(mount_path)
        elif transport == "streamable-http":
            await server.run_streamable_http_async()
        else:
            await server.run_stdio_async()
    finally:
        await _stop_scheduler()
        # Clean up database connection
        global _db
        if _db is not None and not _db.closed:
            _db.close()
            logger.info("Memory MCP: Database connection closed")


def _configure_logging():
    """Configure logging level from settings.

    Falls back to INFO level with basic logging if settings cannot be loaded.
    This allows the server to bootstrap even without a valid .env file,
    so the Setup Wizard can be reached.
    """
    level = logging.INFO
    log_level_str = "INFO"

    try:
        from reverie.config import get_settings
        settings = get_settings()
        level = getattr(logging, settings.log_level.upper(), logging.INFO)
        log_level_str = settings.log_level
    except Exception as e:
        # Settings not available (likely no .env) - use defaults
        # This allows bootstrap to Setup Wizard
        print(f"[Reverie] Using default logging (settings unavailable: {e})")

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set reverie package logger level
    reverie_logger = logging.getLogger("reverie")
    reverie_logger.setLevel(level)

    logger.info(f"Logging configured: level={log_level_str}")


def _prompt_yes_no(message: str) -> bool:
    """Prompt user for yes/no confirmation.

    Returns:
        True if user confirms, False otherwise.
    """
    try:
        response = input(f"{message} [y/N]: ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _install_nodejs() -> bool:
    """Attempt to install Node.js.

    Returns:
        True if installation succeeded, False otherwise.
    """
    import subprocess
    import shutil
    from reverie.docker.docker_manager import get_os_type, OSType

    os_type = get_os_type()

    if os_type == OSType.WINDOWS:
        print("[Reverie] Attempting to install Node.js on Windows...")
        try:
            # Try winget first
            if shutil.which("winget"):
                print("[Reverie] Running: winget install OpenJS.NodeJS.LTS")
                result = subprocess.run(
                    ["winget", "install", "OpenJS.NodeJS.LTS", "--accept-package-agreements", "--accept-source-agreements"],
                    capture_output=True,
                    timeout=300,
                )
                if result.returncode == 0 and shutil.which("npm"):
                    print("[Reverie] Node.js installed successfully via winget!")
                    return True

            # Try fnm via PowerShell (irm/iex pattern)
            print("[Reverie] Trying fnm installation via PowerShell...")
            ps_script = 'irm https://fnm.vercel.app/install | iex; fnm install --lts; fnm use lts-latest'
            result = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                timeout=300,
            )
            if result.returncode == 0 and shutil.which("npm"):
                print("[Reverie] Node.js installed successfully via fnm!")
                return True

        except subprocess.TimeoutExpired:
            print("[Reverie] Node.js installation timed out")
        except Exception as e:
            print(f"[Reverie] Node.js installation error: {e}")

    elif os_type == OSType.LINUX:
        print("[Reverie] Attempting to install Node.js on Linux...")
        try:
            # Try apt-get (Debian/Ubuntu)
            if shutil.which("apt-get"):
                print("[Reverie] Running: sudo apt-get update && sudo apt-get install -y nodejs npm")
                subprocess.run(["sudo", "apt-get", "update"], capture_output=True, timeout=120)
                result = subprocess.run(
                    ["sudo", "apt-get", "install", "-y", "nodejs", "npm"],
                    capture_output=True,
                    timeout=300,
                )
                if result.returncode == 0 and shutil.which("npm"):
                    print("[Reverie] Node.js installed successfully!")
                    return True

            # Try dnf (Fedora/RHEL)
            elif shutil.which("dnf"):
                print("[Reverie] Running: sudo dnf install -y nodejs npm")
                result = subprocess.run(
                    ["sudo", "dnf", "install", "-y", "nodejs", "npm"],
                    capture_output=True,
                    timeout=300,
                )
                if result.returncode == 0 and shutil.which("npm"):
                    print("[Reverie] Node.js installed successfully!")
                    return True

        except subprocess.TimeoutExpired:
            print("[Reverie] Node.js installation timed out")
        except Exception as e:
            print(f"[Reverie] Node.js installation error: {e}")

    elif os_type == OSType.MACOS:
        print("[Reverie] Attempting to install Node.js on macOS...")
        try:
            if shutil.which("brew"):
                print("[Reverie] Running: brew install node")
                result = subprocess.run(
                    ["brew", "install", "node"],
                    capture_output=True,
                    timeout=300,
                )
                if result.returncode == 0 and shutil.which("npm"):
                    print("[Reverie] Node.js installed successfully!")
                    return True
        except Exception as e:
            print(f"[Reverie] Node.js installation error: {e}")

    # Installation failed - show manual guide
    print("\n[Reverie] Automatic installation failed. Please install Node.js manually:")
    print("  - Windows: https://nodejs.org/ or 'winget install OpenJS.NodeJS.LTS'")
    print("  - macOS: 'brew install node' or https://nodejs.org/")
    print("  - Linux: Use your package manager or https://nodejs.org/\n")
    return False


def _compute_source_hash(directory: Path, extensions: set[str]) -> str:
    """Compute a hash of source file contents.

    Args:
        directory: Directory to scan.
        extensions: File extensions to check (e.g., {'.ts', '.tsx'}).

    Returns:
        MD5 hash of all source file contents combined.
    """
    import hashlib

    hasher = hashlib.md5()
    try:
        # Sort files for consistent ordering
        files = sorted(
            f for f in directory.rglob("*")
            if f.is_file() and f.suffix in extensions
        )
        for file in files:
            # Include relative path in hash to detect renames
            hasher.update(str(file.relative_to(directory)).encode())
            hasher.update(file.read_bytes())
    except OSError:
        pass
    return hasher.hexdigest()


def _ensure_dashboard_built(force_rebuild: bool = False) -> bool:
    """Ensure dashboard is built, building if necessary.

    Args:
        force_rebuild: If True, rebuild even if dashboard already exists.

    Returns:
        True if dashboard is available, False otherwise.
    """
    package_dir = Path(__file__).parent.parent
    dashboard_dist = package_dir / "dashboard_dist"
    project_root = package_dir.parent.parent
    dashboard_dir = project_root / "dashboard"

    index_html = dashboard_dist / "index.html"
    build_hash_file = dashboard_dist / ".build-hash"

    if not force_rebuild and index_html.exists():
        # Check if source files changed using content hash
        if dashboard_dir.exists():
            src_dir = dashboard_dir / "src"
            if src_dir.exists():
                current_hash = _compute_source_hash(src_dir, {".ts", ".tsx", ".css"})
                stored_hash = ""
                if build_hash_file.exists():
                    stored_hash = build_hash_file.read_text().strip()

                if current_hash == stored_hash:
                    # Source unchanged, no need to rebuild
                    return True
                print("[Reverie] Dashboard source changed, rebuilding...")
            else:
                return True
        else:
            return True

    if not dashboard_dir.exists():
        print("[Reverie] Dashboard source not found, skipping build")
        return False

    already_exists = index_html.exists()
    if not already_exists:
        print("[Reverie] Building dashboard...")

    import subprocess
    import shutil

    # Check if npm is available
    if not shutil.which("npm"):
        print("[Reverie] npm not found.")
        if not _prompt_yes_no("[Reverie] Would you like to install Node.js?"):
            print("[Reverie] Node.js is required to build the dashboard.")
            print("[Reverie] Exiting.\n")
            import sys
            sys.exit(1)

        if not _install_nodejs():
            print("[Reverie] Could not install Node.js.")
            print("[Reverie] Please install Node.js manually and restart.\n")
            import sys
            sys.exit(1)

        # Refresh PATH check
        if not shutil.which("npm"):
            print("[Reverie] npm still not found after installation.")
            print("[Reverie] You may need to restart your terminal or add Node.js to PATH.")
            print("[Reverie] Exiting.\n")
            import sys
            sys.exit(1)

    try:
        # Always use npm directly (cross-platform)
        # Use encoding="utf-8" and errors="replace" for Windows compatibility
        node_modules = dashboard_dir / "node_modules"
        if node_modules.exists() and (node_modules / ".package-lock.json").exists():
            # node_modules already exists, skip npm ci
            print("[Reverie] node_modules found, skipping npm ci...")
        else:
            print("[Reverie] Running npm ci...")
            install_result = subprocess.run(
                ["npm", "ci"],
                cwd=str(dashboard_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                shell=(os.name == "nt"),  # Use shell on Windows
            )
            if install_result.returncode != 0:
                print(f"[Reverie] npm ci failed: {install_result.stderr}")
                # Try npm install as fallback
                print("[Reverie] Trying npm install...")
                install_result = subprocess.run(
                    ["npm", "install"],
                    cwd=str(dashboard_dir),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=180,
                    shell=(os.name == "nt"),
                )

        print("[Reverie] Running npm run build...")
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(dashboard_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            shell=(os.name == "nt"),
        )

        # Copy dist to dashboard_dist
        dist_dir = dashboard_dir / "dist"
        if dist_dir.exists():
            if dashboard_dist.exists():
                shutil.rmtree(dashboard_dist)
            shutil.copytree(dist_dir, dashboard_dist)

        if dashboard_dist.exists() and (dashboard_dist / "index.html").exists():
            # Save source hash to detect future changes
            src_dir = dashboard_dir / "src"
            if src_dir.exists():
                current_hash = _compute_source_hash(src_dir, {".ts", ".tsx", ".css"})
                build_hash_file = dashboard_dist / ".build-hash"
                build_hash_file.write_text(current_hash)
            print("[Reverie] Dashboard built successfully!")
            return True
        else:
            print(f"[Reverie] Dashboard build failed: {result.stderr if result else 'unknown error'}")
            return False

    except subprocess.TimeoutExpired:
        print("[Reverie] Dashboard build timed out")
        return False
    except Exception as e:
        print(f"[Reverie] Dashboard build error: {e}")
        return False


def _install_docker() -> bool:
    """Attempt to install Docker.

    Returns:
        True if installation succeeded, False otherwise.
    """
    import subprocess
    import shutil
    from reverie.docker.docker_manager import DockerManager, get_os_type, OSType

    os_type = get_os_type()
    docker_mgr = DockerManager()

    if os_type == OSType.WINDOWS:
        print("[Reverie] Attempting to install Docker on Windows...")
        try:
            # Try winget
            if shutil.which("winget"):
                print("[Reverie] Running: winget install Docker.DockerDesktop")
                result = subprocess.run(
                    ["winget", "install", "Docker.DockerDesktop", "--accept-package-agreements", "--accept-source-agreements"],
                    capture_output=True,
                    timeout=600,  # Docker Desktop is large
                )
                if result.returncode == 0:
                    print("[Reverie] Docker Desktop installed.")
                    print("[Reverie] NOTE: You need to start Docker Desktop and restart your terminal.")
                    return True

            # Alternative: chocolatey
            if shutil.which("choco"):
                print("[Reverie] Running: choco install docker-desktop -y")
                result = subprocess.run(
                    ["choco", "install", "docker-desktop", "-y"],
                    capture_output=True,
                    timeout=600,
                )
                if result.returncode == 0:
                    print("[Reverie] Docker Desktop installed via Chocolatey.")
                    return True

        except subprocess.TimeoutExpired:
            print("[Reverie] Docker installation timed out")
        except Exception as e:
            print(f"[Reverie] Docker installation error: {e}")

    elif os_type == OSType.LINUX:
        print("[Reverie] Attempting to install Docker on Linux...")
        try:
            # apt-get (Debian/Ubuntu)
            if shutil.which("apt-get"):
                print("[Reverie] Running: sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin")
                subprocess.run(["sudo", "apt-get", "update"], capture_output=True, timeout=120)
                result = subprocess.run(
                    ["sudo", "apt-get", "install", "-y", "docker.io", "docker-compose-plugin"],
                    capture_output=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    subprocess.run(["sudo", "systemctl", "start", "docker"], capture_output=True)
                    subprocess.run(["sudo", "systemctl", "enable", "docker"], capture_output=True)

                    import os as os_module
                    user = os_module.environ.get("USER", "")
                    if user:
                        subprocess.run(["sudo", "usermod", "-aG", "docker", user], capture_output=True)
                        print(f"[Reverie] Added {user} to docker group")
                        print("[Reverie] NOTE: Log out and back in for group changes to take effect")

                    check_result = docker_mgr.check_docker()
                    if check_result.installed:
                        print("[Reverie] Docker installed successfully!")
                        return True

            # dnf (Fedora/RHEL)
            elif shutil.which("dnf"):
                print("[Reverie] Running: sudo dnf install -y docker docker-compose-plugin")
                result = subprocess.run(
                    ["sudo", "dnf", "install", "-y", "docker", "docker-compose-plugin"],
                    capture_output=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    subprocess.run(["sudo", "systemctl", "start", "docker"], capture_output=True)
                    subprocess.run(["sudo", "systemctl", "enable", "docker"], capture_output=True)

                    import os as os_module
                    user = os_module.environ.get("USER", "")
                    if user:
                        subprocess.run(["sudo", "usermod", "-aG", "docker", user], capture_output=True)

                    check_result = docker_mgr.check_docker()
                    if check_result.installed:
                        print("[Reverie] Docker installed successfully!")
                        return True

        except subprocess.TimeoutExpired:
            print("[Reverie] Docker installation timed out")
        except Exception as e:
            print(f"[Reverie] Docker installation error: {e}")

    elif os_type == OSType.MACOS:
        print("[Reverie] Attempting to install Docker on macOS...")
        try:
            if shutil.which("brew"):
                print("[Reverie] Running: brew install --cask docker")
                result = subprocess.run(
                    ["brew", "install", "--cask", "docker"],
                    capture_output=True,
                    timeout=600,
                )
                if result.returncode == 0:
                    print("[Reverie] Docker Desktop installed via Homebrew.")
                    print("[Reverie] NOTE: You need to start Docker Desktop from Applications.")
                    return True
        except Exception as e:
            print(f"[Reverie] Docker installation error: {e}")

    return False


def _ensure_docker_running() -> bool:
    """Ensure Docker daemon is running, starting Docker Desktop on Windows if needed.

    This function is called at server startup when using PostgreSQL backend.
    On Windows, if Docker daemon is not running, it attempts to start Docker Desktop
    automatically and waits for it to become ready.

    Returns:
        True if Docker is ready or not needed, False if failed.
    """
    from reverie.db.factory import get_backend_from_settings
    from reverie.docker.docker_manager import (
        DockerManager,
        get_os_type,
        OSType,
        DOCKER_INSTALL_GUIDES,
        ContainerStatus,
    )
    from reverie.config import get_settings

    # Skip Docker check for SQLite backend
    backend = get_backend_from_settings()
    if backend != "postgres":
        return True

    docker_mgr = DockerManager()
    result = docker_mgr.check_docker()

    # If Docker is not installed at all, fall through to _ensure_docker_available
    if not result.installed:
        return _ensure_docker_available()

    # If daemon is running, check/start PostgreSQL container
    if result.daemon_running:
        status = docker_mgr.get_status()
        if status.status == ContainerStatus.RUNNING:
            print("[Reverie] Docker and PostgreSQL container ready")
            return True

        # Container not running, start it
        print("[Reverie] Starting PostgreSQL container...")
        settings = get_settings()
        success, msg = docker_mgr.start(
            postgres_user=settings.db_user,
            postgres_password=settings.db_password,
            postgres_db=settings.db_name,
            postgres_port=settings.db_port,
            wait_ready=True,
        )
        if success:
            print(f"[Reverie] {msg}")
            return True
        else:
            print(f"[Reverie] Failed to start PostgreSQL: {msg}")
            return False

    # Daemon not running - try to start Docker Desktop on Windows
    os_type = get_os_type()
    if os_type == OSType.WINDOWS:
        print("[Reverie] Docker daemon not running. Starting Docker Desktop...")
        success, msg = docker_mgr.start_docker_desktop(timeout=120)
        if success:
            print(f"[Reverie] {msg}")
            # Now start PostgreSQL container
            print("[Reverie] Starting PostgreSQL container...")
            settings = get_settings()
            success, msg = docker_mgr.start(
                postgres_user=settings.db_user,
                postgres_password=settings.db_password,
                postgres_db=settings.db_name,
                postgres_port=settings.db_port,
                wait_ready=True,
            )
            if success:
                print(f"[Reverie] {msg}")
                return True
            else:
                print(f"[Reverie] Failed to start PostgreSQL: {msg}")
                return False
        else:
            print(f"[Reverie] {msg}")
            return False

    # Non-Windows: show guidance
    guide = DOCKER_INSTALL_GUIDES.get(os_type, {})
    print(f"\n[Reverie] Docker daemon is not running.")
    print(f"[Reverie] Please start Docker and try again.")
    if guide:
        print(f"[Reverie] See: {guide.get('url', 'https://docs.docker.com/')}")
    return False


def _ensure_docker_available() -> bool:
    """Ensure Docker is available, attempting installation if needed.

    Returns:
        True if Docker is available, False otherwise.
    """
    from reverie.docker.docker_manager import DockerManager, get_os_type, OSType, DOCKER_INSTALL_GUIDES

    docker_mgr = DockerManager()
    result = docker_mgr.check_docker()

    if result.installed and result.compose_available:
        return True

    os_type = get_os_type()
    print(f"\n[Reverie] Docker not available on {os_type.value}")

    # Ask user before attempting installation
    if not _prompt_yes_no("[Reverie] Would you like to install Docker?"):
        print("[Reverie] Skipping Docker installation.")
        print("[Reverie] Docker is required for the PostgreSQL database.")
        print("[Reverie] Exiting.\n")
        import sys
        sys.exit(1)

    # Try automatic installation
    if _install_docker():
        # Re-check
        result = docker_mgr.check_docker()
        if result.installed:
            return True

    # Show installation guide on failure
    guide = DOCKER_INSTALL_GUIDES.get(os_type)
    if guide:
        print(f"\n{'='*60}")
        print(f"Docker Installation Guide: {guide['title']}")
        print(f"{'='*60}")
        print(f"Documentation: {guide['url']}\n")
        print("Steps:")
        for step in guide["steps"]:
            print(f"  {step}")
        print("\nNotes:")
        for note in guide["notes"]:
            print(f"  - {note}")
        print(f"{'='*60}\n")
    else:
        print("[Reverie] Please install Docker manually from https://docs.docker.com/get-docker/")

    print("[Reverie] Docker is required for the PostgreSQL database.")
    print("[Reverie] Please install Docker and restart.\n")
    import sys
    sys.exit(1)


def main():
    """Run the MCP server."""
    import sys
    import signal

    # Configure logging before anything else
    _configure_logging()

    # Check for flags
    skip_checks = "--skip-checks" in sys.argv

    if not skip_checks:
        # Build dashboard if not already built
        _ensure_dashboard_built(force_rebuild=False)

        # Ensure Docker and PostgreSQL are running (for postgres backend)
        if not _ensure_docker_running():
            sys.exit(1)

    transport = "sse" if "--sse" in sys.argv else "stdio"
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8004
    host = "0.0.0.0" if transport == "sse" else "127.0.0.1"

    server = create_server(host=host, port=port)

    # Force exit handler for repeated Ctrl+C
    force_exit_count = 0

    def force_exit_handler(signum, frame):
        nonlocal force_exit_count
        force_exit_count += 1
        if force_exit_count >= 2:
            print("\n[Reverie] Force exit...")
            import os
            os._exit(1)
        print("\n[Reverie] Cleaning up async tasks... (press Ctrl+C again to force)")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, force_exit_handler)

    # Use custom runner that manages IndexScheduler lifecycle
    try:
        asyncio.run(_run_with_scheduler(server, transport))
    except (KeyboardInterrupt, SystemExit):
        print("[Reverie] Shutdown complete.")


if __name__ == "__main__":
    main()
