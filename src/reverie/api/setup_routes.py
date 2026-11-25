"""Setup API routes for Reverie Setup Wizard.

Exposes HTTP endpoints for the Setup Wizard:
- GET  /api/setup/status       - Get overall setup status
- POST /api/setup/check/postgres - Test PostgreSQL connection
- POST /api/setup/check/pgvector - Check pgvector extension
- GET  /api/setup/check/docker - Check Docker installation
- POST /api/setup/docker/start - Start PostgreSQL container
- POST /api/setup/docker/stop  - Stop PostgreSQL container
- GET  /api/setup/docker/status - Get container status
- POST /api/setup/database     - Save database configuration
- POST /api/setup/apikey       - Save and validate API key
- POST /api/setup/profile      - Save profile configuration
- POST /api/setup/migrate      - Run database migrations
- GET  /api/setup/verify       - Run verification checks
- POST /api/setup/complete     - Mark setup as complete
"""

import logging
import os
import threading
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

try:
    import openai
except ImportError:
    openai = None

from starlette.requests import Request
from starlette.responses import JSONResponse

from reverie.config.setup_state import SetupState, PhaseStatus
from reverie.docker.docker_manager import DockerManager, ContainerStatus, get_os_type, OSType, DOCKER_INSTALL_GUIDES

logger = logging.getLogger(__name__)

# Installation progress tracking
_install_progress: dict = {
    "docker": {"status": "idle", "message": "", "progress": 0},
    "postgres": {"status": "idle", "message": "", "progress": 0},
}

# Global Docker manager instance
_docker_manager: DockerManager | None = None

# Global setup state instance (initialized on first use)
_setup_state: SetupState | None = None
_setup_state_lock = threading.Lock()

# Connection timeout constant
DB_CONNECT_TIMEOUT = 10


def _get_docker_manager() -> DockerManager:
    """Get or create the Docker manager instance."""
    global _docker_manager
    if _docker_manager is None:
        project_dir = Path(__file__).parent.parent.parent.parent
        _docker_manager = DockerManager(project_dir=project_dir)
    return _docker_manager


def _get_setup_state() -> SetupState:
    """Get or create the setup state instance.

    Thread-safe initialization using double-checked locking.
    """
    global _setup_state
    # Quick check without lock for performance
    if _setup_state is not None:
        return _setup_state

    with _setup_state_lock:
        # Double-check after acquiring lock
        if _setup_state is None:
            # Use project root (four levels up from this file)
            project_dir = Path(__file__).parent.parent.parent.parent
            _setup_state = SetupState(project_dir=project_dir)
        return _setup_state


def run_migrations() -> tuple[bool, str]:
    """Run database migrations.

    Returns:
        Tuple of (success, message).
    """
    try:
        from reverie.db.migrate import run_migrations as db_run_migrations

        db_run_migrations()
        return True, "Migrations completed successfully"
    except Exception as e:
        logger.exception("Migration failed")
        return False, str(e)


def register_setup_routes(mcp) -> None:
    """Register Setup API routes on the FastMCP server.

    Args:
        mcp: FastMCP server instance.
    """
    def _cors_headers() -> dict:
        # In setup mode, allow all origins to enable Setup Wizard access
        # After setup, use the configured origin
        allowed_origin = os.environ.get("REVERIE_CORS_ORIGIN", "*")
        return {
            "Access-Control-Allow-Origin": allowed_origin,
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }

    def _json_response(data: dict, status_code: int = 200) -> JSONResponse:
        return JSONResponse(data, status_code=status_code, headers=_cors_headers())

    def _error_response(message: str, status_code: int = 400) -> JSONResponse:
        return _json_response({"error": message, "success": False}, status_code=status_code)

    # --- OPTIONS preflight for all /api/setup/* ---

    @mcp.custom_route("/api/setup/{rest:path}", methods=["OPTIONS"])
    async def setup_options(request: Request):
        """Handle CORS preflight for all setup API routes."""
        from starlette.responses import Response
        return Response(status_code=204, headers=_cors_headers())

    # --- GET /api/setup/status ---

    @mcp.custom_route("/api/setup/status", methods=["GET"])
    async def setup_status(request: Request):
        """Get overall setup status.

        Returns:
            JSON with setup_complete flag, phase statuses, and OS info.
        """
        try:
            from reverie.docker.docker_manager import get_os_info

            state = _get_setup_state()
            result = state.to_dict()
            result["os_info"] = get_os_info()

            # Include .env config in database phase for form pre-fill
            db_config = state.get_database_config()
            backend_config = state.get_backend_config()

            # Merge .env values into database phase config
            if "database" not in result["phases"]:
                result["phases"]["database"] = {"status": "pending", "checks": {}, "config": {}}

            # Always include backend config (None if not yet configured)
            result["phases"]["database"]["config"] = {
                **result["phases"]["database"].get("config", {}),
                "backend": backend_config.get("backend"),  # None if not configured
            }

            # Include DB credentials if configured
            if db_config.get("user"):
                result["phases"]["database"]["config"].update({
                    "host": db_config.get("host", "localhost"),
                    "port": db_config.get("port", 5432),
                    "user": db_config.get("user", ""),
                    "password": db_config.get("password", ""),
                    "database": db_config.get("name", "reverie"),
                })

            # Include SQLite path if backend is sqlite
            if backend_config.get("backend") == "sqlite":
                result["phases"]["database"]["config"]["sqlite_path"] = backend_config.get("sqlite_path")

            # Include API key status (not the actual key, just whether it exists)
            api_key = state.get_api_key()
            if "apikeys" not in result["phases"]:
                result["phases"]["apikeys"] = {"status": "pending", "checks": {}}
            result["phases"]["apikeys"]["has_api_key"] = bool(api_key)

            return _json_response(result)
        except Exception:
            logger.exception("setup_status error")
            return _error_response("Internal server error", 500)

    # --- POST /api/setup/check/postgres ---

    @mcp.custom_route("/api/setup/check/postgres", methods=["POST"])
    async def check_postgres(request: Request):
        """Test PostgreSQL connection.

        Request body:
            host: str - Database host
            port: int - Database port
            user: str - Database user
            password: str - Database password
            name: str - Database name (optional, default: reverie)

        Returns:
            success: bool - Whether connection succeeded
            version: str - PostgreSQL version (if successful)
            error: str - Error message (if failed)
        """
        try:
            body = await request.json()
        except Exception:
            return _error_response("Invalid JSON body")

        state = _get_setup_state()
        db_config = state.get_database_config()

        host = body.get("host", db_config["host"])
        port = body.get("port", db_config["port"])
        user = body.get("user", db_config["user"])
        password = body.get("password", db_config["password"])
        # Support both "database" (frontend) and "name" (legacy) field names
        name = body.get("database", body.get("name", db_config["name"]))

        # Track if we're using Docker defaults
        using_docker_defaults = False

        if not user:
            # No user configured - check if Docker container is running
            # If so, use default Docker credentials
            docker_mgr = _get_docker_manager()
            container_status = docker_mgr.get_status()

            if container_status.status == ContainerStatus.RUNNING:
                # Docker container is running, use default credentials
                host = "localhost"
                port = 5432
                user = "reverie"
                password = "reverie"
                name = "reverie"
                using_docker_defaults = True
                logger.info("Using default Docker PostgreSQL credentials")
            else:
                state.update_phase_check(
                    "prerequisites",
                    "postgres",
                    False,
                    "No database user configured",
                )
                return _json_response({
                    "success": False,
                    "error": "No database user configured. Configure database settings first.",
                })

        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                dbname=name,
                connect_timeout=DB_CONNECT_TIMEOUT,
            )

            # Get PostgreSQL version
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]

            conn.close()

            # Update setup state
            state.update_phase_check(
                "prerequisites",
                "postgres",
                True,
                f"Connected: {version[:50]}..." + (" (Docker)" if using_docker_defaults else ""),
            )

            # If using Docker defaults, save credentials to .env immediately
            # This ensures the Database Phase form can pre-fill with these values
            if using_docker_defaults:
                state.update_phase_check(
                    "prerequisites",
                    "docker_postgres",
                    True,
                    "PostgreSQL container running",
                )
                # Save Docker credentials to .env for form pre-fill
                state.save_database_config(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    name=name,
                )
                # Also set backend to postgres so DatabasePhase skips selection
                state.save_backend_config(backend="postgres")

            return _json_response({
                "success": True,
                "version": version,
            })

        except Exception as e:
            state.update_phase_check(
                "prerequisites",
                "postgres",
                False,
                str(e),
            )
            return _json_response({
                "success": False,
                "error": str(e),
            })

    # --- POST /api/setup/check/pgvector ---

    @mcp.custom_route("/api/setup/check/pgvector", methods=["POST"])
    async def check_pgvector(request: Request):
        """Check if pgvector extension is available.

        Request body:
            Same as /api/setup/check/postgres

        Returns:
            success: bool - Whether pgvector is available
            installed: bool - Whether pgvector is installed
            version: str - pgvector version (if installed)
            error: str - Error message (if failed)
        """
        try:
            body = await request.json()
        except Exception:
            return _error_response("Invalid JSON body")

        state = _get_setup_state()
        db_config = state.get_database_config()

        host = body.get("host", db_config["host"])
        port = body.get("port", db_config["port"])
        user = body.get("user", db_config["user"])
        password = body.get("password", db_config["password"])
        # Support both "database" (frontend) and "name" (legacy) field names
        name = body.get("database", body.get("name", db_config["name"]))

        # If no user configured, check if Docker container is running
        if not user:
            docker_mgr = _get_docker_manager()
            container_status = docker_mgr.get_status()

            if container_status.status == ContainerStatus.RUNNING:
                host = "localhost"
                port = 5432
                user = "reverie"
                password = "reverie"
                name = "reverie"
                logger.info("Using default Docker PostgreSQL credentials for pgvector check")

        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                dbname=name,
                connect_timeout=DB_CONNECT_TIMEOUT,
            )

            with conn.cursor() as cur:
                # Check if pgvector is available
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_available_extensions
                        WHERE name = 'vector'
                    )
                """)
                available = cur.fetchone()[0]

                if not available:
                    conn.close()
                    state.update_phase_check(
                        "prerequisites",
                        "pgvector",
                        False,
                        "pgvector extension not available",
                    )
                    return _json_response({
                        "success": False,
                        "installed": False,
                        "error": "pgvector extension not available. Please install it.",
                    })

                # Check if it's installed
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_extension
                        WHERE extname = 'vector'
                    )
                """)
                installed = cur.fetchone()[0]

                # Get version if installed
                version = None
                if installed:
                    cur.execute("""
                        SELECT extversion FROM pg_extension
                        WHERE extname = 'vector'
                    """)
                    version = cur.fetchone()[0]

            conn.close()

            state.update_phase_check(
                "prerequisites",
                "pgvector",
                True,
                f"pgvector {'v' + version if version else 'available'}",
            )

            return _json_response({
                "success": True,
                "installed": installed,
                "version": version,
            })

        except Exception as e:
            state.update_phase_check(
                "prerequisites",
                "pgvector",
                False,
                str(e),
            )
            return _json_response({
                "success": False,
                "installed": False,
                "error": str(e),
            })

    # --- GET /api/setup/check/docker ---

    @mcp.custom_route("/api/setup/check/docker", methods=["GET"])
    async def check_docker(request: Request):
        """Check if Docker is installed and available.

        Returns:
            success: bool - Whether Docker is available
            installed: bool - Whether Docker is installed
            version: str - Docker version (if installed)
            compose_available: bool - Whether Docker Compose is available
            compose_version: str - Compose version (if available)
            os_type: str - Operating system type (windows, linux, macos)
            install_guide: dict - Installation guide for the OS (if not installed)
            error: str - Error message (if failed)
        """
        try:
            from reverie.docker.docker_manager import get_os_info

            docker_mgr = _get_docker_manager()
            result = docker_mgr.check_docker()
            os_info = get_os_info()

            state = _get_setup_state()
            if result.installed:
                state.update_phase_check(
                    "prerequisites",
                    "docker",
                    True,
                    f"Docker {result.version or 'installed'}",
                )
            else:
                state.update_phase_check(
                    "prerequisites",
                    "docker",
                    False,
                    result.error or "Docker not available",
                )

            return _json_response({
                "success": result.installed and result.daemon_running and result.compose_available,
                "installed": result.installed,
                "daemon_running": result.daemon_running,
                "version": result.version,
                "compose_available": result.compose_available,
                "compose_version": result.compose_version,
                "os_type": result.os_type,
                "os_info": os_info,
                "install_guide": result.install_guide,
                "error": result.error,
            })

        except Exception as e:
            logger.exception("check_docker error")
            return _json_response({
                "success": False,
                "installed": False,
                "error": str(e),
            })

    # --- GET /api/setup/docker/install ---

    @mcp.custom_route("/api/setup/docker/install", methods=["GET"])
    async def install_docker(request: Request):
        """Attempt to install Docker on the system.

        Returns SSE stream with progress updates.

        Returns:
            SSE stream with progress events, final event contains result.
        """
        from starlette.responses import StreamingResponse
        import subprocess
        import shutil
        import asyncio

        async def generate_progress():
            os_type = get_os_type()
            docker_mgr = _get_docker_manager()

            def send_event(status: str, message: str, progress: int):
                import json
                data = json.dumps({"status": status, "message": message, "progress": progress})
                return f"data: {data}\n\n"

            yield send_event("starting", f"Installing Docker on {os_type.value}...", 0)
            await asyncio.sleep(0.1)

            success = False
            error_msg = ""

            try:
                if os_type == OSType.WINDOWS:
                    yield send_event("in_progress", "Checking for winget...", 10)
                    await asyncio.sleep(0.1)

                    if shutil.which("winget"):
                        yield send_event("in_progress", "Running: winget install Docker.DockerDesktop", 20)
                        await asyncio.sleep(0.1)

                        process = await asyncio.create_subprocess_exec(
                            "winget", "install", "Docker.DockerDesktop",
                            "--accept-package-agreements", "--accept-source-agreements",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )

                        yield send_event("in_progress", "Installing Docker Desktop (this may take a few minutes)...", 40)

                        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)

                        if process.returncode == 0:
                            yield send_event("in_progress", "Docker Desktop installed!", 90)
                            success = True
                        else:
                            error_msg = stderr.decode("utf-8", errors="replace")
                    else:
                        # winget not available - send error immediately and return
                        guide = DOCKER_INSTALL_GUIDES.get(os_type, {})
                        guide_url = guide.get("url", "https://docs.docker.com/get-docker/")
                        yield send_event("error", f"winget not available. Please install Docker Desktop manually: {guide_url}", 0)
                        return

                elif os_type == OSType.LINUX:
                    yield send_event("in_progress", "Detecting package manager...", 10)
                    await asyncio.sleep(0.1)

                    if shutil.which("apt-get"):
                        yield send_event("in_progress", "Running: sudo apt-get update", 20)

                        process = await asyncio.create_subprocess_exec(
                            "sudo", "apt-get", "update",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        await asyncio.wait_for(process.communicate(), timeout=120)

                        yield send_event("in_progress", "Running: sudo apt-get install docker.io docker-compose-plugin", 40)

                        process = await asyncio.create_subprocess_exec(
                            "sudo", "apt-get", "install", "-y", "docker.io", "docker-compose-plugin",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )

                        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

                        if process.returncode == 0:
                            yield send_event("in_progress", "Starting Docker service...", 70)

                            await asyncio.create_subprocess_exec("sudo", "systemctl", "start", "docker")
                            await asyncio.create_subprocess_exec("sudo", "systemctl", "enable", "docker")

                            yield send_event("in_progress", "Adding user to docker group...", 80)

                            user = os.environ.get("USER", "")
                            if user:
                                await asyncio.create_subprocess_exec("sudo", "usermod", "-aG", "docker", user)

                            success = True
                        else:
                            error_msg = stderr.decode("utf-8", errors="replace")

                    elif shutil.which("dnf"):
                        yield send_event("in_progress", "Running: sudo dnf install docker docker-compose-plugin", 30)

                        process = await asyncio.create_subprocess_exec(
                            "sudo", "dnf", "install", "-y", "docker", "docker-compose-plugin",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )

                        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

                        if process.returncode == 0:
                            yield send_event("in_progress", "Starting Docker service...", 70)
                            await asyncio.create_subprocess_exec("sudo", "systemctl", "start", "docker")
                            await asyncio.create_subprocess_exec("sudo", "systemctl", "enable", "docker")
                            success = True
                        else:
                            error_msg = stderr.decode("utf-8", errors="replace")
                    else:
                        # No supported package manager - send error immediately and return
                        guide = DOCKER_INSTALL_GUIDES.get(os_type, {})
                        guide_url = guide.get("url", "https://docs.docker.com/get-docker/")
                        yield send_event("error", f"No supported package manager found (apt-get or dnf). Please install Docker manually: {guide_url}", 0)
                        return

                elif os_type == OSType.MACOS:
                    yield send_event("in_progress", "Checking for Homebrew...", 10)
                    await asyncio.sleep(0.1)

                    if shutil.which("brew"):
                        yield send_event("in_progress", "Running: brew install --cask docker", 20)

                        process = await asyncio.create_subprocess_exec(
                            "brew", "install", "--cask", "docker",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )

                        yield send_event("in_progress", "Installing Docker Desktop (this may take a few minutes)...", 40)

                        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)

                        if process.returncode == 0:
                            success = True
                        else:
                            error_msg = stderr.decode("utf-8", errors="replace")
                    else:
                        # Homebrew not available - send error immediately and return
                        guide = DOCKER_INSTALL_GUIDES.get(os_type, {})
                        guide_url = guide.get("url", "https://docs.docker.com/get-docker/")
                        yield send_event("error", f"Homebrew not available. Please install Docker Desktop manually: {guide_url}", 0)
                        return

                else:
                    # Unsupported OS - send error immediately and return
                    yield send_event("error", f"Unsupported OS: {os_type.value}. Please install Docker manually.", 0)
                    return

            except asyncio.TimeoutError:
                error_msg = "Installation timed out"
            except Exception as e:
                error_msg = str(e)

            # Final result
            if success:
                # Verify installation
                result = docker_mgr.check_docker()
                if result.installed:
                    yield send_event("complete", "Docker installed successfully! You may need to restart your terminal.", 100)
                else:
                    yield send_event("complete", "Docker installed but not detected. Please restart your terminal or computer.", 100)
            else:
                # Include installation guide in error
                guide = DOCKER_INSTALL_GUIDES.get(os_type, {})
                guide_url = guide.get("url", "https://docs.docker.com/get-docker/")
                yield send_event("error", f"Installation failed: {error_msg}. Manual install: {guide_url}", 0)

        return StreamingResponse(
            generate_progress(),
            media_type="text/event-stream",
            headers={
                **_cors_headers(),
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # --- POST /api/setup/docker/start ---

    @mcp.custom_route("/api/setup/docker/start", methods=["POST"])
    async def docker_start(request: Request):
        """Start PostgreSQL container with Docker.

        Request body (optional):
            postgres_user: str - PostgreSQL username (default: reverie)
            postgres_password: str - PostgreSQL password (default: reverie)
            postgres_db: str - Database name (default: reverie)
            postgres_port: int - Host port (default: 5432)

        Returns:
            success: bool - Whether container started
            message: str - Status message
        """
        try:
            body = await request.json()
        except Exception:
            body = {}

        postgres_user = body.get("postgres_user", "reverie")
        postgres_password = body.get("postgres_password", "reverie")
        postgres_db = body.get("postgres_db", "reverie")
        postgres_port = body.get("postgres_port", 5432)

        try:
            docker_mgr = _get_docker_manager()
            success, message = docker_mgr.start(
                postgres_user=postgres_user,
                postgres_password=postgres_password,
                postgres_db=postgres_db,
                postgres_port=postgres_port,
                wait_ready=True,
            )

            if success:
                # Update setup state with Docker container info
                state = _get_setup_state()
                state.update_phase_check(
                    "prerequisites",
                    "docker_postgres",
                    True,
                    "PostgreSQL container running",
                )

            return _json_response({
                "success": success,
                "message": message,
                "connection_info": {
                    "host": "localhost",
                    "port": postgres_port,
                    "user": postgres_user,
                    "password": postgres_password,
                    "database": postgres_db,
                } if success else None,
            })

        except Exception as e:
            logger.exception("docker_start error")
            return _json_response({
                "success": False,
                "message": str(e),
            })

    # --- POST /api/setup/postgres/start (SSE with progress) ---

    @mcp.custom_route("/api/setup/postgres/start", methods=["POST"])
    async def postgres_start_with_progress(request: Request):
        """Start PostgreSQL container with progress streaming.

        Request body (optional):
            postgres_user: str - PostgreSQL username (default: reverie)
            postgres_password: str - PostgreSQL password (default: reverie)
            postgres_db: str - Database name (default: reverie)
            postgres_port: int - Host port (default: 5432)

        Returns:
            SSE stream with progress events.
        """
        from starlette.responses import StreamingResponse
        import asyncio
        import json as json_module

        try:
            body = await request.json()
        except Exception:
            body = {}

        postgres_user = body.get("postgres_user", "reverie")
        postgres_password = body.get("postgres_password", "reverie")
        postgres_db = body.get("postgres_db", "reverie")
        postgres_port = body.get("postgres_port", 5432)

        async def generate_progress():
            def send_event(status: str, message: str, progress: int, data: dict = None):
                event_data = {"status": status, "message": message, "progress": progress}
                if data:
                    event_data.update(data)
                return f"data: {json_module.dumps(event_data)}\n\n"

            docker_mgr = _get_docker_manager()

            # Check Docker first
            yield send_event("in_progress", "Checking Docker availability...", 5)
            await asyncio.sleep(0.1)

            docker_result = docker_mgr.check_docker()
            if not docker_result.installed:
                yield send_event("error", "Docker is not installed", 0, {
                    "install_guide": docker_result.install_guide
                })
                return

            yield send_event("in_progress", "Docker is available", 10)
            await asyncio.sleep(0.1)

            # Check if container already exists
            yield send_event("in_progress", "Checking for existing container...", 15)
            await asyncio.sleep(0.1)

            status_result = docker_mgr.get_status()
            if status_result.status == ContainerStatus.RUNNING:
                yield send_event("complete", "PostgreSQL container is already running", 100, {
                    "connection_info": {
                        "host": "localhost",
                        "port": postgres_port,
                        "user": postgres_user,
                        "password": postgres_password,
                        "database": postgres_db,
                    }
                })
                return

            # Pull image if needed
            yield send_event("in_progress", "Pulling PostgreSQL image (pgvector/pgvector:pg16)...", 20)
            await asyncio.sleep(0.1)

            # Start container
            yield send_event("in_progress", "Creating and starting PostgreSQL container...", 40)

            try:
                # Run in thread to avoid blocking
                import concurrent.futures
                loop = asyncio.get_event_loop()

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    success, message = await loop.run_in_executor(
                        pool,
                        lambda: docker_mgr.start(
                            postgres_user=postgres_user,
                            postgres_password=postgres_password,
                            postgres_db=postgres_db,
                            postgres_port=postgres_port,
                            wait_ready=False,  # We'll wait ourselves with progress
                        )
                    )

                if not success:
                    yield send_event("error", f"Failed to start container: {message}", 0)
                    return

                yield send_event("in_progress", "Container started, waiting for PostgreSQL to be ready...", 60)

                # Wait for PostgreSQL to be ready with progress updates
                max_wait = 30
                for i in range(max_wait):
                    await asyncio.sleep(1)
                    progress = 60 + int((i / max_wait) * 35)
                    yield send_event("in_progress", f"Waiting for PostgreSQL... ({i+1}/{max_wait}s)", progress)

                    # Check if ready
                    try:
                        import psycopg2
                        conn = psycopg2.connect(
                            host="localhost",
                            port=postgres_port,
                            user=postgres_user,
                            password=postgres_password,
                            dbname=postgres_db,
                            connect_timeout=2,
                        )
                        conn.close()

                        # Update setup state and save connection info to .env
                        state = _get_setup_state()
                        state.update_phase_check(
                            "prerequisites",
                            "docker_postgres",
                            True,
                            "PostgreSQL container running",
                        )
                        # Save Docker PostgreSQL connection info to .env
                        state.save_database_config(
                            host="localhost",
                            port=postgres_port,
                            user=postgres_user,
                            password=postgres_password,
                            name=postgres_db,
                        )

                        yield send_event("complete", "PostgreSQL is ready!", 100, {
                            "connection_info": {
                                "host": "localhost",
                                "port": postgres_port,
                                "user": postgres_user,
                                "password": postgres_password,
                                "database": postgres_db,
                            }
                        })
                        return
                    except Exception:
                        pass  # Not ready yet

                yield send_event("error", "PostgreSQL did not become ready in time", 0)

            except Exception as e:
                yield send_event("error", f"Error starting PostgreSQL: {str(e)}", 0)

        return StreamingResponse(
            generate_progress(),
            media_type="text/event-stream",
            headers={
                **_cors_headers(),
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # --- POST /api/setup/docker/stop ---

    @mcp.custom_route("/api/setup/docker/stop", methods=["POST"])
    async def docker_stop(request: Request):
        """Stop PostgreSQL container.

        Returns:
            success: bool - Whether container stopped
            message: str - Status message
        """
        try:
            docker_mgr = _get_docker_manager()
            success, message = docker_mgr.stop()

            if success:
                state = _get_setup_state()
                state.update_phase_check(
                    "prerequisites",
                    "docker_postgres",
                    False,
                    "Container stopped",
                )

            return _json_response({
                "success": success,
                "message": message,
            })

        except Exception as e:
            logger.exception("docker_stop error")
            return _json_response({
                "success": False,
                "message": str(e),
            })

    # --- GET /api/setup/docker/status ---

    @mcp.custom_route("/api/setup/docker/status", methods=["GET"])
    async def docker_status(request: Request):
        """Get PostgreSQL container status.

        Returns:
            success: bool - Whether check succeeded
            status: str - Container status (running, stopped, not_found, error)
            container_id: str - Container ID (if exists)
            health: str - Container health status
            error: str - Error message (if failed)
        """
        try:
            docker_mgr = _get_docker_manager()
            result = docker_mgr.get_status()

            return _json_response({
                "success": result.status != ContainerStatus.ERROR,
                "status": result.status.value,
                "container_id": result.container_id,
                "health": result.health,
                "error": result.error,
            })

        except Exception as e:
            logger.exception("docker_status error")
            return _json_response({
                "success": False,
                "status": "error",
                "error": str(e),
            })

    # --- POST /api/setup/database ---

    @mcp.custom_route("/api/setup/database", methods=["POST"])
    async def save_database(request: Request):
        """Save database configuration.

        Request body:
            host: str - Database host
            port: int - Database port
            user: str - Database user
            password: str - Database password
            name: str - Database name

        Returns:
            success: bool - Whether save succeeded
        """
        try:
            body = await request.json()
        except Exception:
            return _error_response("Invalid JSON body")

        # Validate required fields
        # Support both "database" (frontend) and "name" (legacy) for database name
        required = ["host", "port", "user", "password"]
        for field in required:
            if field not in body:
                return _error_response(f"'{field}' is required")

        # Check for database name (either "database" or "name")
        if "database" not in body and "name" not in body:
            return _error_response("'database' is required")

        # Validate port range
        try:
            port = int(body["port"])
            if not (1 <= port <= 65535):
                return _error_response("port must be between 1 and 65535")
        except (ValueError, TypeError):
            return _error_response("port must be a valid integer")

        try:
            state = _get_setup_state()
            # Support both "database" (frontend) and "name" (legacy)
            db_name = body.get("database", body.get("name"))
            state.save_database_config(
                host=body["host"],
                port=port,
                user=body["user"],
                password=body["password"],
                name=db_name,
            )
            state.update_phase("database", PhaseStatus.COMPLETE)

            return _json_response({"success": True})

        except Exception as e:
            logger.exception("save_database error")
            return _json_response({
                "success": False,
                "error": str(e),
            })

    # --- POST /api/setup/apikey ---

    @mcp.custom_route("/api/setup/apikey", methods=["POST"])
    async def save_apikey(request: Request):
        """Save and validate OpenAI API key with model settings.

        Request body:
            openai_api_key: str - OpenAI API key
            model: str - Embedding model (optional)
            indexing_ner_model: str - Indexing NER model (optional)
            query_ner_model: str - Query NER model (optional)
            preprocessor_model: str - Preprocessor model (optional)
            enrich_stage1_model: str - Enrich stage 1 model (optional)
            enrich_stage2_model: str - Enrich stage 2 model (optional)

        Returns:
            success: bool - Whether key is valid
            error: str - Error message (if invalid)
        """
        try:
            body = await request.json()
        except Exception:
            return _error_response("Invalid JSON body")

        api_key = body.get("openai_api_key", "").strip()

        state = _get_setup_state()

        # If empty key, check if we have an existing key (for model-only updates)
        if not api_key:
            existing_key = state.get_api_key()
            if existing_key:
                # Keep existing key, just update models
                api_key = existing_key
            else:
                return _error_response("'openai_api_key' is required")

        # Extract model settings
        embedding_model = body.get("model")
        indexing_ner_model = body.get("indexing_ner_model")
        query_ner_model = body.get("query_ner_model")
        preprocessor_model = body.get("preprocessor_model")
        enrich_stage1_model = body.get("enrich_stage1_model")
        enrich_stage2_model = body.get("enrich_stage2_model")

        # Validate API key by making a test request
        if openai is None:
            # OpenAI not installed, just save the key and models
            state.save_api_key(
                api_key,
                embedding_model=embedding_model,
                indexing_ner_model=indexing_ner_model,
                query_ner_model=query_ner_model,
                preprocessor_model=preprocessor_model,
                enrich_stage1_model=enrich_stage1_model,
                enrich_stage2_model=enrich_stage2_model,
            )
            state.update_phase("apikeys", PhaseStatus.COMPLETE)
            return _json_response({
                "success": True,
                "warning": "openai package not installed, key not validated",
            })

        try:
            client = openai.OpenAI(api_key=api_key)
            # Test by listing models (minimal API call)
            client.models.list()

            state.save_api_key(
                api_key,
                embedding_model=embedding_model,
                indexing_ner_model=indexing_ner_model,
                query_ner_model=query_ner_model,
                preprocessor_model=preprocessor_model,
                enrich_stage1_model=enrich_stage1_model,
                enrich_stage2_model=enrich_stage2_model,
            )
            state.update_phase("apikeys", PhaseStatus.COMPLETE)

            return _json_response({"success": True})

        except openai.AuthenticationError:
            state.update_phase_check("apikeys", "openai", False, "Invalid API key")
            return _json_response({
                "success": False,
                "error": "Invalid API key",
            })
        except Exception as e:
            state.update_phase_check("apikeys", "openai", False, str(e))
            return _json_response({
                "success": False,
                "error": str(e),
            })

    # --- POST /api/setup/profile ---

    @mcp.custom_route("/api/setup/profile", methods=["POST"])
    async def save_profile(request: Request):
        """Save profile configuration.

        Request body:
            profile_name: str - Profile name
            sessions_paths: list[str] - Paths to session files (multi-folder)
            sessions_path: str - Single path (deprecated, for backwards compatibility)

        Returns:
            success: bool - Whether save succeeded
        """
        try:
            body = await request.json()
        except Exception:
            return _error_response("Invalid JSON body")

        profile_name = body.get("profile_name", "default").strip()

        # Support both sessions_paths (list) and sessions_path (string, legacy)
        sessions_paths: list[str] = []
        if "sessions_paths" in body:
            raw_paths = body.get("sessions_paths", [])
            if isinstance(raw_paths, list):
                sessions_paths = [p.strip() for p in raw_paths if isinstance(p, str)]
        elif "sessions_path" in body:
            # Legacy single path - convert to list
            single_path = body.get("sessions_path", "").strip()
            if single_path:
                sessions_paths = [single_path]

        # Validate all paths are absolute
        # Must check BEFORE expanduser/resolve as those make relative paths absolute
        for path in sessions_paths:
            if path:
                # Unix absolute: starts with / or ~
                # Windows absolute: starts with drive letter (e.g., C:/ or D:\)
                is_unix_absolute = path.startswith("/") or path.startswith("~")
                is_windows_absolute = len(path) >= 3 and path[1] == ":" and path[2] in "/\\"
                if not is_unix_absolute and not is_windows_absolute:
                    return _json_response({
                        "success": False,
                        "error": f"Path must be absolute: {path}",
                    })
                # Note: We don't require paths to exist; they will be created later

        try:
            state = _get_setup_state()
            state.save_profile_config(
                profile_name=profile_name,
                sessions_paths=sessions_paths,
            )
            state.update_phase("profile", PhaseStatus.COMPLETE)

            return _json_response({"success": True})

        except Exception as e:
            logger.exception("save_profile error")
            return _json_response({
                "success": False,
                "error": str(e),
            })

    # --- POST /api/setup/migrate ---

    @mcp.custom_route("/api/setup/migrate", methods=["POST"])
    async def run_migrate(request: Request):
        """Run database migrations.

        Returns:
            success: bool - Whether migration succeeded
            message: str - Status message
        """
        state = _get_setup_state()
        state.update_phase("migration", PhaseStatus.IN_PROGRESS)

        try:
            success, message = run_migrations()

            if success:
                state.update_phase("migration", PhaseStatus.COMPLETE)
            else:
                state.update_phase("migration", PhaseStatus.FAILED)

            return _json_response({
                "success": success,
                "message": message,
            })

        except Exception as e:
            logger.exception("run_migrate error")
            state.update_phase("migration", PhaseStatus.FAILED)
            return _json_response({
                "success": False,
                "message": str(e),
            })

    # --- GET /api/setup/verify ---

    @mcp.custom_route("/api/setup/verify", methods=["GET"])
    async def verify_setup(request: Request):
        """Run verification checks.

        Returns:
            success: bool - Whether all checks passed
            checks: dict - Individual check results
        """
        state = _get_setup_state()
        checks = {}
        all_passed = True

        # Check database connection
        db_config = state.get_database_config()
        try:
            conn = psycopg2.connect(
                host=db_config["host"],
                port=db_config["port"],
                user=db_config["user"],
                password=db_config["password"],
                dbname=db_config["name"],
                connect_timeout=DB_CONNECT_TIMEOUT,
            )

            # Check tables exist
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'passages'
                    )
                """)
                tables_exist = cur.fetchone()[0]

            conn.close()

            checks["database"] = {
                "success": True,
                "message": "Database connected",
            }
            checks["tables"] = {
                "success": tables_exist,
                "message": "Tables exist" if tables_exist else "Tables not found",
            }
            if not tables_exist:
                all_passed = False

        except Exception as e:
            checks["database"] = {
                "success": False,
                "message": str(e),
            }
            all_passed = False

        # Check API key
        api_key = state.get_api_key()
        if api_key:
            checks["api_key"] = {
                "success": True,
                "message": "API key configured",
            }
        else:
            checks["api_key"] = {
                "success": False,
                "message": "API key not configured",
            }
            all_passed = False

        if all_passed:
            state.update_phase("verification", PhaseStatus.COMPLETE)
        else:
            state.update_phase("verification", PhaseStatus.FAILED)

        return _json_response({
            "success": all_passed,
            "checks": checks,
        })

    # --- POST /api/setup/complete ---

    @mcp.custom_route("/api/setup/complete", methods=["POST"])
    async def mark_complete(request: Request):
        """Mark setup as complete and start IndexScheduler.

        Returns:
            success: bool - Whether marking succeeded
            indexer_started: bool - Whether indexer was started
            message: str - Status message
        """
        try:
            state = _get_setup_state()
            state.mark_complete()

            # Try to start the IndexScheduler now that setup is complete
            indexer_started = False
            indexer_message = ""

            try:
                from reverie.mcp.server import _start_scheduler
                import asyncio

                # Start scheduler in background
                asyncio.create_task(_start_scheduler())
                indexer_started = True
                indexer_message = "IndexScheduler starting in background"
            except Exception as e:
                indexer_message = f"IndexScheduler not started: {e}"
                logger.warning(indexer_message)

            # Clear settings module cache to reload with setup_complete=True
            try:
                import importlib
                import reverie.config.settings
                importlib.reload(reverie.config.settings)
            except Exception:
                pass  # Non-critical

            return _json_response({
                "success": True,
                "indexer_started": indexer_started,
                "message": indexer_message or "Setup complete",
            })

        except Exception as e:
            logger.exception("mark_complete error")
            return _json_response({
                "success": False,
                "error": str(e),
            })

    # --- POST /api/setup/reset (TEST ONLY) ---

    @mcp.custom_route("/api/setup/reset", methods=["POST"])
    async def reset_setup(request: Request):
        """Reset setup state for testing.

        WARNING: This endpoint is for testing only.
        It resets the setup state to initial values and clears .env file.

        Returns:
            success: bool - Whether reset succeeded
        """
        global _setup_state

        try:
            with _setup_state_lock:
                state = _get_setup_state()
                # Reset to default config
                state.config = state.config.default()
                state._save_config()

                # Delete .env file to reset all configuration
                if state.env_path.exists():
                    state.env_path.unlink()
                    logger.info(f"Deleted .env file: {state.env_path}")

                # Force reload on next request
                _setup_state = None

            return _json_response({"success": True, "message": "Setup reset complete"})

        except Exception as e:
            logger.exception("reset_setup error")
            return _json_response({
                "success": False,
                "error": str(e),
            })

    # --- GET /api/setup/indexer/status ---

    @mcp.custom_route("/api/setup/indexer/status", methods=["GET"])
    async def indexer_status(request: Request):
        """Get IndexScheduler status and readiness.

        Returns:
            success: bool - Whether check succeeded
            can_start: bool - Whether scheduler can be started
            reason: str - Reason if cannot start
            running: bool - Whether scheduler is currently running
            stats: dict - Scheduler statistics (if running)
        """
        try:
            from reverie.mcp.server import _can_start_scheduler, _scheduler

            can_start, reason = _can_start_scheduler()

            response = {
                "success": True,
                "can_start": can_start,
                "reason": reason,
                "running": False,
                "stats": None,
            }

            if _scheduler is not None:
                response["running"] = _scheduler.is_running
                response["stats"] = _scheduler.stats

            return _json_response(response)

        except Exception as e:
            logger.exception("indexer_status error")
            return _json_response({
                "success": False,
                "error": str(e),
            })

    # --- POST /api/setup/indexer/start ---

    @mcp.custom_route("/api/setup/indexer/start", methods=["POST"])
    async def start_indexer(request: Request):
        """Manually start the IndexScheduler.

        Returns:
            success: bool - Whether scheduler was started
            message: str - Status message
        """
        try:
            from reverie.mcp.server import _can_start_scheduler, _start_scheduler

            can_start, reason = _can_start_scheduler()
            if not can_start:
                return _json_response({
                    "success": False,
                    "message": f"Cannot start indexer: {reason}",
                })

            import asyncio
            asyncio.create_task(_start_scheduler())

            return _json_response({
                "success": True,
                "message": "IndexScheduler starting",
            })

        except Exception as e:
            logger.exception("start_indexer error")
            return _json_response({
                "success": False,
                "message": str(e),
            })

    # --- GET /api/setup/check/sqlite ---

    @mcp.custom_route("/api/setup/check/sqlite", methods=["GET"])
    async def check_sqlite(request: Request):
        """Check if sqlite-vec extension is available.

        Returns:
            success: bool - Whether check succeeded
            available: bool - Whether sqlite-vec is available
            version: str - sqlite-vec version (if available)
            error: str - Error message (if failed)
        """
        try:
            import sqlite3
            import sqlite_vec

            # Create temporary in-memory database to test sqlite-vec
            conn = sqlite3.connect(":memory:")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)

            # Get version
            cursor = conn.execute("SELECT vec_version()")
            version = cursor.fetchone()[0]
            conn.close()

            state = _get_setup_state()
            state.update_phase_check(
                "prerequisites",
                "sqlite_vec",
                True,
                f"sqlite-vec v{version}",
            )

            return _json_response({
                "success": True,
                "available": True,
                "version": version,
            })

        except ImportError as e:
            state = _get_setup_state()
            state.update_phase_check(
                "prerequisites",
                "sqlite_vec",
                False,
                "sqlite-vec not installed",
            )
            return _json_response({
                "success": True,
                "available": False,
                "version": None,
                "error": "sqlite-vec package not installed. Install with: pip install sqlite-vec",
            })

        except Exception as e:
            state = _get_setup_state()
            state.update_phase_check(
                "prerequisites",
                "sqlite_vec",
                False,
                str(e),
            )
            return _json_response({
                "success": False,
                "available": False,
                "version": None,
                "error": str(e),
            })

    # --- POST /api/setup/sqlite/install ---

    @mcp.custom_route("/api/setup/sqlite/install", methods=["POST"])
    async def install_sqlite_vec(request: Request):
        """Install sqlite-vec package using pip.

        Returns:
            success: bool - Whether installation succeeded
            message: str - Success or error message
        """
        import subprocess
        import sys

        try:
            # Run pip install sqlite-vec
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "sqlite-vec"],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                # Verify installation
                try:
                    import sqlite_vec
                    version = getattr(sqlite_vec, "__version__", "unknown")
                    return _json_response({
                        "success": True,
                        "message": f"sqlite-vec v{version} installed successfully",
                        "version": version,
                    })
                except ImportError:
                    return _json_response({
                        "success": False,
                        "message": "Installation completed but import failed. Try restarting the server.",
                    })
            else:
                return _json_response({
                    "success": False,
                    "message": f"pip install failed: {result.stderr or result.stdout}",
                })

        except subprocess.TimeoutExpired:
            return _json_response({
                "success": False,
                "message": "Installation timed out after 120 seconds",
            })
        except Exception as e:
            return _json_response({
                "success": False,
                "message": str(e),
            })

    # --- POST /api/setup/backend/sqlite ---

    @mcp.custom_route("/api/setup/backend/sqlite", methods=["POST"])
    async def select_sqlite_backend(request: Request):
        """Select SQLite as the database backend.

        Request body (optional):
            sqlite_path: str - Path to SQLite database file (default: ~/.reverie/reverie.db)

        Returns:
            success: bool - Whether selection succeeded
            backend: str - Selected backend ("sqlite")
            sqlite_path: str - Path to SQLite database
        """
        try:
            body = await request.json()
        except Exception:
            body = {}

        sqlite_path = body.get("sqlite_path")

        try:
            state = _get_setup_state()
            state.save_backend_config(
                backend="sqlite",
                sqlite_path=sqlite_path,
            )

            config = state.get_backend_config()

            return _json_response({
                "success": True,
                "backend": "sqlite",
                "sqlite_path": config["sqlite_path"],
            })

        except Exception as e:
            logger.exception("select_sqlite_backend error")
            return _json_response({
                "success": False,
                "error": str(e),
            })

    # --- POST /api/setup/backend/postgres ---

    @mcp.custom_route("/api/setup/backend/postgres", methods=["POST"])
    async def select_postgres_backend(request: Request):
        """Select PostgreSQL as the database backend.

        Returns:
            success: bool - Whether selection succeeded
            backend: str - Selected backend ("postgres")
        """
        try:
            state = _get_setup_state()
            state.save_backend_config(backend="postgres")

            return _json_response({
                "success": True,
                "backend": "postgres",
            })

        except Exception as e:
            logger.exception("select_postgres_backend error")
            return _json_response({
                "success": False,
                "error": str(e),
            })

    # --- GET /api/setup/claude-projects ---

    @mcp.custom_route("/api/setup/claude-projects", methods=["GET"])
    async def list_claude_projects(request: Request):
        """List Claude Code projects for session path selection.

        Detects Claude Code config folder and returns available projects.

        Returns:
            success: bool - Whether Claude Code is installed
            config_path: str - Path to Claude config folder
            projects: list[dict] - List of projects with path and display name
            error: str - Error message (if Claude not found)
        """
        try:
            home = Path.home()

            # Claude Code config locations by OS
            # Linux/macOS: ~/.claude/
            # Windows: %USERPROFILE%\.claude\ or %APPDATA%\Claude\
            possible_paths = [
                home / ".claude",
                home / "AppData" / "Roaming" / "Claude",  # Windows
            ]

            config_path = None
            for path in possible_paths:
                if path.exists() and (path / "projects").exists():
                    config_path = path
                    break

            if not config_path:
                return _json_response({
                    "success": False,
                    "error": "Claude Code not found. Install Claude Code first.",
                    "config_path": None,
                    "projects": [],
                })

            projects_dir = config_path / "projects"
            projects = []

            for project_folder in sorted(projects_dir.iterdir()):
                if project_folder.is_dir():
                    # Convert folder name back to path
                    # Linux/macOS: "-home-eias-my-project" -> "/home/eias/my-project"
                    # Windows: "-D--Development-eb-renpy" -> "D:/Development/eb/renpy"
                    folder_name = project_folder.name
                    # Replace dashes with slashes
                    original_path = folder_name.replace("-", "/")

                    # Handle Windows drive letter (e.g., "/D//..." -> "D:/...")
                    import re
                    windows_drive_match = re.match(r"^/([A-Za-z])//(.*)$", original_path)
                    if windows_drive_match:
                        drive = windows_drive_match.group(1).upper()
                        rest = windows_drive_match.group(2)
                        original_path = f"{drive}:/{rest}"
                    elif not original_path.startswith("/"):
                        original_path = "/" + original_path

                    # Get display name (last component of path)
                    display_name = original_path.rsplit("/", 1)[-1] or original_path

                    # Check if there are any session files
                    session_files = list(project_folder.glob("*.jsonl"))
                    session_count = len(session_files)

                    projects.append({
                        "folder_name": folder_name,
                        "original_path": original_path,
                        "display_name": display_name,
                        "session_path": str(project_folder),
                        "session_count": session_count,
                    })

            return _json_response({
                "success": True,
                "config_path": str(config_path),
                "projects": projects,
            })

        except Exception as e:
            logger.exception("list_claude_projects error")
            return _json_response({
                "success": False,
                "error": str(e),
                "config_path": None,
                "projects": [],
            })

    logger.info("Setup API routes registered")
