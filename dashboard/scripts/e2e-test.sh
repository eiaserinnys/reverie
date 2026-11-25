#!/bin/bash
# E2E Test Script with Docker Isolation
#
# CRITICAL ISOLATION RULES:
# - NEVER touch production DB (reverie) or port 8010
# - Uses Docker container for PostgreSQL on port 5433
# - Runs backend on port 8011
# - All test data stays in Docker container (deleted after tests)
#
# Test Environment:
# - PostgreSQL: localhost:5434 / reverie_test (Docker)
# - Backend: localhost:8011
# - Dashboard: localhost:5173 (Vite dev server)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DASHBOARD_DIR="$PROJECT_ROOT/dashboard"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.test.yml"

# Test environment config - ISOLATED from production
# Port 5434 to avoid conflict with production (5432) and other local test databases (5433)
TEST_DB_PORT=5434
TEST_BACKEND_PORT=8011

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[e2e]${NC} $1"; }
warn() { echo -e "${YELLOW}[e2e]${NC} $1"; }
error() { echo -e "${RED}[e2e]${NC} $1"; }
info() { echo -e "${BLUE}[e2e]${NC} $1"; }

# Cleanup function - runs on exit no matter what
cleanup() {
    local exit_code=$?
    echo ""
    log "=== Cleanup ==="

    # Stop backend if running
    if [ -n "$BACKEND_PID" ]; then
        log "Stopping backend (PID: $BACKEND_PID)"
        kill $BACKEND_PID 2>/dev/null || true
        wait $BACKEND_PID 2>/dev/null || true
    fi

    # Restore original .env if backed up
    if [ -f "$PROJECT_ROOT/.env.e2e-backup" ]; then
        mv "$PROJECT_ROOT/.env.e2e-backup" "$PROJECT_ROOT/.env"
        log "Restored original .env"
    fi

    # Stop and remove Docker containers
    if [ -f "$COMPOSE_FILE" ]; then
        log "Stopping Docker containers..."
        docker compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true
    fi

    if [ $exit_code -eq 0 ]; then
        log "Cleanup complete. Tests passed!"
    else
        warn "Cleanup complete. Tests failed with exit code $exit_code"
    fi

    exit $exit_code
}

# Set trap for cleanup on EXIT, INT, TERM
trap cleanup EXIT INT TERM

echo ""
info "╔════════════════════════════════════════════════════════════════╗"
info "║           Reverie E2E Test - Isolated Environment              ║"
info "╠════════════════════════════════════════════════════════════════╣"
info "║  PostgreSQL: localhost:${TEST_DB_PORT} (Docker)                         ║"
info "║  Backend:    localhost:${TEST_BACKEND_PORT}                                 ║"
info "║  Dashboard:  localhost:5173                                    ║"
info "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check Docker is available
if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    error "Docker daemon is not running. Please start Docker."
    exit 1
fi

# Verify we're not accidentally using production ports
if curl -s "http://localhost:8010/api/setup/status" > /dev/null 2>&1; then
    warn "Production backend detected on port 8010. E2E tests will use port ${TEST_BACKEND_PORT}."
fi

# Step 1: Start Docker PostgreSQL container
log "=== Step 1: Starting Docker PostgreSQL container ==="
docker compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true
docker compose -f "$COMPOSE_FILE" up -d

# Wait for PostgreSQL to be ready
log "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if docker compose -f "$COMPOSE_FILE" exec -T postgres-test pg_isready -U reverie_test -d reverie_test > /dev/null 2>&1; then
        log "PostgreSQL is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        error "PostgreSQL failed to start"
        docker compose -f "$COMPOSE_FILE" logs
        exit 1
    fi
    sleep 1
done

# Step 2: Backup .env and create test environment
log "=== Step 2: Setting up test environment ==="
if [ -f "$PROJECT_ROOT/.env" ]; then
    cp "$PROJECT_ROOT/.env" "$PROJECT_ROOT/.env.e2e-backup"
    log "Backed up original .env"
fi

# Create test .env pointing to Docker PostgreSQL
cat > "$PROJECT_ROOT/.env" << EOF
# E2E Test Environment - Auto-generated
# DO NOT COMMIT THIS FILE

# Database - Docker container on port ${TEST_DB_PORT}
REVERIE_DB_HOST=localhost
REVERIE_DB_PORT=${TEST_DB_PORT}
REVERIE_DB_USER=reverie_test
REVERIE_DB_PASSWORD=reverie_test
REVERIE_DB_NAME=reverie_test

# API Key for embeddings (use test key if not provided)
REVERIE_OPENAI_API_KEY=${OPENAI_API_KEY:-sk-test-key-for-e2e}

# Session directory for tests
REVERIE_SESSION_DIR=/tmp/reverie-test-sessions

# Required model settings for pydantic validation
REVERIE_OPENIE_MODEL=gpt-4o-mini
REVERIE_RECOGNITION_MODEL=gpt-4o-mini
REVERIE_PREPROCESSOR_MODEL=gpt-4o-mini

# CORS origin for API
REVERIE_CORS_ORIGIN=*
EOF

log "Created test .env (DB port: ${TEST_DB_PORT})"

# Step 3: Start backend test server
log "=== Step 3: Starting backend test server ==="
cd "$PROJECT_ROOT"

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    log "Creating venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -q -r requirements.txt
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Start backend on test port
log "Starting backend on port ${TEST_BACKEND_PORT}..."
python -m reverie.mcp.server --sse --port $TEST_BACKEND_PORT &
BACKEND_PID=$!
log "Backend PID: $BACKEND_PID"

# Wait for backend to be ready
log "Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -s "http://localhost:$TEST_BACKEND_PORT/api/setup/status" > /dev/null 2>&1; then
        log "Backend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        error "Backend failed to start"
        exit 1
    fi
    sleep 1
done

# Show initial setup status
log "=== Initial Setup Status ==="
curl -s "http://localhost:$TEST_BACKEND_PORT/api/setup/status" | python3 -m json.tool 2>/dev/null || \
    curl -s "http://localhost:$TEST_BACKEND_PORT/api/setup/status"
echo ""

# Step 4: Run Playwright tests
log "=== Step 4: Running Playwright E2E tests ==="
cd "$DASHBOARD_DIR"

# Use nvm if available
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 20 2>/dev/null || true

# Set environment variables for tests
export BACKEND_URL="http://localhost:$TEST_BACKEND_PORT"
export VITE_API_PORT="$TEST_BACKEND_PORT"

log "Running tests with:"
log "  BACKEND_URL=$BACKEND_URL"
log "  VITE_API_PORT=$VITE_API_PORT"
echo ""

# Run Playwright tests - setup first, then dashboard
npx playwright test --project=setup --project=dashboard

log "=== All E2E tests completed successfully! ==="
