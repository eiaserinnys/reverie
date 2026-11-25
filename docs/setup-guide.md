# Reverie Setup Guide

This guide walks you through setting up Reverie from scratch, including PostgreSQL installation via Docker.

## Overview

Reverie is a long-term memory system for AI coding agents. It requires:
- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- OpenAI API key (for embeddings)

## Quick Start (Docker - Recommended)

The easiest way to get started is using Docker for PostgreSQL. The Setup Wizard handles everything automatically.

### Step 1: Install Docker

**Linux (Ubuntu/Debian)**:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect
```

**macOS**:
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Start Docker Desktop from Applications

**Windows**:
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Enable WSL 2 backend during installation

### Step 2: Clone and Install

```bash
git clone https://github.com/eiaserinnys/reverie.git
cd reverie

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Step 3: Start Reverie

```bash
python -m reverie.mcp.server
```

Open http://localhost:8010/reverie/ in your browser.

### Step 4: Complete Setup Wizard

The Setup Wizard will guide you through:

1. **Prerequisites Check**
   - Verifies Python version and Docker availability
   - If Docker is available, offers to set up PostgreSQL automatically

2. **Database Configuration**
   - Choose "Install with Docker" to auto-configure
   - Or enter your existing PostgreSQL credentials
   - The wizard tests the connection before continuing

3. **API Keys**
   - Enter your OpenAI API key
   - Get one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

4. **Profile Setup**
   - Set your profile name (e.g., your username)
   - Configure the sessions directory (where Claude Code sessions are stored)

5. **Database Migration**
   - Click "Run Migration" to create required tables
   - This only runs once; subsequent starts skip this step

6. **Verification**
   - Final check that everything is configured correctly
   - Once complete, you'll see the main dashboard

## Manual Setup (Without Docker)

If you prefer to manage PostgreSQL yourself:

### Install PostgreSQL with pgvector

**Ubuntu/Debian**:
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql-15 postgresql-15-pgvector

# Start PostgreSQL
sudo systemctl start postgresql

# Create database and user
sudo -u postgres psql <<EOF
CREATE USER reverie WITH PASSWORD 'your-password';
CREATE DATABASE reverie OWNER reverie;
\c reverie
CREATE EXTENSION vector;
EOF
```

**macOS (Homebrew)**:
```bash
brew install postgresql@15 pgvector
brew services start postgresql@15

# Create database
createdb reverie
psql reverie -c "CREATE EXTENSION vector"
```

**Docker (pgvector image)**:
```bash
docker run -d \
  --name reverie-postgres \
  -e POSTGRES_USER=reverie \
  -e POSTGRES_PASSWORD=reverie \
  -e POSTGRES_DB=reverie \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### Configure Environment

Create a `.env` file:
```bash
# Database
REVERIE_DB_HOST=localhost
REVERIE_DB_PORT=5432
REVERIE_DB_USER=reverie
REVERIE_DB_PASSWORD=your-password
REVERIE_DB_NAME=reverie

# OpenAI API
OPENAI_API_KEY=sk-your-api-key

# Server
REVERIE_API_PORT=8010
```

### Run Migrations

```bash
python -m reverie.cli migrate
```

### Start Server

```bash
python -m reverie.mcp.server
```

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REVERIE_DB_HOST` | PostgreSQL host | `localhost` |
| `REVERIE_DB_PORT` | PostgreSQL port | `5432` |
| `REVERIE_DB_NAME` | Database name | `reverie` |
| `REVERIE_DB_USER` | Database user | (required) |
| `REVERIE_DB_PASSWORD` | Database password | (required) |
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `REVERIE_API_PORT` | Server port | `8010` |
| `REVERIE_CORS_ORIGIN` | Allowed CORS origins | `*` |
| `REVERIE_SESSION_DIR` | Claude sessions directory | (optional) |

### Files Created

- `.env` - Environment configuration
- `.reverie/setup.json` - Setup wizard state
- `.reverie/cache/` - Embedding cache

## Using with Claude Code

Add Reverie to your Claude Code configuration:

**~/.claude/mcp.json**:
```json
{
  "mcpServers": {
    "reverie": {
      "command": "python",
      "args": ["-m", "reverie.mcp.server"],
      "cwd": "/path/to/reverie",
      "env": {
        "REVERIE_API_URL": "http://localhost:8010"
      }
    }
  }
}
```

## Indexing Sessions

Index your Claude Code session files:

```bash
# Index a specific session
python -m reverie.indexer.cli /path/to/session.jsonl

# Index all sessions in a directory
python -m reverie.indexer.cli /path/to/sessions/

# Via API
curl -X POST http://localhost:8010/index \
  -H "Content-Type: application/json" \
  -d '{"session_path": "/path/to/session.jsonl"}'
```

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common issues and solutions.

### Quick Fixes

**Docker not found**:
- Ensure Docker is installed and running
- On Linux, make sure your user is in the `docker` group

**PostgreSQL connection failed**:
- Check if PostgreSQL is running: `docker ps` or `systemctl status postgresql`
- Verify credentials in `.env` file

**pgvector not available**:
- Use the pgvector Docker image: `pgvector/pgvector:pg16`
- Or install pgvector extension manually

**Migration failed**:
- Ensure database exists and credentials are correct
- Try running manually: `python -m reverie.cli migrate`

## Next Steps

After setup:
1. Index your Claude Code sessions
2. Open the dashboard at http://localhost:8010/reverie/
3. Query your memories via the search interface
4. Explore the entity-passage graph
