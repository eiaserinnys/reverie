# Troubleshooting Guide

This guide helps you resolve common issues when setting up and using Reverie.

## Setup Wizard Issues

### Setup Wizard Not Loading

**Symptom**: Dashboard shows "Connection Error" or endless loading spinner.

**Causes and Solutions**:

1. **Backend not running**
   ```bash
   # Start the backend server
   python -m reverie.mcp.server
   ```

2. **Wrong port**
   - Default port is `8010`
   - Check if another process is using the port: `lsof -i :8010`
   - Or set a different port: `REVERIE_API_PORT=8011 python -m reverie.mcp.server`

3. **CORS issues**
   - If running dashboard separately, ensure `REVERIE_CORS_ORIGIN` is set correctly
   - Default: `http://localhost:5173`

### Prerequisites Check Fails

**Python Version**

The wizard checks for Python 3.11+. If this fails:
```bash
python --version
# If below 3.11, install newer version
pyenv install 3.11.0
pyenv local 3.11.0
```

## PostgreSQL Issues

### Cannot Connect to PostgreSQL

**Symptom**: "could not connect to server" or "connection refused"

**Solutions**:

1. **PostgreSQL not running**
   ```bash
   # Check status
   sudo systemctl status postgresql

   # Start if stopped
   sudo systemctl start postgresql
   ```

2. **Wrong host/port**
   - Default: `localhost:5432`
   - Check PostgreSQL is listening: `sudo netstat -tlnp | grep 5432`

3. **Authentication failed**
   - Verify username/password
   - Check `pg_hba.conf` allows your connection method:
     ```
     # IPv4 local connections:
     host    all    all    127.0.0.1/32    md5
     ```

4. **Database doesn't exist**
   ```bash
   # Create the database
   createdb -U postgres reverie
   ```

### pgvector Extension Not Available

**Symptom**: "pgvector extension not available"

**Solutions**:

1. **Install pgvector**

   **Ubuntu/Debian**:
   ```bash
   sudo apt install postgresql-15-pgvector
   ```

   **macOS (Homebrew)**:
   ```bash
   brew install pgvector
   ```

   **From source**:
   ```bash
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   make
   sudo make install
   ```

2. **Using Docker (Recommended)**
   ```bash
   docker run -d \
     --name reverie-postgres \
     -e POSTGRES_USER=reverie \
     -e POSTGRES_PASSWORD=reverie \
     -e POSTGRES_DB=reverie \
     -p 5432:5432 \
     pgvector/pgvector:pg15
   ```

3. **Enable the extension**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

### pgvector Available but Not Installed

**Symptom**: Check shows pgvector is available but not installed.

**Solution**:
```sql
-- Connect to your database and run:
CREATE EXTENSION vector;
```

## OpenAI API Key Issues

### Invalid API Key

**Symptom**: "Invalid API key" error during setup.

**Solutions**:

1. **Verify the key format**
   - Should start with `sk-`
   - No extra spaces or newlines

2. **Check key validity**
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer sk-your-key-here"
   ```

3. **Check account status**
   - Visit [platform.openai.com](https://platform.openai.com)
   - Ensure your account has credits and is not suspended

### API Key Not Being Saved

**Symptom**: API key validation succeeds but setup fails.

**Solutions**:

1. **Check file permissions**
   ```bash
   ls -la .env
   # Should be writable by current user
   chmod 600 .env
   ```

2. **Manual addition**
   ```bash
   echo "OPENAI_API_KEY=sk-your-key-here" >> .env
   ```

## Database Migration Issues

### Migration Fails

**Symptom**: "Migration failed" error in Setup Wizard.

**Solutions**:

1. **Check database connection**
   - Verify credentials are correct
   - Ensure database exists

2. **Check for existing tables**
   ```sql
   \dt  -- List tables in psql
   ```

3. **Manual migration**
   ```bash
   python -m reverie.db.migrate
   ```

4. **Reset and retry** (development only)
   ```sql
   DROP SCHEMA public CASCADE;
   CREATE SCHEMA public;
   ```
   Then run migrations again.

### Missing Tables After Migration

**Symptom**: Verification shows "Tables not found"

**Solutions**:

1. **Check migration ran against correct database**
   ```bash
   echo $REVERIE_DB_NAME
   # Should match your target database
   ```

2. **Re-run migrations**
   ```bash
   python -m reverie.db.migrate
   ```

## Profile Setup Issues

### Invalid Session Path

**Symptom**: "sessions_path must be an absolute path"

**Solutions**:

1. **Use absolute paths**
   ```
   /home/user/.claude/projects/sessions  ✓
   ~/.claude/projects/sessions           ✓ (expanded automatically)
   ./sessions                            ✗ (relative path)
   ```

2. **Path doesn't need to exist**
   - The wizard creates the directory if it doesn't exist
   - Just ensure the parent directory is writable

## Verification Issues

### All Checks Fail

**Symptom**: Verification phase shows all checks failed.

**Solutions**:

1. **Go back and complete previous phases**
   - Each phase must be completed in order
   - Click on a phase in the sidebar to go back

2. **Restart the setup**
   ```bash
   rm .reverie/setup.json
   # Restart server and reload dashboard
   ```

## General Tips

### Reset Setup State

To start fresh:
```bash
# Remove setup state file
rm .reverie/setup.json

# Optionally clear .env
rm .env

# Restart server
python -m reverie.mcp.server
```

### Check Logs

Server logs show detailed error information:
```bash
# Run with debug logging
REVERIE_LOG_LEVEL=DEBUG python -m reverie.mcp.server
```

### Environment Variable Priority

Configuration is loaded in this order (later overrides earlier):
1. Default values in code
2. `.env` file
3. Environment variables set in shell
4. Setup Wizard entries

### Getting Help

If you're still stuck:
1. Check the [GitHub Issues](https://github.com/eiaserinnys/reverie/issues)
2. Open a new issue with:
   - Your OS and Python version
   - The exact error message
   - Steps to reproduce
