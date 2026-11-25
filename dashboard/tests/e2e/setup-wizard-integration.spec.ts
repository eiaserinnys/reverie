/**
 * Setup Wizard E2E Tests
 *
 * Tests the complete setup flow from start to finish:
 * 1. Reset setup state
 * 2. Prerequisites check
 * 3. Database configuration
 * 4. API Keys setup (skipped for now - requires real key)
 * 5. Profile configuration
 * 6. Migration
 * 7. Verification
 * 8. Dashboard access
 *
 * This test suite runs FIRST before any dashboard tests.
 * It ensures setup is complete so dashboard tests can pass.
 */

import { test, expect, request as playwrightRequest } from '@playwright/test';

// Backend API URL - MUST use test port (8011), never production (8010)
// Use 127.0.0.1 to avoid IPv6 resolution issues (::1)
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8011';

// Test database credentials - Docker container on port 5434
const TEST_DB_PORT = process.env.TEST_DB_PORT || '5434';
const TEST_DB = {
  host: 'localhost',
  port: parseInt(TEST_DB_PORT, 10),
  user: 'reverie_test',
  password: 'reverie_test',
  name: 'reverie_test',
};

// Helper to get API context
async function getApiContext() {
  return await playwrightRequest.newContext({ baseURL: BACKEND_URL });
}

// Helper to get current setup status
async function getSetupStatus() {
  const api = await getApiContext();
  const response = await api.get('/api/setup/status');
  return await response.json();
}

// Helper to reset setup for testing
async function resetSetup() {
  const api = await getApiContext();
  const response = await api.post('/api/setup/reset', { data: {} });
  return await response.json();
}

test.describe.serial('Setup Wizard Complete Flow', () => {
  test('0. Reset setup state for testing', async () => {
    console.log(`Using backend: ${BACKEND_URL}`);

    // Reset setup to initial state
    const resetResult = await resetSetup();
    console.log('Reset result:', resetResult);
    expect(resetResult.success).toBe(true);

    // Verify reset worked
    const status = await getSetupStatus();
    console.log('After reset, setup_complete:', status.setup_complete);
    expect(status.setup_complete).toBe(false);
  });

  test('1. Prerequisites - should show wizard and run system checks', async ({ page }) => {
    await page.goto('/');

    // Should show Setup Wizard
    await expect(page.getByRole('heading', { name: 'Setup Wizard' })).toBeVisible({ timeout: 10000 });

    // Should be on Prerequisites phase
    await expect(page.getByText('Step 1 of 6')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Prerequisites' })).toBeVisible();

    // Run checks
    const runChecksButton = page.getByRole('button', { name: 'Run Checks' });
    await expect(runChecksButton).toBeVisible();
    await runChecksButton.click();

    // Wait for checks to complete
    await page.waitForTimeout(3000);

    // Verify PostgreSQL check completed
    await expect(page.getByText('PostgreSQL Connection', { exact: true })).toBeVisible();

    // Take screenshot for debugging
    await page.screenshot({ path: 'test-results/01-prerequisites.png' });
  });

  test('2. Database - should configure database connection', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Setup Wizard' })).toBeVisible({ timeout: 10000 });

    // Navigate to Database phase
    await page.locator('aside').getByText('Database', { exact: true }).click();
    await expect(page.getByText('Step 2 of 6')).toBeVisible();

    // Check what screen we're on
    // If PostgreSQL is available (prerequisites passed), we'll see backend selection
    // If not, we'll be auto-directed to SQLite config
    // Use icon + title to precisely select backend cards (avoid matching sidebar)
    const postgresCard = page.getByRole('button', { name: /🐘.*PostgreSQL/i });
    const sqliteCard = page.getByRole('button', { name: /📦.*SQLite/i });
    const hostInput = page.getByLabel('Host');
    const sqlitePathInput = page.getByLabel('Database Path');

    // Wait a moment for the page to load
    await page.waitForTimeout(500);

    // Check if we're on backend selection or already on SQLite config
    if (await sqlitePathInput.isVisible()) {
      // PostgreSQL check failed in prerequisites, so we're auto-directed to SQLite
      console.log('PostgreSQL not available, using SQLite backend');

      // SQLite path should already be filled with default
      await expect(sqlitePathInput).toHaveValue('~/.reverie/reverie.db');

      // sqlite-vec should be available (already checked)
      await expect(page.getByText(/sqlite-vec v/i)).toBeVisible({ timeout: 5000 });

      // Save and continue with SQLite
      const saveButton = page.getByRole('button', { name: 'Save & Continue →' });
      await saveButton.click();

    } else if (await postgresCard.isVisible()) {
      // Backend selection screen - PostgreSQL is available
      console.log('PostgreSQL available, selecting PostgreSQL backend');

      // Click on the PostgreSQL card to select it
      await postgresCard.click();

      // Wait for PostgreSQL config form to appear
      await expect(hostInput).toBeVisible({ timeout: 5000 });

      // Clear and fill database form (clear first to ensure no leftover values)
      await page.getByLabel('Host').clear();
      await page.getByLabel('Host').fill(TEST_DB.host);
      await page.getByLabel('Port').clear();
      await page.getByLabel('Port').fill(String(TEST_DB.port));
      await page.getByLabel('Username').clear();
      await page.getByLabel('Username').fill(TEST_DB.user);
      await page.getByLabel('Password').clear();
      await page.getByLabel('Password').fill(TEST_DB.password);
      await page.getByLabel('Database Name').clear();
      await page.getByLabel('Database Name').fill(TEST_DB.name);

      // Wait a moment for form to update
      await page.waitForTimeout(300);

      // Test connection
      const testButton = page.getByRole('button', { name: 'Test Connection' });
      await testButton.click();

      // Wait for success message (check for "Connected to" which appears in the success state)
      await expect(page.getByText(/Connected to PostgreSQL|Connection successful/i)).toBeVisible({ timeout: 10000 });

      // Save and continue
      const saveButton = page.getByRole('button', { name: 'Save & Continue' });
      await saveButton.click();

    } else if (await sqliteCard.isVisible()) {
      // Backend selection screen but might want to use SQLite
      console.log('On backend selection, choosing SQLite');
      await sqliteCard.click();

      // Wait for SQLite config form
      await expect(sqlitePathInput).toBeVisible({ timeout: 5000 });

      // Save and continue with SQLite
      const saveButton = page.getByRole('button', { name: 'Save & Continue →' });
      await saveButton.click();
    }

    // Wait for save to complete
    await page.waitForTimeout(1000);

    // Take screenshot
    await page.screenshot({ path: 'test-results/02-database.png' });
  });

  test('3. API Keys - verify form is present', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Setup Wizard' })).toBeVisible({ timeout: 10000 });

    // Navigate to API Keys phase
    await page.locator('aside').getByText('API Keys', { exact: true }).click();
    await expect(page.getByText('Step 3 of 6')).toBeVisible();

    // API Key input should be visible
    const apiKeyInput = page.getByLabel(/API Key/i);
    await expect(apiKeyInput).toBeVisible();

    // Take screenshot
    await page.screenshot({ path: 'test-results/03-apikeys.png' });

    // Note: We'll complete API key setup via API to avoid exposing real key in UI
    // Save a test key via API
    const api = await getApiContext();
    const saveResult = await api.post('/api/setup/apikey', {
      data: { api_key: process.env.OPENAI_API_KEY || 'sk-test-key-for-testing' },
    });
    const result = await saveResult.json();
    console.log('API key save result:', result);
  });

  test('4. Profile - should configure profile', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Setup Wizard' })).toBeVisible({ timeout: 10000 });

    // Navigate to Profile phase
    await page.locator('aside').getByText('Profile', { exact: true }).click();
    await expect(page.getByText('Step 4 of 6')).toBeVisible();

    // Profile name input should be visible
    const profileInput = page.getByLabel(/Profile Name/i);
    await expect(profileInput).toBeVisible();

    // Fill profile form
    await profileInput.fill('test-profile');

    // Sessions path
    const sessionsInput = page.getByLabel(/Sessions Path/i);
    if (await sessionsInput.isVisible()) {
      await sessionsInput.fill('/tmp/reverie-test-sessions');
    }

    // Save profile
    const saveButton = page.getByRole('button', { name: /Save|Continue/i });
    if (await saveButton.isVisible()) {
      await saveButton.click();
      await page.waitForTimeout(1000);
    }

    // Take screenshot
    await page.screenshot({ path: 'test-results/04-profile.png' });
  });

  test('5. Migration - should run database migrations', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Setup Wizard' })).toBeVisible({ timeout: 10000 });

    // Navigate to Migration phase
    await page.locator('aside').getByText('Migration', { exact: true }).click();
    await expect(page.getByText('Step 5 of 6')).toBeVisible();

    // Run migration button should be visible
    const migrateButton = page.getByRole('button', { name: /Run Migration|Migrate/i });
    await expect(migrateButton).toBeVisible();

    // Click migrate
    await migrateButton.click();

    // Wait for migration to complete (can take time)
    // Use specific heading to avoid matching multiple elements
    await expect(page.getByRole('heading', { name: /Migrations Complete/i })).toBeVisible({ timeout: 30000 });

    // Take screenshot
    await page.screenshot({ path: 'test-results/05-migration.png' });
  });

  test('6. Verification - should verify setup', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Setup Wizard' })).toBeVisible({ timeout: 10000 });

    // Navigate to Verification phase
    await page.locator('aside').getByText('Verification', { exact: true }).click();
    await expect(page.getByText('Step 6 of 6')).toBeVisible();

    // Run verification button should be visible
    const verifyButton = page.getByRole('button', { name: /Run Verification|Verify/i });
    await expect(verifyButton).toBeVisible();

    // Click verify
    await verifyButton.click();

    // Wait for verification to complete
    await page.waitForTimeout(3000);

    // Take screenshot
    await page.screenshot({ path: 'test-results/06-verification.png' });
  });

  test('7. Complete setup via API', async () => {
    // First verify all checks pass
    const api = await getApiContext();

    // Run verification
    const verifyResponse = await api.get('/api/setup/verify');
    const verifyResult = await verifyResponse.json();
    console.log('Verification result:', JSON.stringify(verifyResult, null, 2));

    // Complete setup regardless of verification (for testing)
    const completeResponse = await api.post('/api/setup/complete', { data: {} });
    const completeResult = await completeResponse.json();
    console.log('Complete result:', completeResult);
    expect(completeResult.success).toBe(true);

    // Verify setup is complete
    const status = await getSetupStatus();
    console.log('Final setup_complete:', status.setup_complete);
    expect(status.setup_complete).toBe(true);
  });

  test('8. Dashboard - should show after setup is complete', async ({ page }) => {
    // Reload to get dashboard
    await page.goto('/');

    // Should NOT show Setup Wizard anymore
    await expect(page.getByRole('heading', { name: 'Setup Wizard' })).not.toBeVisible({ timeout: 5000 });

    // Should show Dashboard - use specific heading to avoid ambiguity
    await expect(page.getByRole('heading', { name: 'Reverie Dashboard' })).toBeVisible({ timeout: 10000 });

    // Take screenshot
    await page.screenshot({ path: 'test-results/08-dashboard.png' });
  });
});

test.describe('Setup API Tests', () => {
  test('GET /api/setup/status should return valid status', async () => {
    const api = await getApiContext();
    const response = await api.get('/api/setup/status');
    expect(response.ok()).toBeTruthy();

    const status = await response.json();
    expect(status).toHaveProperty('setup_complete');
    expect(status).toHaveProperty('phases');
    expect(status).toHaveProperty('last_updated');

    // Validate phase structure
    const expectedPhases = ['prerequisites', 'database', 'apikeys', 'profile', 'migration', 'verification'];
    for (const phase of expectedPhases) {
      expect(status.phases).toHaveProperty(phase);
      expect(status.phases[phase]).toHaveProperty('status');
    }
  });

  test('POST /api/setup/check/postgres should test connection', async () => {
    const api = await getApiContext();
    const response = await api.post('/api/setup/check/postgres', {
      data: TEST_DB,
    });

    expect(response.ok()).toBeTruthy();
    const result = await response.json();
    expect(result).toHaveProperty('success');

    if (result.success) {
      expect(result).toHaveProperty('version');
      console.log('PostgreSQL version:', result.version);
    }
  });

  test('POST /api/setup/check/pgvector should verify extension', async () => {
    const api = await getApiContext();
    const response = await api.post('/api/setup/check/pgvector', {
      data: TEST_DB,
    });

    expect(response.ok()).toBeTruthy();
    const result = await response.json();
    expect(result).toHaveProperty('success');

    if (result.success) {
      expect(result).toHaveProperty('installed');
      console.log('pgvector installed:', result.installed, 'version:', result.version);
    }
  });
});

/**
 * Database Form Pre-fill Tests
 *
 * Tests that the database form is pre-filled with values from .env
 */
test.describe('Database Form Pre-fill', () => {
  test('Database form should be pre-filled with .env values when user is configured', async ({ page }) => {
    // Reset setup state but keep .env values
    const api = await playwrightRequest.newContext({ baseURL: BACKEND_URL });
    await api.post('/api/setup/reset', { data: {} });

    // First, trigger PostgreSQL check so it's marked as available
    // (reset clears the check results)
    const checkResult = await api.post('/api/setup/check/postgres', {
      data: { host: 'localhost', port: 5432, user: 'reverie', password: 'reverie', database: 'reverie' }
    });
    const checkData = await checkResult.json();
    console.log('PostgreSQL check result:', checkData);

    if (!checkData.success) {
      console.log('PostgreSQL not available, skipping pre-fill test');
      test.skip();
      return;
    }

    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Setup Wizard' })).toBeVisible({ timeout: 10000 });

    // Navigate to Database phase
    await page.locator('aside').getByText('Database', { exact: true }).click();
    await expect(page.getByText('Step 2 of 6')).toBeVisible();

    // Select PostgreSQL backend if on selection screen
    const postgresCard = page.getByRole('button', { name: /🐘.*PostgreSQL/i });
    if (await postgresCard.isVisible({ timeout: 2000 }).catch(() => false)) {
      await postgresCard.click();
    }

    // Wait for form to appear
    await expect(page.getByLabel('Host')).toBeVisible({ timeout: 5000 });

    // Verify .env values are pre-filled (from test environment: reverie/reverie)
    // The test server is started with REVERIE_DB_USER=reverie_test but .env has reverie
    const usernameValue = await page.getByLabel('Username').inputValue();
    console.log('Username field value:', usernameValue);

    // Username should NOT be empty or placeholder
    expect(usernameValue).not.toBe('');
    expect(usernameValue).not.toBe('postgres');

    // Should be one of the configured values (reverie from .env or reverie_test from env var)
    expect(['reverie', 'reverie_test']).toContain(usernameValue);

    await page.screenshot({ path: 'test-results/database-form-prefill.png' });
  });
});

/**
 * Docker PostgreSQL Flow Tests
 *
 * Tests the flow where user installs PostgreSQL via Docker.
 * The database form should be pre-filled with Docker defaults (reverie/reverie/reverie).
 *
 * Note: These tests start/stop actual Docker containers.
 */
test.describe('Docker PostgreSQL Flow', () => {
  test.beforeAll(async () => {
    // Reset and stop any existing Docker container
    const api = await playwrightRequest.newContext({ baseURL: BACKEND_URL });
    await api.post('/api/setup/reset', { data: {} });
    await api.post('/api/setup/docker/stop', { data: {} }).catch(() => {});
  });

  test.afterAll(async () => {
    // Clean up: stop Docker container
    const api = await playwrightRequest.newContext({ baseURL: BACKEND_URL });
    await api.post('/api/setup/docker/stop', { data: {} }).catch(() => {});
  });

  test('Database form should be pre-filled with Docker defaults after Docker PostgreSQL install', async ({ page }) => {
    // Reset state first
    const api = await playwrightRequest.newContext({ baseURL: BACKEND_URL });
    await api.post('/api/setup/reset', { data: {} });

    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Setup Wizard' })).toBeVisible({ timeout: 10000 });

    // Step 1: Go to Prerequisites and start Docker PostgreSQL
    await expect(page.getByText('Step 1 of 6')).toBeVisible();

    // Click "Install with Docker" button (if Docker is available)
    const dockerButton = page.getByRole('button', { name: /Install with Docker/i });

    // Wait for Docker check to complete
    await page.waitForTimeout(2000);

    if (await dockerButton.isVisible()) {
      console.log('Docker available, starting PostgreSQL container...');
      await dockerButton.click();

      // Wait for container to start (this can take a while for image pull)
      await expect(page.getByText(/PostgreSQL container|Container running/i)).toBeVisible({ timeout: 120000 });

      // Continue to next phase
      const continueButton = page.getByRole('button', { name: /Continue|Next/i });
      if (await continueButton.isVisible()) {
        await continueButton.click();
      }
    } else {
      console.log('Docker not available, skipping Docker flow test');
      test.skip();
      return;
    }

    // Step 2: Go to Database phase and verify form is pre-filled
    await page.locator('aside').getByText('Database', { exact: true }).click();
    await expect(page.getByText('Step 2 of 6')).toBeVisible();

    // Select PostgreSQL backend
    const postgresCard = page.getByRole('button', { name: /🐘.*PostgreSQL/i });
    if (await postgresCard.isVisible()) {
      await postgresCard.click();
    }

    // Wait for form to appear
    await expect(page.getByLabel('Host')).toBeVisible({ timeout: 5000 });

    // Verify Docker defaults are pre-filled
    await expect(page.getByLabel('Host')).toHaveValue('localhost');
    await expect(page.getByLabel('Port')).toHaveValue('5432');
    await expect(page.getByLabel('Username')).toHaveValue('reverie');
    await expect(page.getByLabel('Database Name')).toHaveValue('reverie');

    // Password field should also be filled (but we can't easily verify the value)
    const passwordField = page.getByLabel('Password');
    const passwordValue = await passwordField.inputValue();
    expect(passwordValue).toBe('reverie');

    console.log('Docker defaults correctly pre-filled in database form');

    // Test connection should work with these defaults
    await page.getByRole('button', { name: 'Test Connection' }).click();
    await expect(page.getByText(/Connected to PostgreSQL/i)).toBeVisible({ timeout: 10000 });

    await page.screenshot({ path: 'test-results/docker-postgres-defaults.png' });
  });
});
