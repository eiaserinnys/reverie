import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for Reverie Dashboard E2E tests.
 * @see https://playwright.dev/docs/test-configuration
 */
// Use test port if specified, otherwise default to TEST port (not production)
// CRITICAL: Default to 8011 (test) not 8010 (production) to prevent accidents
const vitePort = process.env.VITE_PORT || '5173';
const apiPort = process.env.VITE_API_PORT || '8011';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false, // Run tests sequentially so setup completes first
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1, // Single worker to ensure sequential execution
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: `http://localhost:${vitePort}/reverie/`,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'setup',
      testMatch: /setup-wizard-integration\.spec\.ts/,
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'dashboard',
      testMatch: /^(?!setup-wizard).*\.spec\.ts$/,
      dependencies: ['setup'], // Run after setup completes
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: `VITE_API_PORT=${apiPort} npm run dev -- --port ${vitePort}`,
    url: `http://localhost:${vitePort}/reverie/`,
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },
});
