import { test, expect } from '@playwright/test';

test.describe('Indexer Status', () => {
  test.beforeEach(async ({ page, baseURL }) => {
    // Navigate to the app
    await page.goto(baseURL || '/');
    await page.waitForLoadState('domcontentloaded');
  });

  test('should display workers count from API', async ({ page }) => {
    // Intercept the API call to see what's returned
    let indexStatusResponse: unknown = null;

    page.on('response', async (response) => {
      if (response.url().includes('/api/index/status')) {
        try {
          indexStatusResponse = await response.json();
          console.log('Index status response:', indexStatusResponse);
        } catch {
          console.log('Failed to parse index status response');
        }
      }
    });

    // Click on Stats tab
    await page.getByRole('tab', { name: 'Stats' }).click();

    // Wait for the index status API call
    await page.waitForTimeout(4000);

    // Log what we got
    console.log('Final index status:', indexStatusResponse);

    // Check that the Workers card displays the correct value
    const workersCard = page.locator('text=Workers').locator('..').locator('..');
    await expect(workersCard).toBeVisible();

    // Get the value displayed
    const workersValue = await workersCard.locator('.text-2xl').textContent();
    console.log('Workers displayed:', workersValue);

    // If API returned workers > 0, UI should show it
    if (indexStatusResponse && (indexStatusResponse as { workers?: number }).workers) {
      const expectedWorkers = (indexStatusResponse as { workers: number }).workers;
      expect(parseInt(workersValue || '0')).toBe(expectedWorkers);
    }
  });

  test('should fetch /api/index/status endpoint', async ({ page }) => {
    // Click on Stats tab
    await page.getByRole('tab', { name: 'Stats' }).click();

    // Make a direct API call from the page context
    const apiResponse = await page.evaluate(async () => {
      const res = await fetch('/api/index/status');
      return res.json();
    });

    console.log('Direct API response:', apiResponse);

    expect(apiResponse).toHaveProperty('workers');
    expect(apiResponse).toHaveProperty('running');
    expect(apiResponse).toHaveProperty('queue_size');
  });
});
