import { test, expect } from '@playwright/test';

test.describe('Stats Tab Layout', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Click on Stats tab
    await page.click('text=Stats');
    // Wait for Stats tab content to load
    await page.waitForSelector('text=Entities');
  });

  test('DataBrowser should not overflow the viewport', async ({ page }) => {
    // Get viewport height
    const viewportSize = page.viewportSize();
    const viewportHeight = viewportSize?.height || 720;

    // Find the DataBrowser container (the flex container with Entities/Passages)
    const dataBrowser = page.locator('[class*="flex"][class*="gap-4"][class*="overflow-hidden"]').first();

    // Check that DataBrowser is visible
    await expect(dataBrowser).toBeVisible();

    // Get bounding box
    const box = await dataBrowser.boundingBox();
    expect(box).not.toBeNull();

    if (box) {
      // The bottom of the DataBrowser should not exceed the viewport
      const bottomPosition = box.y + box.height;
      expect(bottomPosition).toBeLessThanOrEqual(viewportHeight + 1); // +1 for rounding
    }
  });

  test('Entity list should have scrollbar when content overflows', async ({ page }) => {
    // Find the entity list scroll container
    const scrollContainer = page.locator('[class*="overflow-auto"][class*="scrollbar-thin"]').first();

    await expect(scrollContainer).toBeVisible();

    // Check that container has overflow-auto
    const overflowY = await scrollContainer.evaluate((el) => {
      return window.getComputedStyle(el).overflowY;
    });

    expect(['auto', 'scroll']).toContain(overflowY);
  });

  test('Detail panel should have scrollbar when content overflows', async ({ page }) => {
    // Find the detail panel (second Card with overflow-auto)
    const detailPanel = page.locator('[class*="flex-\\[2\\]"]').first();

    await expect(detailPanel).toBeVisible();

    // Check the content area has overflow handling
    const contentArea = detailPanel.locator('[class*="overflow-auto"]');
    await expect(contentArea).toBeVisible();
  });

  test('Stats cards should not push DataBrowser out of view', async ({ page }) => {
    const viewportSize = page.viewportSize();
    const viewportHeight = viewportSize?.height || 720;

    // Check that the "Entities" or "Passages" button is visible within viewport
    const toggleButton = page.getByRole('button', { name: 'Entities' });
    await expect(toggleButton).toBeVisible();

    const buttonBox = await toggleButton.boundingBox();
    expect(buttonBox).not.toBeNull();

    if (buttonBox) {
      // Button should be within viewport
      expect(buttonBox.y).toBeLessThan(viewportHeight);
      expect(buttonBox.y + buttonBox.height).toBeLessThan(viewportHeight);
    }
  });

  test('switching between Entities and Passages should maintain layout', async ({ page }) => {
    // Start with Entities
    const entitiesButton = page.getByRole('button', { name: 'Entities' });
    const passagesButton = page.getByRole('button', { name: 'Passages' });

    // Click Passages
    await passagesButton.click();
    await page.waitForTimeout(500);

    // Get DataBrowser bounds
    const dataBrowser = page.locator('[class*="flex"][class*="gap-4"][class*="overflow-hidden"]').first();
    const boxAfterPassages = await dataBrowser.boundingBox();

    // Click back to Entities
    await entitiesButton.click();
    await page.waitForTimeout(500);

    const boxAfterEntities = await dataBrowser.boundingBox();

    // Layout should be similar
    expect(boxAfterPassages?.height).toBeCloseTo(boxAfterEntities?.height || 0, 0);
  });
});
