import { test, expect } from '@playwright/test';

test.describe('Memory Explorer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display search bar on page load', async ({ page }) => {
    // Check that the search input is visible
    const searchInput = page.getByPlaceholder(/Search memories/i);
    await expect(searchInput).toBeVisible();

    // Check that search button exists
    const searchButton = page.getByRole('button', { name: /Search/i });
    await expect(searchButton).toBeVisible();
  });

  test('should display Query/Stats tabs', async ({ page }) => {
    // Check Query tab
    const queryTab = page.getByRole('tab', { name: /Query/i });
    await expect(queryTab).toBeVisible();

    // Check Stats tab
    const statsTab = page.getByRole('tab', { name: /Stats/i });
    await expect(statsTab).toBeVisible();
  });

  test('should switch between Query and Stats tabs', async ({ page }) => {
    const queryTab = page.getByRole('tab', { name: /Query/i });
    const statsTab = page.getByRole('tab', { name: /Stats/i });

    // Initially Query tab should be active
    await expect(queryTab).toHaveAttribute('data-state', 'active');

    // Click Stats tab
    await statsTab.click();

    // Now Stats tab should be active
    await expect(statsTab).toHaveAttribute('data-state', 'active');
    await expect(queryTab).toHaveAttribute('data-state', 'inactive');
  });

  test('should show settings panel when Advanced tab is clicked', async ({ page }) => {
    // Click the Advanced button to show settings
    const advancedButton = page.getByRole('button', { name: 'Advanced' });
    await advancedButton.click();

    // Settings panel should show with sliders
    await expect(page.getByText('Query Settings')).toBeVisible({ timeout: 2000 });
    await expect(page.getByText('Damping (PPR)')).toBeVisible({ timeout: 2000 });
  });

  test('should switch back to Content tab from Advanced', async ({ page }) => {
    // Click Advanced to show settings
    const advancedButton = page.getByRole('button', { name: 'Advanced' });
    await advancedButton.click();
    await expect(page.getByText('Damping (PPR)')).toBeVisible({ timeout: 2000 });

    // Click Content to switch back
    const contentButton = page.getByRole('button', { name: 'Content' });
    await contentButton.click();
    await expect(page.getByText('Damping (PPR)')).not.toBeVisible({ timeout: 2000 });
  });

  test('should not allow search with empty query', async ({ page }) => {
    const searchButton = page.getByRole('button', { name: /Search/i });

    // Button should be disabled when input is empty
    await expect(searchButton).toBeDisabled();
  });

  test('should enable search button when query is entered', async ({ page }) => {
    const searchInput = page.getByPlaceholder(/Search memories/i);
    const searchButton = page.getByRole('button', { name: /Search/i });

    // Type a query
    await searchInput.fill('test query');

    // Button should now be enabled
    await expect(searchButton).toBeEnabled();
  });

  test('should show loading state during search', async ({ page }) => {
    const searchInput = page.getByPlaceholder(/Search memories/i);
    const searchButton = page.getByRole('button', { name: /Search/i });

    // Type and submit
    await searchInput.fill('test query');
    await searchButton.click();

    // Should show "Searching..." text while loading
    await expect(page.getByRole('button', { name: /Searching/i })).toBeVisible();
  });
});
