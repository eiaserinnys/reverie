import { chromium } from 'playwright';

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });

await page.goto(process.env.REVERIE_URL || 'http://localhost:8010/reverie/');
await page.waitForTimeout(1000);

// Search
await page.fill('input[placeholder*="Search"]', '에이전트');
await page.click('button:has-text("Search")');
await page.waitForTimeout(6000);

// Click first result in the left list
const resultItem = page.locator('button').filter({ hasText: /S\d+-T\d+/ }).first();
if (await resultItem.count() > 0) {
  await resultItem.click();
  await page.waitForTimeout(2000);
}

// Try to scroll inside the content panel
const contentPanel = page.locator('text=Passage Content').locator('..').locator('..');
console.log('Content panel found:', await contentPanel.count());

// Screenshot before scroll
await page.screenshot({ path: '/tmp/reverie-before-scroll.png' });

// Try scrolling by mouse wheel on the right side
await page.mouse.move(1700, 800);
await page.mouse.wheel(0, 500);
await page.waitForTimeout(500);

// Screenshot after scroll attempt
await page.screenshot({ path: '/tmp/reverie-after-scroll.png' });

// Crop right panel
const viewport = page.viewportSize();
const rightWidth = Math.floor(viewport.width * 0.25);
await page.screenshot({ 
  path: '/tmp/reverie-right.png',
  clip: {
    x: viewport.width - rightWidth,
    y: 0,
    width: rightWidth,
    height: viewport.height
  }
});

await browser.close();
console.log('Screenshots saved');
