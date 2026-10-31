import { expect, test } from '@playwright/test';
import { expectNoErrors } from '../metrics';

test.describe('mobile layout', () => {
  test('keeps tracking across a viewport/orientation change', async ({ page }, testInfo) => {
    test.skip(!testInfo.project.name.startsWith('mobile'), 'touch/orientation emulation projects');
    await page.goto('/?mode=replay&scenario=sweep&useImu=0&runMs=3000');
    await page.waitForFunction(() => typeof window.__fastarTest !== 'undefined' && !window.__fastarTest.done);
    await page.locator('#start').tap();
    await page.waitForTimeout(1000);
    await page.setViewportSize({ width: 820, height: 420 });
    await page.waitForFunction(() => window.__fastarTest.done, undefined, { timeout: 30_000 });
    const r = await page.evaluate(() => window.__fastarTest);
    expectNoErrors(r);
    expect(r.frames.length).toBeGreaterThan(15);
    const views = new Set(r.frames.map((f) => `${f.view[0]}x${f.view[1]}`));
    expect(views.size).toBeGreaterThan(1);
    expect(r.frames.some((f) => f.poses.length)).toBe(true);
  });
});
