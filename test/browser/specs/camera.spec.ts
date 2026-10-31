import { expect, test } from '@playwright/test';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { bestOffset, expectNoErrors, firstLockMs, percentile, runHarness, truth, visibleRatio } from '../metrics';

const fakeVideo = join(process.cwd(), 'test/browser/public/fixtures/sweep.y4m');

test.describe('real getUserMedia path', () => {
  test.skip(!existsSync(fakeVideo), 'fake camera fixture was not generated (needs ffmpeg)');

  test.beforeEach(async ({}, testInfo) => {
    test.skip(testInfo.project.name !== 'camera-chromium', 'only the fake-camera Chromium project runs this');
  });

  test('opens the camera and tracks the target', async ({ page }) => {
    const t = truth('sweep');
    const r = await runHarness(page, { mode: 'camera', scenario: 'sweep', useImu: '0' });
    expectNoErrors(r);
    expect(r.frames.length).toBeGreaterThan(20);
    expect(r.stats?.path).toBe('track-processor');
    expect(firstLockMs(r)).toBeLessThan(2500);
    expect(visibleRatio(r)).toBeGreaterThan(0.5);
    const { errors } = bestOffset(r, t, 30);
    expect(errors.length).toBeGreaterThan(10);
    expect(percentile(errors, 0.95)).toBeLessThan(6);
  });

  test('accepts forceCopyPath on the camera stream', async ({ page }) => {
    const r = await runHarness(page, { mode: 'camera', scenario: 'sweep', useImu: '0', copy: '1' });
    expectNoErrors(r);
    expect(r.stats?.path).toBe('copy');
    expect(r.frames.length).toBeGreaterThan(15);
  });
});
