import { expect, test } from '@playwright/test';
import {
  bestOffset, expectNoErrors, firstLockMs, percentile, runHarness, settled, staticJitterMm, truth,
  videoLockedRatio, videoTimeErrors, visibleRatio,
} from '../metrics';

test.describe('tracker end to end', () => {
  test('replays a static target with IMU: locks, stays visible, and is still', async ({ page }) => {
    const t = truth('static');
    const r = await runHarness(page, { mode: 'replay', scenario: 'static' });
    expectNoErrors(r);
    expect(r.frames.length).toBeGreaterThan(30);
    expect(firstLockMs(r)).toBeLessThan(1000);
    expect(visibleRatio(r)).toBeGreaterThan(0.9);
    const errors = bestOffset(r, t).errors;
    expect(percentile(errors, 0.95)).toBeLessThan(1.5);
    expect(staticJitterMm(r)).toBeLessThan(0.5);
    expect(r.stats?.path).toBe('copy');
    expect(r.stats?.sensors).not.toBe('none');
  });

  test('replays a slow sweep: overlay follows the target', async ({ page }, testInfo) => {
    const t = truth('sweep');
    const r = await runHarness(page, { mode: 'replay', scenario: 'sweep' });
    expectNoErrors(r);
    expect(r.frames.length).toBeGreaterThan(60);
    expect(firstLockMs(r)).toBeLessThan(1200);
    expect(visibleRatio(r)).toBeGreaterThan(0.9);
    // WebKit's scheduling produces a wider spread on this scene; Chromium is
    // held to the tighter numbers.
    const webkit = testInfo.project.name.includes('webkit') || testInfo.project.name.includes('safari');
    const errors = bestOffset(settled(r), t).errors;
    expect(percentile(errors, 0.5)).toBeLessThan(webkit ? 3 : 1.2);
    expect(percentile(errors, 0.95)).toBeLessThan(webkit ? 4.5 : 3.5);
  });

  test('replays an occlusion: reports lost and reacquires', async ({ page }) => {
    const r = await runHarness(page, { mode: 'replay', scenario: 'occlusion' });
    expectNoErrors(r);
    expect(r.events.found.length).toBeGreaterThanOrEqual(1);
    expect(r.events.lost.length).toBeGreaterThanOrEqual(1);
    expect(visibleRatio(r)).toBeGreaterThan(0.7);
  });

  test('replays a video target: the clock locks and matches ground truth', async ({ page }) => {
    const t = truth('video');
    const r = await runHarness(page, { mode: 'replay', scenario: 'video', target: 'targets-video.bin' });
    expectNoErrors(r);
    expect(visibleRatio(r)).toBeGreaterThan(0.85);
    expect(videoLockedRatio(r)).toBeGreaterThan(0.8);
    const errors = videoTimeErrors(r, t);
    expect(errors.length).toBeGreaterThan(20);
    expect(percentile(errors, 0.95)).toBeLessThan(0.05);
  });

  test('tracks a live canvas stream', async ({ page }) => {
    const t = truth('static');
    const r = await runHarness(page, { mode: 'stream', scenario: 'static', useImu: '0' });
    expectNoErrors(r);
    expect(r.frames.length).toBeGreaterThan(20);
    expect(firstLockMs(r)).toBeLessThan(2000);
    expect(visibleRatio(r)).toBeGreaterThan(0.5);
    const { errors } = bestOffset(r, t, 9);
    expect(errors.length).toBeGreaterThan(10);
    expect(percentile(errors, 0.95)).toBeLessThan(5);
  });

  test('syncDisplay emits the frame image', async ({ page }) => {
    const r = await runHarness(page, { mode: 'stream', scenario: 'static', useImu: '0', sync: '1' });
    expectNoErrors(r);
    expect(r.frames.length).toBeGreaterThan(15);
    const withImage = r.frames.filter((f) => f.hasImage).length;
    expect(withImage / r.frames.length).toBeGreaterThan(0.9);
  });

  test('forceCopyPath uses the main-thread grabber', async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'chromium-desktop', 'Chromium-only path switch');
    const r = await runHarness(page, { mode: 'stream', scenario: 'static', useImu: '0', copy: '1' });
    expectNoErrors(r);
    expect(r.stats?.path).toBe('copy');
    expect(r.frames.length).toBeGreaterThan(15);
  });

  test('works without requestVideoFrameCallback', async ({ page }) => {
    const r = await runHarness(page, { mode: 'stream', scenario: 'static', useImu: '0', norvfc: '1' });
    expectNoErrors(r);
    expect(r.frames.length).toBeGreaterThan(15);
  });

  test('falls back from WebGL2 to the 2D luma path', async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'chromium-desktop', 'Chromium-only fallback switch');
    const r = await runHarness(page, { mode: 'stream', scenario: 'static', useImu: '0', 'no-webgl2': '1' });
    expectNoErrors(r);
    expect(r.frames.length).toBeGreaterThan(10);
  });
});
