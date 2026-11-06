import { expect, test, type TestInfo } from '@playwright/test';
import {
  bestOffset, bestSurfaceErrors, expectNoErrors, firstLockMs, frameTruth, percentile, runHarness, settled, staticJitterMm, targetOverlayErrors, truth,
  videoLockedRatio, videoTimeErrors, visibleRatio,
} from '../metrics';

/** WebKitGTK (the Linux Playwright build) does not deliver canvas capture frames. */
function webkitLinux(testInfo: TestInfo): boolean {
  const name = testInfo.project.name;
  return process.platform === 'linux' && (name.includes('webkit') || name.includes('safari'));
}

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

  test('replays a slow sweep: overlay follows the target', async ({ page }) => {
    const t = truth('sweep');
    const r = await runHarness(page, { mode: 'replay', scenario: 'sweep' });
    expectNoErrors(r);
    expect(r.frames.length).toBeGreaterThan(60);
    expect(firstLockMs(r)).toBeLessThan(1200);
    expect(visibleRatio(r)).toBeGreaterThan(0.9);
    // Browser scheduling adds a few outlier frames; the median is the tight
    // signal, while p95 catches a genuinely broken pose. A shared CI runner
    // can delay a handful of frames enough for their pose to be one media tick
    // stale (~10 px), so leave p95 some headroom rather than flake on that.
    const errors = bestOffset(settled(r), t).errors;
    expect(percentile(errors, 0.5)).toBeLessThan(3);
    expect(percentile(errors, 0.95)).toBeLessThan(12);
  });

  test('replays a page that curls: the bending target surface follows it', async ({ page }) => {
    const t = truth('curl');
    const r = await runHarness(page, { mode: 'replay', scenario: 'curl', deformable: 1 });
    expectNoErrors(r);
    expect(visibleRatio(r)).toBeGreaterThan(0.9);
    const errors = bestSurfaceErrors(settled(r), t);
    expect(errors.length).toBeGreaterThan(60);
    // A rigid plane misses the curl by several pixels at the grid's edges.
    expect(percentile(errors, 0.5)).toBeLessThan(1.5);
    expect(percentile(errors, 0.95)).toBeLessThan(4);
  });

  test('replays an occlusion: reports lost and reacquires', async ({ page }) => {
    const r = await runHarness(page, { mode: 'replay', scenario: 'occlusion' });
    expectNoErrors(r);
    expect(r.events.found.length).toBeGreaterThanOrEqual(1);
    expect(r.events.lost.length).toBeGreaterThanOrEqual(1);
    expect(visibleRatio(r)).toBeGreaterThan(0.7);
  });

  test('replays a video target: the clock locks and matches ground truth', async ({ page }, testInfo) => {
    const t = truth('video');
    const r = await runHarness(page, { mode: 'replay', scenario: 'video', target: 'targets-video.bin' });
    expectNoErrors(r);
    expect(visibleRatio(r)).toBeGreaterThan(0.85);
    expect(videoLockedRatio(r)).toBeGreaterThan(0.8);
    const errors = videoTimeErrors(r, t);
    expect(errors.length).toBeGreaterThan(20);
    const webkit = testInfo.project.name.includes('webkit') || testInfo.project.name.includes('safari');
    expect(percentile(errors, 0.95)).toBeLessThan(webkit ? 0.15 : 0.06);
  });

  test('tracks a live canvas stream', async ({ page }, testInfo) => {
    test.skip(webkitLinux(testInfo), 'WebKitGTK does not deliver canvas capture frames');
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

  test('syncDisplay emits the frame image', async ({ page }, testInfo) => {
    test.skip(webkitLinux(testInfo), 'WebKitGTK does not deliver canvas capture frames');
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

  test('works without requestVideoFrameCallback', async ({ page }, testInfo) => {
    test.skip(webkitLinux(testInfo), 'WebKitGTK does not deliver canvas capture frames');
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

  for (const tight of [false, true]) {
    test(`replays two targets side by side: tracks both at once${tight ? ', on a tight tracking budget' : ''}`, async ({ page }) => {
      const t = truth('multi');
      const r = await runHarness(page, { mode: 'replay', scenario: 'multi', target: 'targets-multi.bin', ...(tight ? { trackMs: 1 } : {}) });
      expectNoErrors(r);
      expect([...new Set(r.events.foundIndex)].sort()).toEqual([0, 1]);
      // While both are fully in view, both have poses.
      const both = r.frames.filter((f) => frameTruth(f, t).targets!.every((x) => x.inView > 0.95));
      expect(both.length).toBeGreaterThan(30);
      expect(both.filter((f) => f.poses.length === 2).length / both.length).toBeGreaterThan(0.85);
      // Each target where it really is (replays can start a frame or two off: use target 0's best offset for both).
      const s = settled(r);
      let offset = 0, best = Infinity;
      for (let o = -6; o <= 6; o++) {
        const m = percentile(targetOverlayErrors(s, t, 0, o), 0.5);
        if (m < best) { best = m; offset = o; }
      }
      for (const i of [0, 1]) {
        const errors = targetOverlayErrors(s, t, i, offset);
        expect(errors.length).toBeGreaterThan(30);
        // A 1 ms budget updates the less prominent poster only every few frames (and WebKit
        // times to the millisecond): coarser, but still on the poster.
        expect(percentile(errors, 0.5)).toBeLessThan(tight ? 6 : 3);
        expect(percentile(errors, 0.95)).toBeLessThan(tight ? 20 : 12);
      }
      expect(r.stats!.budgetMs).toBeGreaterThan(0);
      if (tight) expect(r.stats!.budgetMs).toBe(1);
      for (const x of r.stats!.targets) {
        expect(x.priority).toBeGreaterThanOrEqual(0);
        expect(x.points).toBeGreaterThanOrEqual(24);
        expect(x.points).toBeLessThanOrEqual(150);
      }
    });
  }
});
