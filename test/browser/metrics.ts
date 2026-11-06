// Assertion helpers shared by the Playwright specs.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Page } from '@playwright/test';
import { projectPoint } from '../../src/vision/pose';
import { SURFACE_GRID, type FrameSample, type HarnessResult, type Truth } from './types';

export const FIXTURES = join(process.cwd(), 'test/browser/fixtures/data');

export function truth(name: string): Truth {
  return JSON.parse(readFileSync(join(FIXTURES, `${name}.truth.json`), 'utf8')) as Truth;
}

export async function runHarness(page: Page, params: Record<string, string | number | boolean>, base = ''): Promise<HarnessResult> {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) q.set(k, String(v));
  await page.goto(`${base}/?${q.toString()}`);
  await page.waitForFunction(() => typeof window.__fastarTest !== 'undefined' && !window.__fastarTest.done);
  await page.locator('#start').click();
  await page.waitForFunction(() => window.__fastarTest.done, undefined, { timeout: 75_000 });
  return page.evaluate(() => window.__fastarTest);
}

function truthIndex(mediaTime: number, t: Truth, offsetFrames = 0): number {
  const n = t.frames.length;
  const frame = mediaTime * t.fps + offsetFrames;
  const wrapped = ((frame % n) + n) % n;
  return Math.round(wrapped) % n;
}

export function frameTruth(f: FrameSample, t: Truth, offsetFrames = 0): Truth['frames'][number] {
  return t.frames[truthIndex(f.mediaTime, t, offsetFrames)];
}

/** Per-frame mean corner error in processing pixels. */
export function overlayErrors(r: HarnessResult, t: Truth, offsetFrames = 0): number[] {
  const hw = t.posterWidthM / 2;
  const hh = (t.posterWidthM * t.targetHeight) / t.targetWidth / 2;
  const corners = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
  const pe = new Float64Array(2), pg = new Float64Array(2);
  const out: number[] = [];
  for (const f of r.frames) {
    const gt = frameTruth(f, t, offsetFrames);
    if (!gt?.visible || !f.poses.length) continue;
    for (const pose of f.poses) {
      // Several targets: each pose against its own target's truth, while that is in view.
      const g = gt.targets ? gt.targets[pose.index] : gt;
      if (!g || ('inView' in g && g.inView < 0.95)) continue;
      let e = 0, n = 0;
      for (const [X, Y] of corners) {
        if (!projectPoint(t.intrinsics, pose.R, pose.t, X, Y, pe)) continue;
        if (!projectPoint(t.intrinsics, g.R, g.tv, X, Y, pg)) continue;
        e += Math.hypot(pe[0] - pg[0], pe[1] - pg[1]);
        n++;
      }
      if (n) out.push(e / n);
    }
  }
  return out;
}

/** overlayErrors of one target only. */
export function targetOverlayErrors(r: HarnessResult, t: Truth, index: number, offsetFrames = 0): number[] {
  return overlayErrors({ ...r, frames: r.frames.map((f) => ({ ...f, poses: f.poses.filter((p) => p.index === index) })) }, t, offsetFrames);
}

/**
 * Bending targets: per-frame mean image distance (processing pixels) between
 * the reported surface at SURFACE_GRID (through the pose) and the true sheet.
 */
export function surfaceErrors(r: HarnessResult, t: Truth, offsetFrames = 0): number[] {
  const K = t.intrinsics, out: number[] = [];
  for (const f of r.frames) {
    const gt = frameTruth(f, t, offsetFrames);
    if (!gt?.visible || !gt.surface) continue;
    for (const pose of f.poses) {
      if (!pose.surface) continue;
      const R = pose.R, tv = pose.t, s = pose.surface, g = gt.surface;
      let e = 0;
      for (let k = 0; k < SURFACE_GRID.length; k++) {
        // App axes (y up, z out of the target) to OpenCV target axes, then into the camera.
        const x = s[3 * k], y = -s[3 * k + 1], z = -s[3 * k + 2];
        const cx = R[0] * x + R[1] * y + R[2] * z + tv[0], cy = R[3] * x + R[4] * y + R[5] * z + tv[1], cz = R[6] * x + R[7] * y + R[8] * z + tv[2];
        const gx = g[3 * k], gy = g[3 * k + 1], gz = g[3 * k + 2];
        e += Math.hypot(K.fx * (cx / cz - gx / gz), K.fy * (cy / cz - gy / gz));
      }
      out.push(e / SURFACE_GRID.length);
    }
  }
  return out;
}

/** surfaceErrors at the frame offset with the best median (replays can start a frame or two off). */
export function bestSurfaceErrors(r: HarnessResult, t: Truth, range = 6): number[] {
  let best = surfaceErrors(r, t, 0), bestMedian = percentile(best, 0.5);
  for (let o = -range; o <= range; o++) {
    const e = surfaceErrors(r, t, o), m = percentile(e, 0.5);
    if (m < bestMedian) { best = e; bestMedian = m; }
  }
  return best;
}

export function percentile(xs: number[], p: number): number {
  const a = xs.filter(Number.isFinite).sort((x, y) => x - y);
  return a.length ? a[Math.min(a.length - 1, Math.floor(p * a.length))] : NaN;
}

/** Choose the frame offset with the best median error (live streams have a start offset). */
export function bestOffset(r: HarnessResult, t: Truth, range = 6): { offset: number; errors: number[] } {
  let best = { offset: 0, errors: overlayErrors(r, t, 0) };
  let bestMedian = percentile(best.errors, 0.5);
  for (let o = -range; o <= range; o++) {
    if (o === 0) continue;
    const errors = overlayErrors(r, t, o);
    const m = percentile(errors, 0.5);
    if (m < bestMedian) { best = { offset: o, errors }; bestMedian = m; }
  }
  return best;
}

/** Drop acquisition frames: accuracy is asserted after the filter settles. */
export function settled(r: HarnessResult, frames = 15): HarnessResult {
  const first = r.frames.findIndex((f) => f.poses.length);
  return first < 0 ? r : { ...r, frames: r.frames.slice(first + frames) };
}

export function firstLockMs(r: HarnessResult): number {
  const firstFrame = r.frames[0]?.captureTime;
  const firstPose = r.frames.find((f) => f.poses.length)?.captureTime;
  return firstFrame !== undefined && firstPose !== undefined ? firstPose - firstFrame : Infinity;
}

/** Fraction of frames after the first pose that still have a pose. */
export function visibleRatio(r: HarnessResult): number {
  const first = r.frames.findIndex((f) => f.poses.length);
  if (first < 0) return 0;
  const after = r.frames.slice(first);
  return after.length ? after.filter((f) => f.poses.length).length / after.length : 0;
}

export function staticJitterMm(r: HarnessResult): number {
  const ts = r.frames.filter((f) => f.poses.length).map((f) => f.poses[0].t);
  if (ts.length < 2) return NaN;
  const mean = [0, 1, 2].map((k) => ts.reduce((s, t) => s + t[k], 0) / ts.length);
  const ss = ts.reduce((s, t) => s + [0, 1, 2].reduce((a, k) => a + (t[k] - mean[k]) ** 2, 0), 0);
  return Math.sqrt(ss / (ts.length - 1)) * 1000;
}

export function videoTimeErrors(r: HarnessResult, t: Truth): number[] {
  const D = t.video?.duration;
  if (!D) return [];
  const out: number[] = [];
  for (const f of r.frames) {
    const gt = frameTruth(f, t);
    for (const pose of f.poses) {
      if (pose.videoTime == null || gt.videoTime == null) continue;
      let d = (((pose.videoTime - gt.videoTime) % D) + D) % D;
      if (d > D / 2) d -= D;
      out.push(Math.abs(d));
    }
  }
  return out;
}

export function videoLockedRatio(r: HarnessResult): number {
  const poses = r.frames.flatMap((f) => f.poses);
  return poses.length ? poses.filter((p) => p.locked).length / poses.length : 0;
}

export function expectNoErrors(r: HarnessResult): void {
  if (r.error) throw new Error(`harness error: ${r.error}`);
}
