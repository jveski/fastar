// End-to-end synthetic tests (vision-only filter): render scripted
// trajectories, run the full pipeline with acquisition, compare to ground truth.
import { describe, expect, it } from 'vitest';
import { K, POSTER_W_M, getPoster } from './fixtures';
import { scenarios, relativePose } from '../src/sim/scene';
import { runScenario, jitter, percentile } from '../src/sim/harness';
import { projectPoint } from '../src/vision/pose';

const poster = getPoster();
const base = { K, poster, posterWidthM: POSTER_W_M };

describe('end-to-end, vision only', () => {
  it('static at 30 cm: jitter < 0.5 mm and < 0.05 deg; first pose < 300 ms', () => {
    const r = runScenario(scenarios.static(0.3, 2), base);
    expect(r.firstLockTime).toBeLessThan(0.3);
    expect(r.frames.every((f) => f.visible)).toBe(true);
    const j = jitter(r.frames.filter((f) => f.t > 0.3));
    expect(j.transStd).toBeLessThan(0.0005);
    expect(j.rotStdDeg).toBeLessThan(0.05);
  });

  it('slow sweep: overlay matches the presented frame (0 frames of lag)', () => {
    const r = runScenario(scenarios.slowSweep(), base);
    expect(r.acquisitions).toBe(1);
    expect(r.frames.every((f) => f.visible)).toBe(true);
    const err = r.frames.map((f) => f.overlayErrPx);
    expect(percentile(err, 0.5)).toBeLessThan(0.1);
    expect(percentile(err, 0.95)).toBeLessThan(0.5);
    // One frame of lag would show up as the inter-frame image motion.
    const scn = scenarios.slowSweep();
    const p0 = new Float32Array(2), p1 = new Float32Array(2);
    const motion: number[] = [];
    for (let k = 1; k < r.frames.length; k++) {
      const a = relativePose(scn.camera((k - 1) / 30), scn.target(0)), b = relativePose(scn.camera(k / 30), scn.target(0));
      projectPoint(K, a.R, a.t, 0.1, 0.075, p0);
      projectPoint(K, b.R, b.t, 0.1, 0.075, p1);
      motion.push(Math.hypot(p1[0] - p0[0], p1[1] - p0[1]));
    }
    expect(percentile(err, 0.5)).toBeLessThan(0.1 * percentile(motion, 0.5));
  });

  it('moving target: keeps tracking without re-acquisition', () => {
    const r = runScenario(scenarios.movingTarget(), base);
    expect(r.acquisitions).toBe(1);
    expect(r.frames.every((f) => f.visible)).toBe(true);
    expect(percentile(r.frames.map((f) => f.overlayErrPx), 0.95)).toBeLessThan(1);
  });
});
