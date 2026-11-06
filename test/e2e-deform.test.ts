// End-to-end synthetic tests of bending targets: render a sheet of paper
// curling, flexing and lifting a corner, track it as a deforming surface and
// compare the recovered shape to ground truth.
import { describe, expect, it } from 'vitest';
import { K, POSTER_W_M, getPoster } from './fixtures';
import { scenarios } from '../src/sim/scene';
import { runScenario, jitter, percentile, type RunResult } from '../src/sim/harness';

const poster = getPoster();
const base = { K, poster, posterWidthM: POSTER_W_M };
const surface = { ...base, pipeline: { deformable: [0] } };

function stats(r: RunResult) {
  const vis = r.frames.filter((f) => f.visible);
  return {
    visible: vis.length / r.frames.length,
    px: percentile(vis.map((f) => f.surfaceErrPx), 0.5),
    px95: percentile(vis.map((f) => f.surfaceErrPx), 0.95),
    worst: percentile(vis.map((f) => f.surfaceMaxPx), 0.5),
    mm: percentile(vis.map((f) => f.surfaceErr3d), 0.5) * 1000,
  };
}

describe('bending targets, end to end', () => {
  it('flat sheet: as accurate and steady as rigid tracking', () => {
    const r = runScenario(scenarios.static(0.3, 2), surface);
    expect(r.firstLockTime).toBeLessThan(0.3);
    const s = stats(r);
    expect(s.visible).toBe(1);
    expect(s.px).toBeLessThan(0.1);
    expect(s.worst).toBeLessThan(0.2);
    const j = jitter(r.frames.filter((f) => f.t > 0.3));
    expect(j.transStd).toBeLessThan(0.0005);
    expect(j.rotStdDeg).toBeLessThan(0.05);
  });

  it('curl: recovers the shape that rigid tracking misses', () => {
    const scn = scenarios.curl(-6, 0, 0.35, 2);
    const rigid = stats(runScenario(scn, base));
    const s = stats(runScenario(scn, surface));
    expect(s.visible).toBe(1);
    expect(rigid.px).toBeGreaterThan(1.5);
    expect(s.px).toBeLessThan(0.3);
    expect(s.worst).toBeLessThan(1.5);
    expect(s.mm).toBeLessThan(1);
  });

  it('curl away from the camera, across a diagonal', () => {
    const s = stats(runScenario(scenarios.curl(5, Math.PI / 4, 0.35, 2), surface));
    expect(s.visible).toBe(1);
    expect(s.px).toBeLessThan(0.3);
    expect(s.worst).toBeLessThan(1.5);
  });

  it('flexing at 1 Hz: follows the bend as it changes, without re-acquiring', () => {
    const r = runScenario(scenarios.flex(-8, 1, 0.3, 0.35, 2.5), surface);
    const s = stats(r);
    expect(r.acquisitions).toBe(1);
    expect(s.visible).toBe(1);
    expect(s.px).toBeLessThan(0.15);
    expect(s.px95).toBeLessThan(0.5);
  });

  it('lifted corner, with the IMU', () => {
    const r = runScenario(scenarios.cornerLift(-12, 0.35, 2), { ...surface, useImu: true, imu: { gyroBias: [0.01, -0.02, 0.015] } });
    const s = stats(r);
    expect(s.visible).toBe(1);
    expect(s.px).toBeLessThan(0.15);
    expect(s.worst).toBeLessThan(1.2);
  });
});
