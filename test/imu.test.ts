import { describe, expect, it } from 'vitest';
import { TdEstimator } from '../src/fusion/td';
import { ImuBuffer } from '../src/fusion/imu';
import { scenarios, simulateImu, relativePose } from '../src/sim/scene';
import { Rng } from '../src/math/rng';
import { mat3, mat3Mul, so3Exp } from '../src/math/rot';
import { K, POSTER_W_M, getPoster } from './fixtures';
import { runScenario, percentile } from '../src/sim/harness';

/** Feed ground-truth (slightly noisy) vision rotations with a known camera-clock offset. */
function estimateWith(td: number, signs: [number, number, number] = [1, 1, 1]) {
  const scn = scenarios.calibrateThenShake(0.35, 3, 0);
  const imu = simulateImu(scn, { gyroBias: [0.01, -0.01, 0.005], gyroNoise: 0.01 });
  const buf = new ImuBuffer(1024);
  for (const s of imu) buf.push(s.t, s.gyro[0] * signs[0], s.gyro[1] * signs[1], s.gyro[2] * signs[2]);
  const est = new TdEstimator();
  const rng = new Rng(4);
  const n = mat3();
  for (let k = 0; k < 3 * 30; k++) {
    const t = k / 30;
    const g = relativePose(scn.camera(t), scn.target(t));
    mat3Mul(so3Exp(rng.gauss() * 1e-3, rng.gauss() * 1e-3, rng.gauss() * 1e-3, n), g.R, n);
    est.addVision(t - td, n);
  }
  return est.estimate(buf);
}

describe('t_d estimation', () => {
  for (const td of [-0.08, -0.02, 0, 0.035, 0.12]) {
    it(`recovers t_d = ${td * 1000} ms within 2 ms`, () => {
      const r = estimateWith(td);
      expect(r).not.toBeNull();
      expect(Math.abs(r!.td - td)).toBeLessThan(0.002);
      expect(r!.axisSigns).toEqual([1, 1, 1]);
    });
  }

  it('detects flipped gyro axes', () => {
    const r = estimateWith(0.02, [1, -1, -1]);
    expect(r).not.toBeNull();
    expect(r!.axisSigns).toEqual([1, -1, -1]);
    expect(Math.abs(r!.td - 0.02)).toBeLessThan(0.002);
  });

  it('waits for enough motion', () => {
    const est = new TdEstimator();
    const scn = scenarios.static();
    const buf = new ImuBuffer(1024);
    for (const s of simulateImu(scn)) buf.push(s.t, s.gyro[0], s.gyro[1], s.gyro[2]);
    for (let k = 0; k < 90; k++) est.addVision(k / 30, relativePose(scn.camera(k / 30), scn.target(0)).R);
    expect(est.estimate(buf)).toBeNull();
  });
});

const base = { K, poster: getPoster(), posterWidthM: POSTER_W_M, useImu: true };

describe('end-to-end with IMU', () => {
  it('calibrates t_d, then survives a 300 deg/s shake without re-acquisition', () => {
    const r = runScenario(scenarios.calibrateThenShake(), { ...base, td: 0.03, imu: { gyroBias: [0.01, -0.02, 0.015] } });
    expect(r.firstLockTime).toBeLessThan(0.3);
    expect(r.pipeline.tdCalibrated).toBe(true);
    expect(Math.abs(r.pipeline.slot(0)!.eskf.td - 0.03)).toBeLessThan(0.002);
    const shake = r.frames.filter((f) => f.t >= 2.5);
    expect(shake.every((f) => f.visible)).toBe(true);
    expect(shake.some((f) => f.acquired)).toBe(false);
    expect(percentile(shake.map((f) => f.overlayErrPx), 0.95)).toBeLessThan(2);
    // Before calibration the overlay must still be usable.
    expect(percentile(r.frames.filter((f) => f.t < 2.5).map((f) => f.overlayErrPx), 0.95)).toBeLessThan(2);
  });

  it('holds the overlay through a 0.6 s occlusion on the gyro and re-locks without acquisition', () => {
    const scn = scenarios.occlusion(0.35, 3, 1.2, 0.6);
    const r = runScenario(scn, { ...base, td: 0, imu: { gyroBias: [0.01, -0.02, 0.015] } });
    const occ = r.frames.filter((f) => f.t >= 1.2 && f.t < 1.8);
    expect(occ.every((f) => f.visible)).toBe(true);
    expect(occ.every((f) => f.status === 'coasting')).toBe(true);
    expect(Math.max(...occ.map((f) => f.rotErrDeg))).toBeLessThan(0.3);
    // First 0.5 s of coasting stays within a few percent of the image width.
    expect(Math.max(...occ.filter((f) => f.t < 1.7).map((f) => f.overlayErrPx))).toBeLessThan(20);
    expect(r.acquisitions).toBe(1);
    const after = r.frames.filter((f) => f.t >= 1.85);
    expect(after.every((f) => f.status === 'tracking')).toBe(true);
    expect(percentile(after.map((f) => f.overlayErrPx), 0.95)).toBeLessThan(0.5);
  });

  it('detects a moving target, follows it, and returns to static mode when it stops', () => {
    let movingSeen = false;
    const r = runScenario(scenarios.movingTarget(0.4, 4.5), {
      ...base, td: 0, onFrame: (p) => { if (p.slot(0)?.eskf.moving) movingSeen = true; },
    });
    expect(movingSeen).toBe(true);
    expect(r.acquisitions).toBe(1);
    expect(r.frames.every((f) => f.visible)).toBe(true);
    expect(percentile(r.frames.map((f) => f.overlayErrPx), 0.95)).toBeLessThan(0.5);
    expect(Math.max(...r.frames.map((f) => f.overlayErrPx))).toBeLessThan(4);
    expect(r.pipeline.slot(0)!.eskf.moving).toBe(false);
  });

  it('static with IMU keeps jitter within target', async () => {
    const { jitter } = await import('../src/sim/harness');
    const r = runScenario(scenarios.static(0.3, 2), { ...base, td: 0.02 });
    const j = jitter(r.frames.filter((f) => f.t > 0.3));
    expect(j.transStd).toBeLessThan(0.0005);
    expect(j.rotStdDeg).toBeLessThan(0.05);
  });
});
