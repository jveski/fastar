import { describe, expect, it } from 'vitest';
import { Eskf } from '../src/fusion/eskf';
import { ImuBuffer } from '../src/fusion/imu';
import { scenarios, simulateImu, relativePose } from '../src/sim/scene';
import { Rng } from '../src/math/rng';
import { mat3, mat3Mul, so3Exp, rotationAngle, vec3, so3Log, mat3MulABt } from '../src/math/rot';
import { invertSPD } from '../src/math/linalg';

const cov6 = (sr: number, st: number) => {
  const c = new Float64Array(36);
  for (let i = 0; i < 3; i++) { c[i * 7] = sr * sr; c[(i + 3) * 7] = st * st; }
  return c;
};

function feedImu(buf: ImuBuffer, _scn: ReturnType<typeof scenarios.slowSweep>, upTo: number, samples: ReturnType<typeof simulateImu>, from: { i: number }) {
  while (from.i < samples.length && samples[from.i].t <= upTo) {
    const s = samples[from.i++];
    buf.push(s.t, s.gyro[0], s.gyro[1], s.gyro[2], s.accel[0], s.accel[1], s.accel[2]);
  }
}

describe('ESKF', () => {
  const scn = scenarios.slowSweep();
  const imuSamples = simulateImu(scn, { gyroNoise: 0.002, gyroBias: [0.005, -0.004, 0.002] });
  const C = cov6(0.002, 0.0005);

  it('delayed measurements give the same result as in-order processing', () => {
    const run = (delay: number) => {
      const buf = new ImuBuffer(512);
      const f = new Eskf({ useImu: true, estimateTd: false });
      const cur = { i: 0 };
      feedImu(buf, scn, 0.0, imuSamples, cur);
      const g0 = relativePose(scn.camera(0), scn.target(0));
      f.initialize(0, g0.R, g0.t, C, null);
      for (let k = 1; k <= 30; k++) {
        const tc = k / 30;
        feedImu(buf, scn, tc + delay, imuSamples, cur);
        f.flushPending(buf);
        f.propagateTo(buf.latestTime, buf);
        const g = relativePose(scn.camera(tc), scn.target(tc));
        f.updatePose(tc, g.R, g.t, C, buf);
      }
      f.flushPending(buf, true);
      feedImu(buf, scn, 1.2, imuSamples, cur);
      f.propagateTo(1.2, buf);
      return f;
    };
    const a = run(0.011), b = run(0.12);
    for (let i = 0; i < 21; i++) expect(b.x[i]).toBeCloseTo(a.x[i], 8);
    for (let i = 0; i < 19 * 19; i += 7) expect(b.P[i]).toBeCloseTo(a.P[i], 10);
  });

  it('gates outliers and re-anchors after three rejections', () => {
    const buf = new ImuBuffer(512);
    const f = new Eskf({ useImu: true, estimateTd: false });
    const cur = { i: 0 };
    feedImu(buf, scn, 0, imuSamples, cur);
    const g0 = relativePose(scn.camera(0), scn.target(0));
    f.initialize(0, g0.R, g0.t, C, null);
    for (let k = 1; k <= 10; k++) {
      const tc = k / 30;
      feedImu(buf, scn, tc + 0.02, imuSamples, cur);
      const g = relativePose(scn.camera(tc), scn.target(tc));
      expect(f.updatePose(tc, g.R, g.t, C, buf).accepted).toBe(true);
    }
    // The target "jumps" by 10 deg (the gyro pins rotation, so this is a
    // clear outlier): rejected twice, then re-anchored on the third.
    const jump = so3Exp(0, 0, (10 * Math.PI) / 180, mat3());
    const results = [];
    for (let k = 11; k <= 13; k++) {
      const tc = k / 30;
      feedImu(buf, scn, tc + 0.02, imuSamples, cur);
      const g = relativePose(scn.camera(tc), scn.target(tc));
      results.push(f.updatePose(tc, mat3Mul(jump, g.R, mat3()), g.t, C, buf));
    }
    expect(results.map((r) => r.accepted)).toEqual([false, false, false]);
    expect(results.map((r) => r.reinitialized)).toEqual([false, false, true]);
    const R = mat3(), t = vec3();
    const tc = 13 / 30;
    f.poseAt(tc, R, t, buf);
    const g = relativePose(scn.camera(tc), scn.target(tc));
    expect(rotationAngle(R, mat3Mul(jump, g.R, mat3())) * 180 / Math.PI).toBeLessThan(0.5);
  });

  it('is consistent: mean NIS ~ measurement dimension with matched noise', () => {
    const rng = new Rng(21);
    const buf = new ImuBuffer(512);
    const f = new Eskf({ useImu: true, estimateTd: false });
    const cur = { i: 0 };
    feedImu(buf, scn, 0, imuSamples, cur);
    const g0 = relativePose(scn.camera(0), scn.target(0));
    f.initialize(0, g0.R, g0.t, C, null);
    const nis: number[] = [];
    const Rr = mat3();
    for (let k = 1; k < 120; k++) {
      const tc = k / 30;
      feedImu(buf, scn, tc + 0.02, imuSamples, cur);
      const g = relativePose(scn.camera(tc), scn.target(tc));
      // Sample measurement noise from C.
      mat3Mul(so3Exp(rng.gauss() * 0.002, rng.gauss() * 0.002, rng.gauss() * 0.002, mat3()), g.R, Rr);
      const tm = Float64Array.of(g.t[0] + rng.gauss() * 5e-4, g.t[1] + rng.gauss() * 5e-4, g.t[2] + rng.gauss() * 5e-4);
      const r = f.updatePose(tc, Rr, tm, C, buf);
      if (k > 20) nis.push(r.nis);
    }
    const mean = nis.reduce((a, b) => a + b, 0) / nis.length;
    // The constant-velocity model is deliberately conservative, so NIS sits
    // at or a bit below 6; it must not be wildly over- or under-confident.
    expect(mean).toBeGreaterThan(2);
    expect(mean).toBeLessThan(9);
  });

  it('tracks the relative pose with NEES within bounds', () => {
    const rng = new Rng(3);
    const buf = new ImuBuffer(512);
    const f = new Eskf({ useImu: true, estimateTd: false });
    const cur = { i: 0 };
    feedImu(buf, scn, 0, imuSamples, cur);
    const g0 = relativePose(scn.camera(0), scn.target(0));
    f.initialize(0, g0.R, g0.t, C, null);
    const R = mat3(), t = vec3(), Pr = new Float64Array(36), Pi = new Float64Array(36), e = new Float64Array(6);
    let nees = 0, n = 0;
    for (let k = 1; k < 120; k++) {
      const tc = k / 30;
      feedImu(buf, scn, tc + 0.02, imuSamples, cur);
      const g = relativePose(scn.camera(tc), scn.target(tc));
      const Rm = mat3Mul(so3Exp(rng.gauss() * 0.002, rng.gauss() * 0.002, rng.gauss() * 0.002, mat3()), g.R, mat3());
      const tm = Float64Array.of(g.t[0] + rng.gauss() * 5e-4, g.t[1] + rng.gauss() * 5e-4, g.t[2] + rng.gauss() * 5e-4);
      f.updatePose(tc, Rm, tm, C, buf);
      if (k < 20) continue;
      f.propagateTo(buf.latestTime, buf);
      f.poseAt(f.t - f.td, R, t, buf);
      const tn = f.t - f.td;
      const gt = relativePose(scn.camera(tn), scn.target(tn));
      const w = so3Log(mat3MulABt(R, gt.R, mat3()), vec3());
      e.set([w[0], w[1], w[2], t[0] - gt.t[0], t[1] - gt.t[1], t[2] - gt.t[2]]);
      f.relativeCovariance(Pr);
      invertSPD(Pr, 6, Pi);
      let s = 0;
      for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) s += e[i] * Pi[i * 6 + j] * e[j];
      nees += s; n++;
      expect(rotationAngle(R, gt.R)).toBeLessThan(0.01);
    }
    expect(nees / n).toBeLessThan(12);
    expect(nees / n).toBeGreaterThan(0.5);
  });

  it('re-anchoring keeps the relative position as uncertain as the measurement', () => {
    const buf = new ImuBuffer(512);
    const f = new Eskf({ useImu: true, estimateTd: false });
    const cur = { i: 0 };
    feedImu(buf, scn, 0, imuSamples, cur);
    const g0 = relativePose(scn.camera(0), scn.target(0));
    f.initialize(0, g0.R, g0.t, C, [0, -1, 0]);
    // Two seconds without vision: the camera's absolute position uncertainty grows a lot.
    feedImu(buf, scn, 2, imuSamples, cur);
    f.propagateTo(buf.latestTime, buf);
    const g = relativePose(scn.camera(2), scn.target(2));
    const Cm = cov6(0.003, 0.001);
    f.reanchorTarget(g.R, g.t, Cm);
    const rel = f.relativeCovariance(new Float64Array(36));
    // Position block exact; rotation only gets the small attitude slack.
    for (let i = 3; i < 6; i++) for (let j = 3; j < 6; j++) expect(rel[i * 6 + j]).toBeCloseTo(Cm[i * 6 + j], 8);
    for (let i = 0; i < 3; i++) expect(Math.sqrt(rel[i * 7])).toBeLessThan(0.05);
  });
});
