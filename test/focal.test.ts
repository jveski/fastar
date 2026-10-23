import { describe, expect, it } from 'vitest';
import { focalFromHomography, FocalCalibrator } from '../src/vision/focal';
import { poseToHomography } from '../src/vision/pose';
import { intrinsicsFromFocal } from '../src/vision/camera';
import { Rng } from '../src/math/rng';
import { mat3, so3Exp } from '../src/math/rot';
import { createSceneRenderer, lookAt, pose, WALL_R, type Scenario } from '../src/sim/scene';
import { targetFor } from '../src/sim/harness';
import { Acquirer } from '../src/track/acquire';
import { getPoster, POSTER_W_M } from './fixtures';

describe('focal self-calibration', () => {
  it('recovers f exactly from noise-free tilted homographies', () => {
    const K = intrinsicsFromFocal(640, 480, 560);
    const rng = new Rng(2);
    for (let i = 0; i < 10; i++) {
      const R = so3Exp(rng.range(-0.7, 0.7), rng.range(-0.7, 0.7), rng.range(-3, 3), mat3());
      const t = Float64Array.of(rng.range(-0.05, 0.05), rng.range(-0.05, 0.05), rng.range(0.25, 0.6));
      const H = poseToHomography(K, R, t, 0.0003, 320, 240, new Float64Array(9));
      const f = focalFromHomography(H, K.cx, K.cy);
      if (Number.isFinite(f)) expect(f).toBeCloseTo(560, 3);
    }
  });

  it('estimates f within 3% from acquisition homographies of rendered views (true f differs from the FOV guess)', () => {
    const poster = getPoster();
    const target = targetFor(poster, POSTER_W_M);
    const Ktrue = intrinsicsFromFocal(640, 480, 600);
    const Kguess = intrinsicsFromFocal(640, 480, 520);
    const acq = new Acquirer([target], 640, 480, Kguess);
    const cal = new FocalCalibrator(640, 480);
    const eyes = [[0.25, -0.25, 0.05], [-0.22, -0.28, 0.0], [0.0, -0.25, 0.22], [0.05, -0.27, -0.2], [0.2, -0.22, 0.15], [-0.18, -0.25, -0.15], [0.28, -0.3, -0.05]];
    for (const eye of eyes) {
      const cam = pose(lookAt(eye, [0, 0, 0]), eye);
      const scn: Scenario = { name: 'f', duration: 1, camera: () => cam, target: () => pose(WALL_R, [0, 0, 0]) };
      const img = createSceneRenderer(scn, { K: Ktrue, poster, posterWidthM: POSTER_W_M }).render(0, scn.camera, { noise: 1 });
      const r = acq.acquire(img.data);
      if (r) cal.addHomography(r.H);
    }
    expect(cal.samples.length).toBeGreaterThanOrEqual(5);
    const est = cal.estimate();
    expect(est).not.toBeNull();
    expect(Math.abs(est!.f - 600) / 600).toBeLessThan(0.03);
  });
});
