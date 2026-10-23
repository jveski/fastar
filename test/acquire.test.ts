import { describe, expect, it } from 'vitest';
import { K, POSTER_W_M, getPoster } from './fixtures';
import { compileTarget } from '../src/target/compiler';
import { encodeTargets, decodeTargets } from '../src/target/format';
import { ippe } from '../src/vision/ippe';
import { homographyFrom4, homographyDLT, applyH, quadConsistent } from '../src/vision/homography';
import { Acquirer, chooseSolution } from '../src/track/acquire';
import { createSceneRenderer, lookAt, pose, relativePose, WALL_R, type Scenario } from '../src/sim/scene';
import { mat3, mat3Mul, so3Exp, rotationAngle } from '../src/math/rot';
import { Rng } from '../src/math/rng';
import { projectPoint } from '../src/vision/pose';

const poster = getPoster();
const target = compileTarget(poster, { physicalWidth: POSTER_W_M, name: 'poster' });

describe('target compiler + format', () => {
  it('produces a healthy target', () => {
    const q = target.meta.quality;
    expect(q.features).toBeGreaterThan(1000);
    expect(q.coverage).toBeGreaterThan(0.8);
    expect(q.trackingPoints).toBeGreaterThan(150);
    expect(q.warnings).toEqual([]);
    expect(target.meta.levelScales.length).toBeGreaterThanOrEqual(8);
    expect(target.meta.images.map((i) => i.width)).toEqual([256, 512, 640]);
  });

  it('round-trips through the binary format without copies', () => {
    const buf = encodeTargets([target, target]);
    const [a, b] = decodeTargets(buf);
    expect(a.meta).toEqual(target.meta);
    expect(b.kp).toEqual(target.kp);
    expect(a.desc.buffer).toBe(buf);
    expect(Array.from(a.tree)).toEqual(Array.from(target.tree));
    expect(Array.from(a.trackPoints)).toEqual(Array.from(target.trackPoints));
    expect(a.images[2]).toEqual(target.images[2]);
  });

  it('warns on a flat image', () => {
    const flat = { width: 300, height: 300, data: new Uint8Array(90000).fill(128) };
    const t = compileTarget(flat, { physicalWidth: 0.1 });
    expect(t.meta.quality.warnings.length).toBeGreaterThan(0);
  });
});

describe('geometry', () => {
  it('homography from 4 points and DLT agree with ground truth', () => {
    const H = Float64Array.of(1.2, 0.1, 30, -0.05, 0.9, 12, 1e-4, -2e-4, 1);
    const rng = new Rng(4);
    const src = new Float64Array(40), dst = new Float64Array(40);
    for (let i = 0; i < 20; i++) {
      src[2 * i] = rng.range(0, 500); src[2 * i + 1] = rng.range(0, 400);
      applyH(H, src[2 * i], src[2 * i + 1], dst, 2 * i);
    }
    const H4 = new Float64Array(9), Hd = new Float64Array(9);
    expect(homographyFrom4(src, dst, H4)).toBe(true);
    expect(homographyDLT(src, dst, 20, Hd)).toBe(true);
    for (let i = 0; i < 9; i++) {
      expect(H4[i]).toBeCloseTo(H[i], 6);
      expect(Hd[i]).toBeCloseTo(H[i], 6);
    }
  });

  it('rejects mirrored quads', () => {
    const s = [0, 0, 1, 0, 1, 1, 0, 1];
    expect(quadConsistent(s, [0, 0, 2, 0, 2, 2, 0, 2])).toBe(true);
    expect(quadConsistent(s, [0, 0, -2, 0, -2, 2, 0, 2])).toBe(false);
  });

  it('IPPE recovers the true pose among its two solutions', () => {
    const rng = new Rng(9);
    for (let trial = 0; trial < 20; trial++) {
      const R = so3Exp(rng.range(-0.8, 0.8), rng.range(-0.8, 0.8), rng.range(-3, 3), mat3());
      const t = Float64Array.of(rng.range(-0.05, 0.05), rng.range(-0.05, 0.05), rng.range(0.2, 0.8));
      const n = 30, obj = new Float64Array(2 * n), img = new Float64Array(2 * n);
      for (let i = 0; i < n; i++) {
        const X = rng.range(-0.1, 0.1), Y = rng.range(-0.08, 0.08);
        obj[2 * i] = X; obj[2 * i + 1] = Y;
        const z = R[6] * X + R[7] * Y + t[2];
        img[2 * i] = (R[0] * X + R[1] * Y + t[0]) / z;
        img[2 * i + 1] = (R[3] * X + R[4] * Y + t[1]) / z;
      }
      const sols = ippe(obj, img, n);
      expect(sols.length).toBe(2);
      expect(sols[0].err).toBeLessThan(1e-6);
      expect(rotationAngle(sols[0].R, R)).toBeLessThan(1e-4);
      expect(Math.hypot(sols[0].t[0] - t[0], sols[0].t[1] - t[1], sols[0].t[2] - t[2])).toBeLessThan(1e-5);
    }
  });

  it('gravity prior breaks IPPE ambiguity toward wall/table planes', () => {
    const wall = { R: Float64Array.of(1, 0, 0, 0, 0, -1, 0, 1, 0), t: new Float64Array(3), err: 1e-3 };
    const tilted = { R: so3Exp(0.5, 0, 0, mat3()), t: new Float64Array(3), err: 0.9e-3 };
    // Camera y is down, so "up" is -y in camera coordinates.
    expect(chooseSolution([tilted, wall], [0, -1, 0])).toBe(wall);
    expect(chooseSolution([tilted, wall], null)).toBe(tilted);
  });
});

describe('acquisition', () => {
  const acq = new Acquirer([target], 640, 480, K);
  const cases: Array<{ name: string; eye: number[]; roll?: number; at?: number[] }> = [
    { name: 'frontal 30cm', eye: [0, -0.3, 0] },
    { name: 'far 70cm', eye: [0.05, -0.7, 0.02] },
    { name: 'tilted 50deg', eye: [0.3, -0.25, 0.05] },
    { name: 'rolled 90deg', eye: [0, -0.35, 0], roll: Math.PI / 2 },
    { name: 'off-centre close', eye: [0.04, -0.2, -0.03], at: [0.05, 0, -0.02] },
  ];
  for (const c of cases) {
    it(`finds the target: ${c.name}`, () => {
      let R = lookAt(c.eye, c.at ?? [0, 0, 0]);
      if (c.roll) R = mat3Mul(R, so3Exp(0, 0, c.roll, mat3()), mat3());
      const cam = pose(R, c.eye);
      const scn: Scenario = { name: c.name, duration: 1, camera: () => cam, target: () => pose(WALL_R, [0, 0, 0]) };
      const img = createSceneRenderer(scn, { K, poster, posterWidthM: POSTER_W_M }).render(0, scn.camera, { noise: 2, blurPasses: 1 });
      const res = acq.acquire(img.data, { up: [0, -Math.cos(c.roll ?? 0), Math.sin(c.roll ?? 0)] });
      expect(res).not.toBeNull();
      const gt = relativePose(cam, scn.target(0));
      const dist = Math.hypot(gt.t[0], gt.t[1], gt.t[2]);
      expect(rotationAngle(res!.R, gt.R) * 180 / Math.PI).toBeLessThan(2.5);
      expect(Math.hypot(res!.t[0] - gt.t[0], res!.t[1] - gt.t[1], res!.t[2] - gt.t[2]) / dist).toBeLessThan(0.02);
      expect(res!.inliers).toBeGreaterThanOrEqual(12);
      // Homography agrees with the pose.
      const p = new Float32Array(2), q = new Float64Array(2);
      projectPoint(K, gt.R, gt.t, 0, 0, p);
      applyH(res!.H, (poster.width - 1) / 2, (poster.height - 1) / 2, q);
      expect(Math.hypot(p[0] - q[0], p[1] - q[1])).toBeLessThan(3);
    });
  }

  it('rejects frames without the target', () => {
    const cam = pose(lookAt([0, -0.3, 0], [0, 0, 0]), [0, -0.3, 0]);
    const scn: Scenario = { name: 'empty', duration: 1, camera: () => cam, target: () => pose(WALL_R, [5, 0, 0]) };
    const img = createSceneRenderer(scn, { K, poster, posterWidthM: POSTER_W_M }).render(0, scn.camera, { noise: 2 });
    expect(acq.acquire(img.data)).toBeNull();
  });
});
