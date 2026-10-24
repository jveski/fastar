import { describe, expect, it } from 'vitest';
import { K, POSTER_W_M, getPoster, texelToMetres } from './fixtures';
import { createSceneRenderer, scenarios, relativePose } from '../src/sim/scene';
import { Pyramid, pyrDown, PyrScratch } from '../src/image/pyramid';
import { Klt } from '../src/vision/klt';
import { NccMatcher } from '../src/vision/ncc';
import { PoseSolver, projectPoint, poseToHomography } from '../src/vision/pose';
import { mat3, mat3Mul, mat3Invert, so3Exp, rotationAngle } from '../src/math/rot';
import { Rng } from '../src/math/rng';
import { invertSPD } from '../src/math/linalg';
import { createGray, blur5 } from '../src/image/gray';
import { buildKmTree, KmTreeSearcher } from '../src/vision/kmtree';

const poster = getPoster();

function renderAt(scn: ReturnType<typeof scenarios.slowSweep>, t: number, opts = {}) {
  const r = createSceneRenderer(scn, { K, poster, posterWidthM: POSTER_W_M });
  return r.render(t, scn.camera, { noise: 1, ...opts });
}

/** Well-textured target texels (simple gradient-energy pick). */
function texturedTexels(count: number, seed = 1): Array<[number, number]> {
  const rng = new Rng(seed);
  const out: Array<[number, number]> = [];
  const { width, height, data } = poster;
  while (out.length < count) {
    const u = 20 + rng.int(width - 40), v = 20 + rng.int(height - 40);
    let gx = 0, gy = 0;
    for (let k = -3; k <= 3; k++) {
      gx += Math.abs(data[(v + k) * width + u + 3] - data[(v + k) * width + u - 3]);
      gy += Math.abs(data[(v + 3) * width + u + k] - data[(v - 3) * width + u + k]);
    }
    if (gx > 200 && gy > 200) out.push([u, v]);
  }
  return out;
}

describe('renderer conventions', () => {
  it('projects target texels where the renderer draws them', () => {
    const scn = scenarios.static(0.3);
    const r = createSceneRenderer(scn, { K, poster, posterWidthM: POSTER_W_M });
    const img = r.render(0, scn.camera, {});
    const { R, t } = relativePose(scn.camera(0), scn.target(0));
    const p = new Float32Array(2);
    let err = 0, n = 0;
    for (const [u, v] of texturedTexels(200)) {
      const [X, Y] = texelToMetres(poster, POSTER_W_M, u, v);
      expect(projectPoint(K, R, t, X, Y, p)).toBe(true);
      const x = Math.round(p[0]), y = Math.round(p[1]);
      if (x < 0 || y < 0 || x >= K.width || y >= K.height) continue;
      // Compare against poster sampled at the exact inverse location of the rounded pixel.
      err += Math.abs(img.data[y * K.width + x] - poster.data[v * poster.width + u]);
      n++;
    }
    expect(n).toBeGreaterThan(150);
    expect(err / n).toBeLessThan(40);
  });

  it('homography from pose matches point projection', () => {
    const scn = scenarios.slowSweep();
    const { R, t } = relativePose(scn.camera(0.7), scn.target(0.7));
    const s = POSTER_W_M / poster.width;
    const H = poseToHomography(K, R, t, s, (poster.width - 1) / 2, (poster.height - 1) / 2, new Float64Array(9));
    const p = new Float32Array(2);
    for (const [u, v] of [[0, 0], [100, 50], [639, 479]]) {
      const [X, Y] = texelToMetres(poster, POSTER_W_M, u, v);
      projectPoint(K, R, t, X, Y, p);
      const w = H[6] * u + H[7] * v + H[8];
      expect((H[0] * u + H[1] * v + H[2]) / w).toBeCloseTo(p[0], 3);
      expect((H[3] * u + H[4] * v + H[5]) / w).toBeCloseTo(p[1], 3);
    }
  });
});

describe('pyramid', () => {
  it('SWAR blur5 matches the scalar reference', () => {
    const rng = new Rng(12);
    for (const [w, h] of [[640, 480], [12, 5], [8, 1], [16, 3]]) {
      const src = createGray(w, h);
      for (let i = 0; i < w * h; i++) src.data[i] = rng.int(256);
      const out = blur5(src, createGray(w, h));
      const k = [1, 4, 6, 4, 1];
      const px = (x: number, y: number) => src.data[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        let acc = 0;
        for (let a = 0; a < 5; a++) {
          let row = 0;
          for (let b = 0; b < 5; b++) row += k[b] * px(x + b - 2, y + a - 2);
          acc += k[a] * row;
        }
        expect(out.data[y * w + x]).toBe((acc + 128) >> 8);
      }
    }
  });

  it('matches a reference 5-tap decimating filter (odd sizes too)', () => {
    const rng = new Rng(8);
    for (const [w, h] of [[640, 480], [37, 23], [6, 6], [12, 7], [4, 4], [20, 9]]) {
      const src = createGray(w, h);
      for (let i = 0; i < w * h; i++) src.data[i] = rng.int(256);
      const dst = createGray(w >> 1, h >> 1);
      pyrDown(src, dst, new PyrScratch(w, h));
      const k = [1, 4, 6, 4, 1];
      const px = (x: number, y: number) => src.data[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
      for (let j = 0; j < dst.height; j++) {
        for (let i = 0; i < dst.width; i++) {
          let acc = 0;
          for (let a = 0; a < 5; a++) {
            let row = 0;
            for (let b = 0; b < 5; b++) row += k[b] * px(2 * i + b - 2, 2 * j + a - 2);
            acc += k[a] * row;
          }
          expect(dst.data[j * dst.width + i]).toBe((acc + 128) >> 8);
        }
      }
    }
  });

  it('halves sizes and preserves flat images', () => {
    const pyr = new Pyramid(640, 480, 4);
    expect(pyr.levels.map((l) => [l.width, l.height])).toEqual([[640, 480], [320, 240], [160, 120], [80, 60]]);
    pyr.build(new Uint8Array(640 * 480).fill(77));
    for (const l of pyr.levels) expect(l.data.every((v) => v === 77)).toBe(true);
  });
});

describe('KLT', () => {
  const scn = scenarios.slowSweep();
  const t0 = 1.0, t1 = 1.0 + 1 / 30;
  const imgA = renderAt(scn, t0), imgB = renderAt(scn, t1);
  const pa = new Pyramid(640, 480, 3).build(imgA.data);
  const pb = new Pyramid(640, 480, 3).build(imgB.data);
  const A = relativePose(scn.camera(t0), scn.target(t0));
  const B = relativePose(scn.camera(t1), scn.target(t1));
  const texels = texturedTexels(150, 4);
  const pts = new Float32Array(300), gt = new Float32Array(300);
  const tmp = new Float32Array(2);
  let n = 0;
  for (const [u, v] of texels) {
    const [X, Y] = texelToMetres(poster, POSTER_W_M, u, v);
    if (!projectPoint(K, A.R, A.t, X, Y, tmp)) continue;
    pts[2 * n] = tmp[0]; pts[2 * n + 1] = tmp[1];
    projectPoint(K, B.R, B.t, X, Y, tmp);
    gt[2 * n] = tmp[0]; gt[2 * n + 1] = tmp[1];
    n++;
  }

  it('tracks sub-pixel without a prediction', () => {
    const out = pts.slice();
    const status = new Uint8Array(n);
    const ok = new Klt().trackBatch(pa, pb, pts, out, n, status);
    expect(ok / n).toBeGreaterThan(0.8);
    const errs: number[] = [];
    for (let i = 0; i < n; i++) if (status[i]) errs.push(Math.hypot(out[2 * i] - gt[2 * i], out[2 * i + 1] - gt[2 * i + 1]));
    errs.sort((a, b) => a - b);
    expect(errs[errs.length >> 1]).toBeLessThan(0.15);
  });

  it('recovers large motion when seeded with a prediction', () => {
    // Shift frame B by 40 px: KLT alone would fail, prediction rescues it.
    const shifted = createGray(640, 480);
    for (let y = 0; y < 480; y++) for (let x = 0; x < 640; x++) shifted.data[y * 640 + x] = imgB.data[y * 640 + Math.min(639, Math.max(0, x - 40))];
    const ps = new Pyramid(640, 480, 3).build(shifted.data);
    const out = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) { out[2 * i] = gt[2 * i] + 40 + 1.5; out[2 * i + 1] = gt[2 * i + 1] - 1; }
    const status = new Uint8Array(n);
    const ok = new Klt().trackBatch(pa, ps, pts, out, n, status);
    let good = 0;
    for (let i = 0; i < n; i++) if (status[i] && Math.hypot(out[2 * i] - gt[2 * i] - 40, out[2 * i + 1] - gt[2 * i + 1]) < 0.3) good++;
    expect(good / ok).toBeGreaterThan(0.9);
    expect(ok / n).toBeGreaterThan(0.6);
  });
});

describe('NCC drift correction', () => {
  it('recovers a 1.6 px prediction error to sub-pixel accuracy', () => {
    const scn = scenarios.slowSweep();
    const t = 0.4;
    const img = renderAt(scn, t, { noise: 0.5 });
    const { R, t: tt } = relativePose(scn.camera(t), scn.target(t));
    const s = POSTER_W_M / poster.width;
    const H = poseToHomography(K, R, tt, s, (poster.width - 1) / 2, (poster.height - 1) / 2, new Float64Array(9));
    // Prediction error: translate by (1.2, -1.1) px in the frame.
    const T = Float64Array.of(1, 0, 1.2, 0, 1, -1.1, 0, 0, 1);
    const Hp = mat3Mul(T, H, mat3());
    const Hinv = new Float64Array(9);
    mat3Invert(Hp, Hinv);
    const m = new NccMatcher(4, 4);
    const errs: number[] = [];
    for (const [u, v] of texturedTexels(100, 9)) {
      const w = Hp[6] * u + Hp[7] * v + Hp[8];
      const cx = (Hp[0] * u + Hp[1] * v + Hp[2]) / w, cy = (Hp[3] * u + Hp[4] * v + Hp[5]) / w;
      if (!m.buildTemplate(poster, Hinv, cx, cy) || !m.match(img)) continue;
      const wt = H[6] * u + H[7] * v + H[8];
      errs.push(Math.hypot(m.x - (H[0] * u + H[1] * v + H[2]) / wt, m.y - (H[3] * u + H[4] * v + H[5]) / wt));
    }
    expect(errs.length).toBeGreaterThan(50);
    errs.sort((a, b) => a - b);
    expect(errs[errs.length >> 1]).toBeLessThan(0.2);
  });
});

describe('pose solver', () => {
  function problem(rng: Rng, n: number, noise: number, outlierFrac: number) {
    const Rt = mat3Mul(so3Exp(0.3, -0.2, 0.1, mat3()), so3Exp(0, 0, 0.5, mat3()), mat3());
    const tt = Float64Array.of(0.02, -0.01, 0.35);
    const obj = new Float32Array(2 * n), img = new Float32Array(2 * n), p = new Float32Array(2);
    for (let i = 0; i < n; i++) {
      const X = rng.range(-0.1, 0.1), Y = rng.range(-0.075, 0.075);
      obj[2 * i] = X; obj[2 * i + 1] = Y;
      projectPoint(K, Rt, tt, X, Y, p);
      if (rng.next() < outlierFrac) {
        img[2 * i] = p[0] + rng.range(-40, 40); img[2 * i + 1] = p[1] + rng.range(-40, 40);
      } else {
        img[2 * i] = p[0] + rng.gauss() * noise; img[2 * i + 1] = p[1] + rng.gauss() * noise;
      }
    }
    return { Rt, tt, obj, img };
  }

  it('converges from a perturbed start with outliers', () => {
    const rng = new Rng(2);
    const { Rt, tt, obj, img } = problem(rng, 150, 0.3, 0.2);
    const R = mat3Mul(so3Exp(0.06, -0.05, 0.04, mat3()), Rt, mat3());
    const t = Float64Array.of(tt[0] + 0.015, tt[1] - 0.01, tt[2] + 0.02);
    const solver = new PoseSolver();
    expect(solver.solve(K, obj, img, 150, R, t)).toBe(true);
    expect(rotationAngle(R, Rt) * 180 / Math.PI).toBeLessThan(0.15);
    expect(Math.hypot(t[0] - tt[0], t[1] - tt[1], t[2] - tt[2])).toBeLessThan(0.002);
    expect(solver.inliers).toBeGreaterThan(110);
    expect(solver.inliers).toBeLessThan(130);
  });

  it('covariance is consistent (NEES ~ 6) and depth is the noisiest axis', () => {
    const rng = new Rng(5);
    const solver = new PoseSolver();
    let nees = 0;
    const trials = 200;
    const Ci = new Float64Array(36);
    const e = new Float64Array(6);
    const dR = mat3();
    for (let k = 0; k < trials; k++) {
      const { Rt, tt, obj, img } = problem(rng, 100, 0.5, 0);
      const R = Rt.slice(), t = tt.slice();
      solver.solve(K, obj, img, 100, R, t);
      // Error in solver parametrization: R = Exp(e) Rt.
      mat3Mul(R, Float64Array.from([Rt[0], Rt[3], Rt[6], Rt[1], Rt[4], Rt[7], Rt[2], Rt[5], Rt[8]]), dR);
      e[0] = 0.5 * (dR[7] - dR[5]); e[1] = 0.5 * (dR[2] - dR[6]); e[2] = 0.5 * (dR[3] - dR[1]);
      e[3] = t[0] - tt[0]; e[4] = t[1] - tt[1]; e[5] = t[2] - tt[2];
      invertSPD(solver.cov.slice(), 6, Ci);
      for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) nees += e[i] * Ci[i * 6 + j] * e[j];
      if (k === 0) expect(solver.cov[5 * 6 + 5]).toBeGreaterThan(solver.cov[3 * 6 + 3]);
    }
    nees /= trials;
    expect(nees).toBeGreaterThan(4.5);
    expect(nees).toBeLessThan(8);
  });
});

describe('FAST', () => {
  it('SWAR pre-test finds exactly the scalar corners', async () => {
    const { FastDetector } = await import('../src/vision/fast');
    const scn = scenarios.slowSweep();
    const img = renderAt(scn, 0.5, { noise: 3 });
    const a = new FastDetector(640, 480, 30000, 2000), b = new FastDetector(640, 480, 30000, 2000);
    b.swarEnabled = false;
    for (const t of [8, 14, 30]) {
      const opts = { threshold: t, maxPoints: 2000, border: 16, cell: 32 };
      const na = a.detect(img, opts), nb = b.detect(img, opts);
      expect(na).toBe(nb);
      expect(Array.from(a.x.subarray(0, na))).toEqual(Array.from(b.x.subarray(0, nb)));
      expect(Array.from(a.y.subarray(0, na))).toEqual(Array.from(b.y.subarray(0, nb)));
    }
  });
});

describe('GL projection', () => {
  it('poseToGl + glProjection land target points on the same CSS pixel as OpenCV + object-fit', async () => {
    const { glProjection, poseToGl } = await import('../src/vision/camera');
    const R = so3Exp(0.2, -0.3, 0.1, mat3()), t = Float64Array.of(0.02, -0.01, 0.4);
    for (const [W, H, fit] of [[390, 844, 'cover'], [1280, 720, 'cover'], [800, 800, 'contain'], [640, 480, 'fill']] as const) {
      const P = glProjection(K, W, H, 0.01, 100, fit), M = poseToGl(R, t);
      for (const [X, Y] of [[0.05, 0.03], [-0.08, 0.06]]) {
        // OpenCV target (X, Y, 0) is GL target (X, -Y, 0).
        const g = [X, -Y, 0, 1];
        const e = [0, 1, 2, 3].map((r) => M[r] * g[0] + M[4 + r] * g[1] + M[8 + r] * g[2] + M[12 + r] * g[3]);
        const c = [0, 1, 3].map((r) => P[r] * e[0] + P[4 + r] * e[1] + P[8 + r] * e[2] + P[12 + r] * e[3]);
        const px = ((c[0] / c[2] + 1) / 2) * W, py = ((1 - c[1] / c[2]) / 2) * H;
        const p = new Float32Array(2);
        projectPoint(K, R, t, X, Y, p);
        const sx = fit === 'fill' ? W / K.width : fit === 'cover' ? Math.max(W / K.width, H / K.height) : Math.min(W / K.width, H / K.height);
        const sy = fit === 'fill' ? H / K.height : sx;
        expect(px).toBeCloseTo((W - K.width * sx) / 2 + (p[0] + 0.5) * sx, 3);
        expect(py).toBeCloseTo((H - K.height * sy) / 2 + (p[1] + 0.5) * sy, 3);
      }
    }
  });
});

describe('k-means tree search', () => {
  it('ratio test ignores look-alikes that are not covisible', () => {
    // Feature 1 is a near copy of feature 0 far away, feature 2 is random.
    const rng = new Rng(9);
    const desc = new Uint32Array(3 * 8);
    for (let i = 0; i < 8; i++) { desc[i] = (rng.next() * 2 ** 32) >>> 0; desc[16 + i] = (rng.next() * 2 ** 32) >>> 0; }
    desc.set(desc.subarray(0, 8), 8);
    desc[8] ^= 0b111;
    const kp = Float32Array.of(10, 10, 1, 0, 200, 50, 1, 0, 90, 90, 1, 0);
    const tree = buildKmTree(desc, 3);
    const q = desc.slice(0, 8);
    const s = new KmTreeSearcher(16);
    expect(s.search(tree, desc, kp, q, 0)).toBe(0);
    expect(s.dist2).toBe(3); // ambiguous
    expect(s.search(tree, desc, kp, q, 0, 6, (a, b) => a + b !== 1)).toBe(0);
    expect(s.dist2).toBeGreaterThan(64); // only the random one competes
  });
});
