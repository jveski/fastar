import { describe, expect, it } from 'vitest';
import { bandCholesky, bandSolve, solveSPD } from '../src/math/linalg';
import { absoluteOrientation, mat3, mat3Mul, so3Exp, rotationAngle, vec3 } from '../src/math/rot';
import { Rng } from '../src/math/rng';
import { sampleGrid } from '../src/deform/grid';
import { SurfaceMesh } from '../src/deform/mesh';
import { SurfaceSolver } from '../src/deform/solver';
import { SheetShape, bends } from '../src/sim/bend';
import { K } from './fixtures';

/** A tilted view of a 20 x 15 cm sheet at 35 cm. */
function view(): { R: Float64Array; t: Float64Array } {
  return { R: mat3Mul(so3Exp(0.35, 0, 0, mat3()), so3Exp(0, -0.25, 0.1, mat3()), mat3()), t: vec3(0.01, -0.005, 0.35) };
}

function toCam(R: Float64Array, t: Float64Array, p: ArrayLike<number>, out: Float64Array, o = 0): void {
  out[o] = R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0];
  out[o + 1] = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1];
  out[o + 2] = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2];
}

describe('banded Cholesky', () => {
  it('matches the dense solve', () => {
    const rng = new Rng(3);
    const n = 40, b = 7;
    const dense = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = Math.max(0, i - b); j < i; j++) {
        const v = rng.range(-1, 1);
        dense[i * n + j] = dense[j * n + i] = v;
      }
      dense[i * n + i] = 2 * b + rng.range(0, 1);
    }
    const band = new Float64Array(n * (b + 1));
    for (let i = 0; i < n; i++) for (let j = Math.max(0, i - b); j <= i; j++) band[i * (b + 1) + j - i + b] = dense[i * n + j];
    const rhs = Float64Array.from({ length: n }, () => rng.range(-1, 1));
    const x1 = rhs.slice(), x2 = rhs.slice();
    expect(solveSPD(dense.slice(), n, x1)).toBe(true);
    expect(bandCholesky(band, n, b)).toBe(true);
    bandSolve(band, n, b, x2);
    for (let i = 0; i < n; i++) expect(x2[i]).toBeCloseTo(x1[i], 10);
  });
});

describe('absolute orientation', () => {
  it('recovers a rigid motion', () => {
    const rng = new Rng(5);
    const R0 = so3Exp(0.4, -0.7, 1.1, mat3()), t0 = vec3(0.1, -0.2, 0.5);
    const n = 20, a = new Float64Array(3 * n), b = new Float64Array(3 * n);
    for (let i = 0; i < 3 * n; i++) a[i] = rng.range(-0.1, 0.1);
    for (let i = 0; i < n; i++) toCam(R0, t0, a.subarray(3 * i), b, 3 * i);
    const R = mat3(), t = vec3();
    expect(absoluteOrientation(a, b, n, null, R, t)).toBe(true);
    expect(rotationAngle(R, R0)).toBeLessThan(1e-9);
    for (let k = 0; k < 3; k++) expect(t[k]).toBeCloseTo(t0[k], 9);
  });
});

describe('surface mesh', () => {
  const mesh = new SurfaceMesh(0.2, 0.15, 9, 7);

  it('binds points with barycentric weights that reproduce them', () => {
    const idx = new Int32Array(3), w = new Float64Array(3);
    for (const [X, Y] of [[0, 0], [0.031, -0.044], [-0.099, 0.074], [0.1, 0.075], [0.06, 0.01]]) {
      mesh.bind(X, Y, idx, w, 0);
      let x = 0, y = 0, s = 0;
      for (let k = 0; k < 3; k++) {
        expect(w[k]).toBeGreaterThanOrEqual(-1e-9);
        x += w[k] * mesh.rest[3 * idx[k]];
        y += w[k] * mesh.rest[3 * idx[k] + 1];
        s += w[k];
      }
      expect(s).toBeCloseTo(1, 12);
      expect(x).toBeCloseTo(X, 12);
      expect(y).toBeCloseTo(Y, 12);
    }
  });

  it('samples a smooth surface through the vertices, exact on planes', () => {
    const p = new Float64Array(3), du = new Float64Array(3), dv = new Float64Array(3);
    sampleGrid(mesh.rest, 9, 7, 3, 2, p);
    expect(p[0]).toBeCloseTo(mesh.rest[3 * (2 * 9 + 3)], 12);
    // Between vertices and past the border, a flat grid stays flat and linear.
    for (const [gx, gy] of [[0.3, 0.7], [7.6, 5.9], [4.5, 3.25]]) {
      sampleGrid(mesh.rest, 9, 7, gx, gy, p, du, dv);
      expect(p[0]).toBeCloseTo(-0.1 + gx * mesh.cw, 12);
      expect(p[1]).toBeCloseTo(-0.075 + gy * mesh.ch, 12);
      expect(du[0]).toBeCloseTo(mesh.cw, 12);
      expect(dv[1]).toBeCloseTo(mesh.ch, 12);
    }
  });
});

describe('surface solver', () => {
  const mesh = new SurfaceMesh(0.2, 0.15, 9, 7);

  /** Tracked points of a bent sheet seen from `view()`, solved over several frames from flat. */
  function recover(shape: SheetShape | null, frames = 8, noise = 0.3, outliers = 0, anneal = false) {
    const rng = new Rng(11);
    const { R, t } = view();
    const n = 250;
    const idx = new Int32Array(3 * n), w = new Float32Array(3 * n), img = new Float32Array(2 * n);
    const p = new Float64Array(3), c = new Float64Array(3);
    for (let k = 0; k < n; k++) {
      const X = rng.range(-0.095, 0.095), Y = rng.range(-0.07, 0.07);
      mesh.bind(X, Y, idx, w, 3 * k);
      if (shape) shape.point(X, Y, p);
      else { p[0] = X; p[1] = Y; p[2] = 0; }
      toCam(R, t, p, c);
      img[2 * k] = K.fx * c[0] / c[2] + K.cx + rng.gauss() * noise;
      img[2 * k + 1] = K.fy * c[1] / c[2] + K.cy + rng.gauss() * noise;
      if (k < outliers) { img[2 * k] += rng.range(-40, 40); img[2 * k + 1] += rng.range(-40, 40); }
    }
    const V = new Float64Array(3 * mesh.count);
    for (let v = 0; v < mesh.count; v++) toCam(R, t, mesh.rest.subarray(3 * v), V, 3 * v);
    const solver = new SurfaceSolver(mesh, n);
    const nrm = [R[2], R[5], R[8]];
    const sigmas = anneal ? [32, 8, 2] : [];
    for (let f = 0; f < frames; f++) expect(solver.solve(K, V, V.slice(), nrm, idx, w, img, n, sigmas[f])).toBe(true);
    // Vertex error against the true sheet, camera frame.
    let err = 0, worst = 0;
    for (let v = 0; v < mesh.count; v++) {
      const X = mesh.rest[3 * v], Y = mesh.rest[3 * v + 1];
      if (shape) shape.point(X, Y, p);
      else { p[0] = X; p[1] = Y; p[2] = 0; }
      toCam(R, t, p, c);
      const e = Math.hypot(V[3 * v] - c[0], V[3 * v + 1] - c[1], V[3 * v + 2] - c[2]);
      err += e / mesh.count;
      worst = Math.max(worst, e);
    }
    return { err, worst, solver };
  }

  it('keeps a flat sheet flat', () => {
    const r = recover(null);
    expect(r.err).toBeLessThan(0.0008);
    expect(r.solver.rmsInlier).toBeLessThan(0.5);
  });

  it('recovers a curl from flat in a few frames', () => {
    const shape = new SheetShape(bends.curl(0.2, -6), 0.1, 0.075);
    const r = recover(shape, 4);
    expect(r.err).toBeLessThan(0.001);
    expect(r.worst).toBeLessThan(0.005);
    expect(r.solver.rmsInlier).toBeLessThan(0.5);
    expect(r.solver.inliers).toBe(250);
  });

  it('recovers a lifted corner despite outliers, annealing the robust scale', () => {
    const shape = new SheetShape(bends.lift(Math.PI / 4, 0.04, -12, 0.02), 0.1, 0.075);
    const r = recover(shape, 6, 0.3, 30, true);
    expect(r.err).toBeLessThan(0.001);
    // The corner vertex itself lies past the last points, in the tightest curl.
    expect(r.worst).toBeLessThan(0.015);
    expect(r.solver.inliers).toBeGreaterThanOrEqual(218);
    expect(r.solver.inliers).toBeLessThanOrEqual(222);
  });
});
