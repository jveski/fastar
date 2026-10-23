import { describe, expect, it } from 'vitest';
import {
  mat3, mat3Mul, mat3MulAtB, mat3MulABt, mat3FromQuat, quatFromMat3, quat, so3Exp, so3Log, vec3,
  quatIntegrateRight, rotationAngle,
} from '../src/math/rot';
import { eigenSym, invertSPD, solveLinear } from '../src/math/linalg';
import { Rng } from '../src/math/rng';

describe('rotations', () => {
  it('exp/log round trip, including near 0 and near pi', () => {
    const R = mat3();
    const w = vec3();
    const rng = new Rng(3);
    const cases: number[][] = [[1e-9, 0, 0], [0.3, -0.2, 0.1], [Math.PI - 1e-4, 0, 0], [0, 2.5, 1.5]];
    for (let i = 0; i < 50; i++) cases.push([rng.gauss(), rng.gauss(), rng.gauss()]);
    for (const c of cases) {
      let [x, y, z] = c;
      const th = Math.hypot(x, y, z);
      if (th > Math.PI) {
        const s = (th - 2 * Math.PI * Math.round(th / (2 * Math.PI))) / th;
        x *= s; y *= s; z *= s;
      }
      so3Exp(x, y, z, R);
      so3Log(R, w);
      const R2 = so3Exp(w[0], w[1], w[2], mat3());
      expect(rotationAngle(R, R2)).toBeLessThan(1e-6);
    }
  });

  it('quaternion and matrix agree', () => {
    const R = so3Exp(0.4, -0.7, 1.1, mat3());
    const q = quatFromMat3(R, quat());
    const R2 = mat3FromQuat(q, mat3());
    for (let i = 0; i < 9; i++) expect(R2[i]).toBeCloseTo(R[i], 12);
  });

  it('right integration composes like matrices', () => {
    const q = quat();
    const R = so3Exp(0, 0, 0, mat3());
    const dR = mat3();
    for (let i = 0; i < 100; i++) {
      quatIntegrateRight(q, 0.01, 0.02, -0.005);
      mat3Mul(R, so3Exp(0.01, 0.02, -0.005, dR), R);
    }
    expect(rotationAngle(mat3FromQuat(q, mat3()), R)).toBeLessThan(1e-9);
  });

  it('transpose products', () => {
    const A = so3Exp(0.1, 0.2, 0.3, mat3());
    const I1 = mat3MulAtB(A, A, mat3());
    const I2 = mat3MulABt(A, A, mat3());
    for (let i = 0; i < 9; i++) {
      expect(I1[i]).toBeCloseTo(i % 4 === 0 ? 1 : 0, 12);
      expect(I2[i]).toBeCloseTo(i % 4 === 0 ? 1 : 0, 12);
    }
  });
});

describe('linalg', () => {
  it('inverts SPD matrices', () => {
    const n = 6;
    const rng = new Rng(1);
    const B = new Float64Array(n * n).map(() => rng.gauss());
    const A = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let s = i === j ? 1 : 0;
      for (let k = 0; k < n; k++) s += B[i * n + k] * B[j * n + k];
      A[i * n + j] = s;
    }
    const Ac = A.slice();
    const inv = new Float64Array(n * n);
    expect(invertSPD(Ac, n, inv)).toBe(true);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let p = 0;
      for (let k = 0; k < n; k++) p += A[i * n + k] * inv[k * n + j];
      expect(p).toBeCloseTo(i === j ? 1 : 0, 9);
    }
  });

  it('solves general systems', () => {
    const A = Float64Array.of(0, 2, 1, 1, 1, 1, 2, 0, 3);
    const b = Float64Array.of(5, 6, 13);
    expect(solveLinear(A, 3, b)).toBe(true);
    expect([...b].map((v) => +v.toFixed(9))).toEqual([2, 1, 3]);
  });

  it('eigen-decomposes symmetric matrices', () => {
    const A = Float64Array.of(4, 1, 0, 1, 3, 1, 0, 1, 2);
    const vals = new Float64Array(3);
    const V = new Float64Array(9);
    eigenSym(A.slice(), 3, vals, V);
    for (let k = 0; k < 3; k++) {
      for (let i = 0; i < 3; i++) {
        let s = 0;
        for (let j = 0; j < 3; j++) s += A[i * 3 + j] * V[j * 3 + k];
        expect(s).toBeCloseTo(vals[k] * V[i * 3 + k], 9);
      }
    }
    expect(vals[0]).toBeLessThanOrEqual(vals[1]);
  });
});
