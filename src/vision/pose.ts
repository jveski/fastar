import type { Intrinsics } from './camera';
import { so3Exp, mat3Mul, mat3 } from '../math/rot';
import { invertSPD, solveSPD } from '../math/linalg';
import { median } from '../math/select';

export interface PoseSolverOptions {
  maxIter: number;
  /** Stop when the update norm is below this. */
  tol: number;
  /** Tukey constant in units of the robust residual scale. */
  tukeyC: number;
  /** Lower bound on the robust scale used for weighting (px). */
  minWeightSigma: number;
  /** Lower bound on the per-axis noise used for the covariance (px). */
  minCovSigma: number;
}

export const DEFAULT_POSE_OPTIONS: PoseSolverOptions = {
  maxIter: 5,
  tol: 1e-6,
  tukeyC: 4.685,
  minWeightSigma: 0.5,
  minCovSigma: 0.1,
};

/**
 * Gauss-Newton on SE(3) for planar targets (object points have z = 0),
 * minimizing reprojection error with Tukey weights. Perturbation:
 * R = Exp(dtheta) R_hat, t = t_hat + dt, state order [dtheta, dt].
 * Output covariance: sigma^2 (J^T W J)^-1 (6x6, same ordering).
 */
export class PoseSolver {
  readonly cov = new Float64Array(36);
  readonly weights: Float32Array;
  readonly residuals: Float32Array;
  sigma2 = 0;
  inliers = 0;
  iterations = 0;
  rmsInlier = 0;

  private readonly norms: Float32Array;
  private readonly H = new Float64Array(36);
  private readonly g = new Float64Array(6);
  private readonly Htmp = new Float64Array(36);
  private readonly col = new Float64Array(6);
  private readonly dR = mat3();

  constructor(maxPoints = 1024, readonly opts: PoseSolverOptions = DEFAULT_POSE_OPTIONS) {
    this.weights = new Float32Array(maxPoints);
    this.residuals = new Float32Array(maxPoints);
    this.norms = new Float32Array(maxPoints);
  }

  /**
   * Refine (R, t) in place. obj: [X, Y] per point in metres (z = 0); img:
   * [u, v] in pixels. Returns false if the problem is degenerate.
   */
  solve(K: Intrinsics, obj: Float32Array, img: Float32Array, n: number, R: Float64Array, t: Float64Array): boolean {
    if (n < 4) return false;
    const { maxIter, tol } = this.opts;
    let ok = true;
    this.iterations = 0;
    for (let it = 0; it < maxIter; it++) {
      this.iterations = it + 1;
      if (!this.linearize(K, obj, img, n, R, t)) { ok = false; break; }
      const H = this.Htmp, g = this.col;
      H.set(this.H);
      for (let i = 0; i < 6; i++) g[i] = -this.g[i];
      if (!solveSPD(H, 6, g)) { ok = false; break; }
      so3Exp(g[0], g[1], g[2], this.dR);
      mat3Mul(this.dR, R, R);
      t[0] += g[3]; t[1] += g[4]; t[2] += g[5];
      const step = Math.hypot(g[0], g[1], g[2], g[3], g[4], g[5]);
      if (!(step === step)) { ok = false; break; }
      if (step < tol) break;
    }
    if (!ok) return false;
    // Final linearization at the solution gives weights, sigma and covariance.
    if (!this.linearize(K, obj, img, n, R, t)) return false;
    let wsum = 0, wr2 = 0, inl = 0, inlr2 = 0;
    const w = this.weights, r = this.residuals;
    for (let i = 0; i < n; i++) {
      if (w[i] > 0) {
        wsum += w[i];
        wr2 += w[i] * r[i] * r[i];
        inl++;
        inlr2 += r[i] * r[i];
      }
    }
    this.inliers = inl;
    this.rmsInlier = inl > 0 ? Math.sqrt(inlr2 / inl) : Infinity;
    const dof = Math.max(1, 2 * wsum - 6);
    const minS = this.opts.minCovSigma;
    this.sigma2 = Math.max(wr2 / dof, minS * minS);
    this.Htmp.set(this.H);
    if (!invertSPD(this.Htmp, 6, this.cov, this.col)) return false;
    for (let i = 0; i < 36; i++) this.cov[i] *= this.sigma2;
    return true;
  }

  /** Build weighted normal equations H, g at (R, t); fills residuals/weights. */
  private linearize(K: Intrinsics, obj: Float32Array, img: Float32Array, n: number, R: Float64Array, t: Float64Array): boolean {
    const { fx, fy, cx, cy } = K;
    const r0 = R[0], r1 = R[1], r3 = R[3], r4 = R[4], r6 = R[6], r7 = R[7];
    const t0 = t[0], t1 = t[1], t2 = t[2];
    const res = this.residuals, norms = this.norms, w = this.weights;
    let valid = 0;
    for (let i = 0; i < n; i++) {
      const X = obj[2 * i], Y = obj[2 * i + 1];
      const pz = r6 * X + r7 * Y + t2;
      if (pz <= 1e-6) { res[i] = Infinity; continue; }
      const px = r0 * X + r1 * Y + t0, py = r3 * X + r4 * Y + t1;
      const iz = 1 / pz;
      const ex = fx * px * iz + cx - img[2 * i], ey = fy * py * iz + cy - img[2 * i + 1];
      const nr = Math.sqrt(ex * ex + ey * ey);
      res[i] = nr;
      norms[valid++] = nr;
    }
    if (valid < 4) return false;
    // Robust scale: median of 2D residual norm = 1.1774 * sigma for Gaussian noise.
    const sigma = Math.max(median(norms, valid) / 1.1774, this.opts.minWeightSigma);
    const c = this.opts.tukeyC * sigma;
    const ic2 = 1 / (c * c);
    const H = this.H, g = this.g;
    // Upper-triangular accumulators kept in locals (no calls with double
    // arguments in the hot loop, so nothing gets boxed).
    let h00 = 0, h01 = 0, h02 = 0, h03 = 0, h04 = 0, h05 = 0;
    let h11 = 0, h12 = 0, h13 = 0, h14 = 0, h15 = 0;
    let h22 = 0, h23 = 0, h24 = 0, h25 = 0;
    let h33 = 0, h35 = 0, h44 = 0, h45 = 0, h55 = 0;
    let g0 = 0, g1 = 0, g2 = 0, g3 = 0, g4 = 0, g5 = 0;
    let wcount = 0;
    for (let i = 0; i < n; i++) {
      const nr = res[i];
      if (!(nr < c)) { w[i] = 0; continue; }
      const q = 1 - nr * nr * ic2;
      const wi = q * q;
      w[i] = wi;
      wcount++;
      const X = obj[2 * i], Y = obj[2 * i + 1];
      const ax = r0 * X + r1 * Y, ay = r3 * X + r4 * Y, az = r6 * X + r7 * Y;
      const px = ax + t0, py = ay + t1, pz = az + t2;
      const iz = 1 / pz;
      const ex = fx * px * iz + cx - img[2 * i], ey = fy * py * iz + cy - img[2 * i + 1];
      const d0x = fx * iz, d0z = -fx * px * iz * iz;
      const d1y = fy * iz, d1z = -fy * py * iz * iz;
      // Jacobian rows wrt [dtheta, dt]; dPc/dtheta = -[R X]x. Row 0 has
      // j04 = 0 and row 1 has j13 = 0.
      const a0 = d0z * ay, a1 = d0x * az - d0z * ax, a2 = -d0x * ay, a3 = d0x, a5 = d0z;
      const b0 = -d1y * az + d1z * ay, b1 = -d1z * ax, b2 = d1y * ax, b4 = d1y, b5 = d1z;
      const wa0 = wi * a0, wa1 = wi * a1, wa2 = wi * a2, wa3 = wi * a3, wa5 = wi * a5;
      const wb0 = wi * b0, wb1 = wi * b1, wb2 = wi * b2, wb4 = wi * b4, wb5 = wi * b5;
      h00 += wa0 * a0 + wb0 * b0; h01 += wa0 * a1 + wb0 * b1; h02 += wa0 * a2 + wb0 * b2;
      h03 += wa0 * a3; h04 += wb0 * b4; h05 += wa0 * a5 + wb0 * b5;
      h11 += wa1 * a1 + wb1 * b1; h12 += wa1 * a2 + wb1 * b2; h13 += wa1 * a3; h14 += wb1 * b4; h15 += wa1 * a5 + wb1 * b5;
      h22 += wa2 * a2 + wb2 * b2; h23 += wa2 * a3; h24 += wb2 * b4; h25 += wa2 * a5 + wb2 * b5;
      h33 += wa3 * a3; h35 += wa3 * a5;
      h44 += wb4 * b4; h45 += wb4 * b5;
      h55 += wa5 * a5 + wb5 * b5;
      g0 += wa0 * ex + wb0 * ey; g1 += wa1 * ex + wb1 * ey; g2 += wa2 * ex + wb2 * ey;
      g3 += wa3 * ex; g4 += wb4 * ey; g5 += wa5 * ex + wb5 * ey;
    }
    if (wcount < 3) return false;
    H[0] = h00; H[1] = h01; H[2] = h02; H[3] = h03; H[4] = h04; H[5] = h05;
    H[7] = h11; H[8] = h12; H[9] = h13; H[10] = h14; H[11] = h15;
    H[14] = h22; H[15] = h23; H[16] = h24; H[17] = h25;
    H[21] = h33; H[22] = 0; H[23] = h35;
    H[28] = h44; H[29] = h45;
    H[35] = h55;
    for (let i = 0; i < 6; i++) for (let j = 0; j < i; j++) H[i * 6 + j] = H[j * 6 + i];
    g[0] = g0; g[1] = g1; g[2] = g2; g[3] = g3; g[4] = g4; g[5] = g5;
    return true;
  }
}

/** Project a target point (X, Y, 0) with pose (R, t); returns false if behind camera. */
export function projectPoint(K: Intrinsics, R: ArrayLike<number>, t: ArrayLike<number>, X: number, Y: number, out: Float32Array | Float64Array, o = 0): boolean {
  const pz = R[6] * X + R[7] * Y + t[2];
  if (pz <= 1e-6) return false;
  out[o] = (K.fx * (R[0] * X + R[1] * Y + t[0])) / pz + K.cx;
  out[o + 1] = (K.fy * (R[3] * X + R[4] * Y + t[1])) / pz + K.cy;
  return true;
}

/**
 * Homography mapping target pixels (u, v) to frame pixels, given the target
 * pixel->metre mapping X = (u - cu) * s, Y = (v - cv) * s.
 * H = K [r1 r2 t] A with A the pixel->metre affine.
 */
export function poseToHomography(
  K: Intrinsics, R: ArrayLike<number>, t: ArrayLike<number>, s: number, cu: number, cv: number, out: Float64Array,
): Float64Array {
  // M = K [r1 r2 t], then right-multiply by A = [[s, 0, -s cu], [0, s, -s cv], [0, 0, 1]].
  const m00 = K.fx * R[0] + K.cx * R[6], m01 = K.fx * R[1] + K.cx * R[7], m02 = K.fx * t[0] + K.cx * t[2];
  const m10 = K.fy * R[3] + K.cy * R[6], m11 = K.fy * R[4] + K.cy * R[7], m12 = K.fy * t[1] + K.cy * t[2];
  const m20 = R[6], m21 = R[7], m22 = t[2];
  out[0] = m00 * s; out[1] = m01 * s; out[2] = -(m00 * cu + m01 * cv) * s + m02;
  out[3] = m10 * s; out[4] = m11 * s; out[5] = -(m10 * cu + m11 * cv) * s + m12;
  out[6] = m20 * s; out[7] = m21 * s; out[8] = -(m20 * cu + m21 * cv) * s + m22;
  return out;
}
