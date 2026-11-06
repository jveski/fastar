// Shape of a bending target from tracked points: Gauss-Newton over the mesh
// vertices (camera frame) minimizing
//   robust reprojection error of the tracked points (barycentric in their faces)
//   + stretch: edges (grid lines and both diagonals of each cell) keep their
//     rest lengths, linearized at the current shape (Gauss-Newton)
//   + bend: second differences along grid lines and diagonals stay small
//   + inertia: vertices stay near the predicted shape (more so along the sheet's normal)
// The regularizers are measured in metres and scaled to pixels at the sheet's
// depth, so their weights compare with a point's squared pixel error. The
// normal equations are banded (vertices ordered along the short side of the
// grid) and solved by banded Cholesky. Isometry fixes depth and scale: the
// mesh has the target's real size.
import type { Intrinsics } from '../vision/camera';
import { bandCholesky, bandSolve } from '../math/linalg';
import { median } from '../math/select';
import type { SurfaceMesh } from './mesh';

export interface SurfaceSolverOptions {
  /** Weight of an edge's change in length (per edge). */
  stretch: number;
  /** Weight of second differences along grid lines and diagonals. */
  bend: number;
  /** Pull toward the predicted shape in the sheet's plane, and along its normal. */
  inertia: number;
  inertiaNormal: number;
  iterations: number;
  /** Tukey constant in units of the robust residual scale. */
  tukeyC: number;
  /** Lower bound on the robust scale (px). */
  minSigma: number;
}

export const DEFAULT_SURFACE_SOLVER: SurfaceSolverOptions = {
  stretch: 10,
  bend: 0.003,
  inertia: 0.001,
  inertiaNormal: 0.002,
  iterations: 2,
  tukeyC: 4.685,
  minSigma: 0.5,
};

export class SurfaceSolver {
  readonly o: SurfaceSolverOptions;
  /** Per point, after solve: robust weight (0 = outlier) and residual (px). */
  readonly weights: Float32Array;
  readonly residuals: Float32Array;
  inliers = 0;
  rmsInlier = 0;

  private readonly slot: Int32Array;
  private readonly band: number;
  private readonly size: number;
  private readonly A: Float64Array;
  private readonly x: Float64Array;
  private readonly triplets: Int32Array;
  /** Edges [p, q] with rest lengths: grid lines and both diagonals of each cell. */
  private readonly edges: Int32Array;
  private readonly edgeLen: Float64Array;
  private readonly norms: Float32Array;

  constructor(readonly mesh: SurfaceMesh, maxPoints: number, o: Partial<SurfaceSolverOptions> = {}) {
    this.o = { ...DEFAULT_SURFACE_SOLVER, ...o };
    const { nx, ny } = mesh;
    // Order vertices along the short side so couplings stay close together.
    this.slot = new Int32Array(mesh.count);
    const nf = Math.min(nx, ny);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) this.slot[j * nx + i] = nx <= ny ? j * nx + i : i * ny + j;
    this.band = 3 * (2 * nf + 2) + 2;
    this.size = 3 * mesh.count;
    this.A = new Float64Array(this.size * (this.band + 1));
    this.x = new Float64Array(this.size);
    const t: number[] = [];
    const v = (i: number, j: number) => j * nx + i;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        if (i > 0 && i < nx - 1) t.push(v(i - 1, j), v(i, j), v(i + 1, j));
        if (j > 0 && j < ny - 1) t.push(v(i, j - 1), v(i, j), v(i, j + 1));
        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
          t.push(v(i - 1, j - 1), v(i, j), v(i + 1, j + 1));
          t.push(v(i + 1, j - 1), v(i, j), v(i - 1, j + 1));
        }
      }
    }
    this.triplets = Int32Array.from(t);
    const e: number[] = [];
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        if (i + 1 < nx) e.push(v(i, j), v(i + 1, j));
        if (j + 1 < ny) e.push(v(i, j), v(i, j + 1));
        if (i + 1 < nx && j + 1 < ny) e.push(v(i, j), v(i + 1, j + 1), v(i + 1, j), v(i, j + 1));
      }
    }
    this.edges = Int32Array.from(e);
    this.edgeLen = new Float64Array(e.length / 2);
    const r = mesh.rest;
    for (let k = 0; k < this.edgeLen.length; k++) {
      const p = 3 * e[2 * k], q = 3 * e[2 * k + 1];
      this.edgeLen[k] = Math.hypot(r[q] - r[p], r[q + 1] - r[p + 1]);
    }
    this.weights = new Float32Array(maxPoints);
    this.residuals = new Float32Array(maxPoints);
    this.norms = new Float32Array(maxPoints);
  }

  /**
   * Refine camera-frame vertices V in place. Vp: predicted vertices (the
   * inertia term's anchor); normal: the sheet's normal in camera axes. Point
   * k lies at sum_m w[3k+m] V[idx[3k+m]] and was seen at img[2k..2k+1].
   * minSigma: floor of the robust scale (px); start wide and shrink it to
   * pull in a shape far from the start without outliers taking over.
   * stiffness: multiplies the bend weight (start stiff, then relax, so early
   * coarse matches only shape the sheet broadly). iterations: Gauss-Newton
   * steps (from a good start, as when tracking, two do). Returns false if the system is degenerate (V may then be partly updated).
   */
  solve(
    K: Intrinsics, V: Float64Array, Vp: Float64Array, normal: ArrayLike<number>,
    idx: Int32Array, w: Float32Array, img: Float32Array, n: number, minSigma = this.o.minSigma, stiffness = 1,
    iterations = this.o.iterations,
  ): boolean {
    const o = this.o, N = this.mesh.count;
    let zs = 0;
    for (let v = 0; v < N; v++) zs += V[3 * v + 2];
    const zm = zs / N;
    if (!(zm > 1e-3)) return false;
    const s2 = (K.fx / zm) * (K.fx / zm);
    for (let it = 0; it < iterations; it++) {
      this.robustWeights(K, V, idx, w, img, n, minSigma);
      this.A.fill(0);
      this.x.fill(0);
      this.addData(K, V, idx, w, img, n);
      this.addLengths(V, o.stretch * s2);
      this.addBend(o.bend * stiffness * s2);
      this.addInertia(Vp, normal, o.inertia * s2, o.inertiaNormal * s2);
      if (!bandCholesky(this.A, this.size, this.band)) return false;
      bandSolve(this.A, this.size, this.band, this.x);
      for (let v = 0; v < N; v++) {
        const s = 3 * this.slot[v];
        V[3 * v] = this.x[s]; V[3 * v + 1] = this.x[s + 1]; V[3 * v + 2] = this.x[s + 2];
      }
    }
    this.robustWeights(K, V, idx, w, img, n, minSigma);
    let inl = 0, r2 = 0;
    for (let k = 0; k < n; k++) {
      if (this.weights[k] > 0) { inl++; r2 += this.residuals[k] * this.residuals[k]; }
    }
    this.inliers = inl;
    this.rmsInlier = inl ? Math.sqrt(r2 / inl) : Infinity;
    return true;
  }

  /** Largest relative change in an edge's length at V (paper hardly stretches: large values mean a bad fit). */
  strain(V: Float64Array): number {
    const E = this.edges, L = this.edgeLen;
    let m = 0;
    for (let k = 0; k < L.length; k++) {
      const p = 3 * E[2 * k], q = 3 * E[2 * k + 1];
      const ex = V[q] - V[p], ey = V[q + 1] - V[p + 1], ez = V[q + 2] - V[p + 2];
      const d = Math.abs(Math.sqrt(ex * ex + ey * ey + ez * ez) / L[k] - 1);
      if (d > m) m = d;
    }
    return m;
  }

  /** Residuals and Tukey weights at V. */
  private robustWeights(K: Intrinsics, V: Float64Array, idx: Int32Array, w: Float32Array, img: Float32Array, n: number, minSigma: number): void {
    const res = this.residuals, wt = this.weights, norms = this.norms;
    let m = 0;
    for (let k = 0; k < n; k++) {
      const a = 3 * idx[3 * k], b = 3 * idx[3 * k + 1], c = 3 * idx[3 * k + 2];
      const wa = w[3 * k], wb = w[3 * k + 1], wc = w[3 * k + 2];
      const px = wa * V[a] + wb * V[b] + wc * V[c];
      const py = wa * V[a + 1] + wb * V[b + 1] + wc * V[c + 1];
      const pz = wa * V[a + 2] + wb * V[b + 2] + wc * V[c + 2];
      if (pz <= 1e-6) { res[k] = Infinity; continue; }
      const ex = K.fx * px / pz + K.cx - img[2 * k], ey = K.fy * py / pz + K.cy - img[2 * k + 1];
      const r = Math.sqrt(ex * ex + ey * ey);
      res[k] = r;
      norms[m++] = r;
    }
    const sigma = Math.max(m ? median(norms, m) / 1.1774 : 0, minSigma);
    const c = this.o.tukeyC * sigma, ic2 = 1 / (c * c);
    for (let k = 0; k < n; k++) {
      const r = res[k];
      if (!(r < c)) { wt[k] = 0; continue; }
      const q = 1 - r * r * ic2;
      wt[k] = q * q;
    }
  }

  // The helpers below take only integer arguments: scalars travel in fields,
  // so no doubles are boxed in the assembly loops.
  /** Scalar for addId. */
  private sc = 0;
  /** Right-hand side for addRhs. */
  private r0 = 0;
  private r1 = 0;
  private r2 = 0;
  /** Block for addBlock (symmetric 3x3, row-major). */
  private readonly G = new Float64Array(9);
  /** Scaled copy of G for addBlock. */
  private readonly Gs = new Float64Array(9);

  /** A[p, q] += sc I and its transpose (lower band). */
  private addId(p: number, q: number): void {
    const A = this.A, w = this.band + 1, b = this.band, s = this.sc;
    const sp = 3 * this.slot[p], sq = 3 * this.slot[q];
    if (sp === sq) {
      for (let k = 0; k < 3; k++) A[(sp + k) * w + b] += s;
      return;
    }
    const r = sp > sq ? sp : sq, c = sp > sq ? sq : sp;
    for (let k = 0; k < 3; k++) A[(r + k) * w + (c + k) - (r + k) + b] += s;
  }

  /** rhs[p] += (r0, r1, r2). */
  private addRhs(p: number): void {
    const s = 3 * this.slot[p];
    this.x[s] += this.r0; this.x[s + 1] += this.r1; this.x[s + 2] += this.r2;
  }

  /** A[p, q] += Gs and its transpose (lower band). */
  private addBlock(p: number, q: number): void {
    const A = this.A, w = this.band + 1, b = this.band, g = this.Gs;
    const sp = 3 * this.slot[p], sq = 3 * this.slot[q];
    if (sp === sq) {
      for (let r = 0; r < 3; r++) for (let c = 0; c <= r; c++) A[(sp + r) * w + (sp + c) - (sp + r) + b] += g[3 * r + c];
      return;
    }
    const R = sp > sq ? sp : sq, C = sp > sq ? sq : sp;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) A[(R + r) * w + (C + c) - (R + r) + b] += g[3 * r + c];
  }

  /** Gs = s G. */
  private scaleG(s: number): void {
    const G = this.G, Gs = this.Gs;
    for (let k = 0; k < 9; k++) Gs[k] = s * G[k];
  }

  private addData(K: Intrinsics, V: Float64Array, idx: Int32Array, w: Float32Array, img: Float32Array, n: number): void {
    const G = this.G, wt = this.weights;
    for (let k = 0; k < n; k++) {
      const wk = wt[k];
      if (wk <= 0) continue;
      const ia = idx[3 * k], ib = idx[3 * k + 1], ic = idx[3 * k + 2];
      const a = 3 * ia, b = 3 * ib, c = 3 * ic;
      const wa = w[3 * k], wb = w[3 * k + 1], wc = w[3 * k + 2];
      const px = wa * V[a] + wb * V[b] + wc * V[c];
      const py = wa * V[a + 1] + wb * V[b + 1] + wc * V[c + 1];
      const pz = wa * V[a + 2] + wb * V[b + 2] + wc * V[c + 2];
      const iz = 1 / pz;
      const ex = K.fx * px * iz + K.cx - img[2 * k], ey = K.fy * py * iz + K.cy - img[2 * k + 1];
      // J = [[fx/z, 0, -fx x/z^2], [0, fy/z, -fy y/z^2]]
      const j00 = K.fx * iz, j02 = -K.fx * px * iz * iz, j11 = K.fy * iz, j12 = -K.fy * py * iz * iz;
      G[0] = wk * j00 * j00; G[1] = 0; G[2] = wk * j00 * j02;
      G[3] = 0; G[4] = wk * j11 * j11; G[5] = wk * j11 * j12;
      G[6] = G[2]; G[7] = G[5]; G[8] = wk * (j02 * j02 + j12 * j12);
      // Right-hand side: J^T W (J P - e).
      const rx = j00 * px + j02 * pz - ex, ry = j11 * py + j12 * pz - ey;
      const gx = wk * j00 * rx, gy = wk * j11 * ry, gz = wk * (j02 * rx + j12 * ry);
      this.scaleG(wa * wa); this.addBlock(ia, ia);
      this.scaleG(wb * wb); this.addBlock(ib, ib);
      this.scaleG(wc * wc); this.addBlock(ic, ic);
      this.scaleG(wa * wb); this.addBlock(ia, ib);
      this.scaleG(wa * wc); this.addBlock(ia, ic);
      this.scaleG(wb * wc); this.addBlock(ib, ic);
      this.r0 = wa * gx; this.r1 = wa * gy; this.r2 = wa * gz; this.addRhs(ia);
      this.r0 = wb * gx; this.r1 = wb * gy; this.r2 = wb * gz; this.addRhs(ib);
      this.r0 = wc * gx; this.r1 = wc * gy; this.r2 = wc * gz; this.addRhs(ic);
    }
  }

  /** Edge lengths, linearized at V (rotation-invariant, so bending converges in a few steps). */
  private addLengths(V: Float64Array, weight: number): void {
    if (weight <= 0) return;
    const E = this.edges, L = this.edgeLen, G = this.G, cw2 = this.mesh.cw * this.mesh.cw;
    for (let k = 0; k < L.length; k++) {
      const p = E[2 * k], q = E[2 * k + 1];
      const ex = V[3 * q] - V[3 * p], ey = V[3 * q + 1] - V[3 * p + 1], ez = V[3 * q + 2] - V[3 * p + 2];
      const l = Math.sqrt(ex * ex + ey * ey + ez * ez);
      if (!(l > 1e-9)) continue;
      const ux = ex / l, uy = ey / l, uz = ez / l;
      // Strain-like: longer edges weigh less for the same stretch.
      const s = (weight * cw2) / (L[k] * L[k]);
      G[0] = ux * ux; G[1] = ux * uy; G[2] = ux * uz;
      G[3] = G[1]; G[4] = uy * uy; G[5] = uy * uz;
      G[6] = G[2]; G[7] = G[5]; G[8] = uz * uz;
      this.scaleG(s);
      this.addBlock(p, p);
      this.addBlock(q, q);
      this.scaleG(-s);
      this.addBlock(p, q);
      const b = s * L[k];
      this.r0 = b * ux; this.r1 = b * uy; this.r2 = b * uz;
      this.addRhs(q);
      this.r0 = -this.r0; this.r1 = -this.r1; this.r2 = -this.r2;
      this.addRhs(p);
    }
  }

  private addBend(weight: number): void {
    if (weight <= 0) return;
    const t = this.triplets;
    for (let k = 0; k < t.length; k += 3) {
      const a = t[k], b = t[k + 1], c = t[k + 2];
      this.sc = weight; this.addId(a, a); this.addId(c, c); this.addId(a, c);
      this.sc = 4 * weight; this.addId(b, b);
      this.sc = -2 * weight; this.addId(a, b); this.addId(b, c);
    }
  }

  private addInertia(Vp: Float64Array, nrm: ArrayLike<number>, wt: number, wn: number): void {
    const G = this.G, d = wn - wt;
    const nx = nrm[0], ny = nrm[1], nz = nrm[2];
    G[0] = wt + d * nx * nx; G[1] = d * nx * ny; G[2] = d * nx * nz;
    G[3] = G[1]; G[4] = wt + d * ny * ny; G[5] = d * ny * nz;
    G[6] = G[2]; G[7] = G[5]; G[8] = wt + d * nz * nz;
    this.scaleG(1);
    for (let v = 0; v < this.mesh.count; v++) {
      this.addBlock(v, v);
      const x = Vp[3 * v], y = Vp[3 * v + 1], z = Vp[3 * v + 2];
      this.r0 = G[0] * x + G[1] * y + G[2] * z; this.r1 = G[3] * x + G[4] * y + G[5] * z; this.r2 = G[6] * x + G[7] * y + G[8] * z;
      this.addRhs(v);
    }
  }
}
