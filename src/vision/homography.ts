import { solveLinear, eigenSym } from '../math/linalg';

/** Apply row-major 3x3 H to (x, y); writes into out[o], out[o+1]. Returns false at infinity. */
export function applyH(H: ArrayLike<number>, x: number, y: number, out: Float32Array | Float64Array, o = 0): boolean {
  const w = H[6] * x + H[7] * y + H[8];
  if (Math.abs(w) < 1e-12) return false;
  out[o] = (H[0] * x + H[1] * y + H[2]) / w;
  out[o + 1] = (H[3] * x + H[4] * y + H[5]) / w;
  return true;
}

const A8 = new Float64Array(64);
const b8 = new Float64Array(8);

/** Exact homography from 4 correspondences (src -> dst). */
export function homographyFrom4(src: ArrayLike<number>, dst: ArrayLike<number>, H: Float64Array): boolean {
  for (let i = 0; i < 4; i++) {
    const x = src[2 * i], y = src[2 * i + 1], u = dst[2 * i], v = dst[2 * i + 1];
    const r0 = 2 * i * 8, r1 = r0 + 8;
    A8[r0] = x; A8[r0 + 1] = y; A8[r0 + 2] = 1; A8[r0 + 3] = 0; A8[r0 + 4] = 0; A8[r0 + 5] = 0;
    A8[r0 + 6] = -u * x; A8[r0 + 7] = -u * y;
    A8[r1] = 0; A8[r1 + 1] = 0; A8[r1 + 2] = 0; A8[r1 + 3] = x; A8[r1 + 4] = y; A8[r1 + 5] = 1;
    A8[r1 + 6] = -v * x; A8[r1 + 7] = -v * y;
    b8[2 * i] = u; b8[2 * i + 1] = v;
  }
  if (!solveLinear(A8, 8, b8)) return false;
  for (let i = 0; i < 8; i++) H[i] = b8[i];
  H[8] = 1;
  for (let i = 0; i < 9; i++) if (!Number.isFinite(H[i])) return false;
  return true;
}

const AtA = new Float64Array(81);
const evals = new Float64Array(9);
const evecs = new Float64Array(81);

/**
 * Normalized DLT over n correspondences (optionally only where mask[i]).
 * Hartley normalization; solution is the smallest eigenvector of A^T A.
 */
export function homographyDLT(
  src: ArrayLike<number>, dst: ArrayLike<number>, n: number, H: Float64Array, mask?: ArrayLike<number>,
): boolean {
  let m = 0, sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    sx += src[2 * i]; sy += src[2 * i + 1]; dx += dst[2 * i]; dy += dst[2 * i + 1];
    m++;
  }
  if (m < 4) return false;
  sx /= m; sy /= m; dx /= m; dy /= m;
  let ss = 0, ds = 0;
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    ss += Math.hypot(src[2 * i] - sx, src[2 * i + 1] - sy);
    ds += Math.hypot(dst[2 * i] - dx, dst[2 * i + 1] - dy);
  }
  const s1 = Math.SQRT2 / (ss / m || 1), s2 = Math.SQRT2 / (ds / m || 1);
  AtA.fill(0);
  const r = new Float64Array(9);
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    const x = (src[2 * i] - sx) * s1, y = (src[2 * i + 1] - sy) * s1;
    const u = (dst[2 * i] - dx) * s2, v = (dst[2 * i + 1] - dy) * s2;
    // Row 1: [x y 1 0 0 0 -ux -uy -u]
    r[0] = x; r[1] = y; r[2] = 1; r[3] = 0; r[4] = 0; r[5] = 0; r[6] = -u * x; r[7] = -u * y; r[8] = -u;
    addOuter(r);
    r[0] = 0; r[1] = 0; r[2] = 0; r[3] = x; r[4] = y; r[5] = 1; r[6] = -v * x; r[7] = -v * y; r[8] = -v;
    addOuter(r);
  }
  for (let i = 0; i < 9; i++) for (let j = 0; j < i; j++) AtA[i * 9 + j] = AtA[j * 9 + i];
  eigenSym(AtA, 9, evals, evecs);
  const h = new Float64Array(9);
  for (let i = 0; i < 9; i++) h[i] = evecs[i * 9];
  // Denormalize: H = T2^-1 Hn T1.
  // T1 = [[s1,0,-s1 sx],[0,s1,-s1 sy],[0,0,1]], T2^-1 = [[1/s2,0,dx],[0,1/s2,dy],[0,0,1]]
  const t1 = [s1, 0, -s1 * sx, 0, s1, -s1 * sy, 0, 0, 1];
  const t2i = [1 / s2, 0, dx, 0, 1 / s2, dy, 0, 0, 1];
  const tmp = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let s = 0;
    for (let k = 0; k < 3; k++) s += h[i * 3 + k] * t1[k * 3 + j];
    tmp[i * 3 + j] = s;
  }
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let s = 0;
    for (let k = 0; k < 3; k++) s += t2i[i * 3 + k] * tmp[k * 3 + j];
    H[i * 3 + j] = s;
  }
  const z = H[8];
  if (Math.abs(z) < 1e-15) return false;
  for (let i = 0; i < 9; i++) H[i] /= z;
  return true;
}

function addOuter(r: Float64Array): void {
  for (let i = 0; i < 9; i++) {
    const ri = r[i];
    if (ri === 0) continue;
    for (let j = i; j < 9; j++) AtA[i * 9 + j] += ri * r[j];
  }
}

const cross = (ax: number, ay: number, bx: number, by: number, cx: number, cy: number) =>
  (bx - ax) * (cy - by) - (by - ay) * (cx - bx);

/**
 * Minimal-quadrilateral check: order the 4 source points around their
 * centroid; both the source and destination quads must be convex with the
 * same orientation (a planar target cannot be seen mirrored).
 */
export function quadConsistent(src: ArrayLike<number>, dst: ArrayLike<number>): boolean {
  const cxs = (src[0] + src[2] + src[4] + src[6]) / 4, cys = (src[1] + src[3] + src[5] + src[7]) / 4;
  const ang = [0, 1, 2, 3].map((i) => Math.atan2(src[2 * i + 1] - cys, src[2 * i] - cxs));
  const ord = [0, 1, 2, 3].sort((a, b) => ang[a] - ang[b]);
  let sgnS = 0, sgnD = 0;
  for (let k = 0; k < 4; k++) {
    const a = ord[k], b = ord[(k + 1) % 4], c = ord[(k + 2) % 4];
    const cs = cross(src[2 * a], src[2 * a + 1], src[2 * b], src[2 * b + 1], src[2 * c], src[2 * c + 1]);
    const cd = cross(dst[2 * a], dst[2 * a + 1], dst[2 * b], dst[2 * b + 1], dst[2 * c], dst[2 * c + 1]);
    const ss = Math.sign(cs), sd = Math.sign(cd);
    if (ss === 0 || sd === 0) return false;
    if (sgnS === 0) { sgnS = ss; sgnD = sd; }
    if (ss !== sgnS || sd !== sgnD) return false;
  }
  return sgnS === sgnD;
}

export interface ProsacOptions {
  maxIter: number;
  /** Inlier threshold in destination pixels. */
  threshold: number;
  confidence: number;
}

export const DEFAULT_PROSAC: ProsacOptions = { maxIter: 400, threshold: 4, confidence: 0.995 };

/**
 * PROSAC: correspondences must be sorted best-first. Samples are drawn from
 * a progressively growing top-n set. Returns the inlier count; H and mask
 * hold the refined model and its inliers.
 */
export function prosacHomography(
  src: Float32Array, dst: Float32Array, n: number, H: Float64Array, mask: Uint8Array,
  rng: { int(n: number): number }, o: ProsacOptions = DEFAULT_PROSAC,
): number {
  if (n < 4) return 0;
  const s4 = new Float64Array(8), d4 = new Float64Array(8), Hc = new Float64Array(9);
  const idx = [0, 0, 0, 0];
  let best = 0;
  let maxIter = o.maxIter;
  const th2 = o.threshold * o.threshold;
  const p = new Float64Array(2);
  for (let it = 0; it < maxIter; it++) {
    // Growth schedule: start from the top 8 and reach n halfway through.
    const top = Math.min(n, Math.max(4, Math.floor(8 + (it * 2 * n) / o.maxIter)));
    idx[0] = top - 1;
    let dup = false;
    for (let k = 1; k < 4; k++) {
      const c = rng.int(Math.max(1, top - 1));
      for (let j = 0; j < k; j++) if (idx[j] === c) dup = true;
      idx[k] = c;
    }
    if (dup) continue;
    for (let k = 0; k < 4; k++) {
      s4[2 * k] = src[2 * idx[k]]; s4[2 * k + 1] = src[2 * idx[k] + 1];
      d4[2 * k] = dst[2 * idx[k]]; d4[2 * k + 1] = dst[2 * idx[k] + 1];
    }
    if (!quadConsistent(s4, d4)) continue;
    if (!homographyFrom4(s4, d4, Hc)) continue;
    let cnt = 0;
    for (let i = 0; i < n; i++) {
      if (!applyH(Hc, src[2 * i], src[2 * i + 1], p)) continue;
      const ex = p[0] - dst[2 * i], ey = p[1] - dst[2 * i + 1];
      if (ex * ex + ey * ey < th2) cnt++;
    }
    if (cnt > best) {
      best = cnt;
      H.set(Hc);
      const w = cnt / n;
      const denom = Math.log(1 - w * w * w * w);
      if (denom < 0) maxIter = Math.min(o.maxIter, Math.ceil(Math.log(1 - o.confidence) / denom) + it + 1);
    }
  }
  if (best < 4) return 0;
  // Refine with all inliers (two rounds).
  for (let round = 0; round < 2; round++) {
    let cnt = 0;
    for (let i = 0; i < n; i++) {
      mask[i] = 0;
      if (!applyH(H, src[2 * i], src[2 * i + 1], p)) continue;
      const ex = p[0] - dst[2 * i], ey = p[1] - dst[2 * i + 1];
      if (ex * ex + ey * ey < th2) { mask[i] = 1; cnt++; }
    }
    best = cnt;
    if (cnt < 8 || !homographyDLT(src, dst, n, Hc, mask)) break;
    H.set(Hc);
  }
  let cnt = 0;
  for (let i = 0; i < n; i++) {
    mask[i] = 0;
    if (!applyH(H, src[2 * i], src[2 * i + 1], p)) continue;
    const ex = p[0] - dst[2 * i], ey = p[1] - dst[2 * i + 1];
    if (ex * ex + ey * ey < th2) { mask[i] = 1; cnt++; }
  }
  return cnt || best;
}
