// IPPE: Infinitesimal Plane-based Pose Estimation (Collins & Bartoli 2014).
// Returns both pose solutions of a planar target, sorted by reprojection
// error. Object points lie on z = 0; image points are normalized
// (K^-1 applied).
import { homographyDLT } from './homography';
import { solveSPD } from '../math/linalg';

export interface IppeSolution {
  R: Float64Array;
  t: Float64Array;
  /** RMS reprojection error in normalized units. */
  err: number;
}

export function ippe(obj: ArrayLike<number>, img: ArrayLike<number>, n: number): IppeSolution[] {
  if (n < 4) return [];
  // Centre object points so the Jacobian is taken at the centroid.
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += obj[2 * i]; my += obj[2 * i + 1]; }
  mx /= n; my /= n;
  const oc = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) { oc[2 * i] = obj[2 * i] - mx; oc[2 * i + 1] = obj[2 * i + 1] - my; }
  const H = new Float64Array(9);
  if (!homographyDLT(oc, img, n, H)) return [];
  // H maps centred plane points to normalized image points; H33 = 1.
  const p = H[2], q = H[5];
  const j00 = H[0] - H[6] * p, j01 = H[1] - H[7] * p;
  const j10 = H[3] - H[6] * q, j11 = H[4] - H[7] * q;
  const rots = ippeRotations(j00, j01, j10, j11, p, q);
  if (!rots) return [];
  const out: IppeSolution[] = [];
  for (const R of rots) {
    const t = translationLS(R, oc, img, n);
    if (!t) continue;
    // Undo centring: R (X - m) + t = R X + (t - R m).
    t[0] -= R[0] * mx + R[1] * my;
    t[1] -= R[3] * mx + R[4] * my;
    t[2] -= R[6] * mx + R[7] * my;
    out.push({ R, t, err: reprojErr(R, t, obj, img, n) });
  }
  out.sort((a, b) => a.err - b.err);
  return out;
}

/** The two rotations consistent with Jacobian J at image point (p, q). */
export function ippeRotations(j00: number, j01: number, j10: number, j11: number, p: number, q: number): [Float64Array, Float64Array] | null {
  // Rv: rotation whose third column is the unit ray v = [p, q, 1] / |.|.
  const nv = Math.hypot(p, q, 1);
  const ax = p / nv, ay = q / nv, az = 1 / nv;
  const d = 1 / (1 + az);
  const Rv = Float64Array.of(
    1 - ax * ax * d, -ax * ay * d, ax,
    -ax * ay * d, 1 - ay * ay * d, ay,
    -ax, -ay, az,
  );
  // B = [[1, 0, -p], [0, 1, -q]] * Rv[:, 0:2]
  const b00 = Rv[0] - p * Rv[6], b01 = Rv[1] - p * Rv[7];
  const b10 = Rv[3] - q * Rv[6], b11 = Rv[4] - q * Rv[7];
  const det = b00 * b11 - b01 * b10;
  if (Math.abs(det) < 1e-15) return null;
  const i00 = b11 / det, i01 = -b01 / det, i10 = -b10 / det, i11 = b00 / det;
  // A = B^-1 J
  const a00 = i00 * j00 + i01 * j10, a01 = i00 * j01 + i01 * j11;
  const a10 = i10 * j00 + i11 * j10, a11 = i10 * j01 + i11 * j11;
  // Largest singular value of A.
  const s00 = a00 * a00 + a10 * a10, s01 = a00 * a01 + a10 * a11, s11 = a01 * a01 + a11 * a11;
  const gamma = Math.sqrt(0.5 * (s00 + s11 + Math.sqrt((s00 - s11) ** 2 + 4 * s01 * s01)));
  if (!(gamma > 1e-12)) return null;
  const r00 = a00 / gamma, r01 = a01 / gamma, r10 = a10 / gamma, r11 = a11 / gamma;
  const b0 = Math.sqrt(Math.max(0, 1 - r00 * r00 - r10 * r10));
  let b1 = Math.sqrt(Math.max(0, 1 - r01 * r01 - r11 * r11));
  if (-r00 * r01 - r10 * r11 < 0) b1 = -b1;
  const make = (c0: number, c1: number): Float64Array => {
    // Columns in the Rv frame: m1 = [r00, r10, c0], m2 = [r01, r11, c1], m3 = m1 x m2.
    const m1 = [r00, r10, c0], m2 = [r01, r11, c1];
    const m3 = [m1[1] * m2[2] - m1[2] * m2[1], m1[2] * m2[0] - m1[0] * m2[2], m1[0] * m2[1] - m1[1] * m2[0]];
    const R = new Float64Array(9);
    for (let i = 0; i < 3; i++) {
      R[i * 3] = Rv[i * 3] * m1[0] + Rv[i * 3 + 1] * m1[1] + Rv[i * 3 + 2] * m1[2];
      R[i * 3 + 1] = Rv[i * 3] * m2[0] + Rv[i * 3 + 1] * m2[1] + Rv[i * 3 + 2] * m2[2];
      R[i * 3 + 2] = Rv[i * 3] * m3[0] + Rv[i * 3 + 1] * m3[1] + Rv[i * 3 + 2] * m3[2];
    }
    return R;
  };
  return [make(b0, b1), make(-b0, -b1)];
}

/** Least-squares translation given R: x (r3.X + t3) = r1.X + t1, same for y. */
function translationLS(R: Float64Array, obj: ArrayLike<number>, img: ArrayLike<number>, n: number): Float64Array | null {
  const A = new Float64Array(9), b = new Float64Array(3);
  for (let i = 0; i < n; i++) {
    const X = obj[2 * i], Y = obj[2 * i + 1];
    const x = img[2 * i], y = img[2 * i + 1];
    const rx = R[0] * X + R[1] * Y, ry = R[3] * X + R[4] * Y, rz = R[6] * X + R[7] * Y;
    // Rows: [-1, 0, x] t = rx - x rz ; [0, -1, y] t = ry - y rz
    const e0 = rx - x * rz, e1 = ry - y * rz;
    A[0] += 1; A[2] += -x; A[8] += x * x;
    b[0] += -e0; b[2] += x * e0;
    A[4] += 1; A[5] += -y; A[8] += y * y;
    b[1] += -e1; b[2] += y * e1;
  }
  A[6] = A[2]; A[7] = A[5];
  return solveSPD(A, 3, b) ? b : null;
}

function reprojErr(R: Float64Array, t: Float64Array, obj: ArrayLike<number>, img: ArrayLike<number>, n: number): number {
  let s = 0;
  for (let i = 0; i < n; i++) {
    const X = obj[2 * i], Y = obj[2 * i + 1];
    const z = R[6] * X + R[7] * Y + t[2];
    if (z <= 0) return Infinity;
    const ex = (R[0] * X + R[1] * Y + t[0]) / z - img[2 * i];
    const ey = (R[3] * X + R[4] * Y + t[1]) / z - img[2 * i + 1];
    s += ex * ex + ey * ey;
  }
  return Math.sqrt(s / n);
}
