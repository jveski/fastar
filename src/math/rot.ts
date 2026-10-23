// Rotation utilities. Mat3 is row-major Float64Array(9). Quat is [w, x, y, z]
// (Hamilton). Rotation perturbations are left (global) unless noted:
// R = Exp(dtheta) * R_hat.

export type Mat3 = Float64Array;
export type Vec3 = Float64Array;
export type Quat = Float64Array;

export const mat3 = (): Mat3 => new Float64Array(9);
export const vec3 = (x = 0, y = 0, z = 0): Vec3 => Float64Array.of(x, y, z);
export const quat = (): Quat => Float64Array.of(1, 0, 0, 0);

export function mat3Mul(a: ArrayLike<number>, b: ArrayLike<number>, out: Mat3): Mat3 {
  const a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7], a8 = a[8];
  const b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7], b8 = b[8];
  out[0] = a0 * b0 + a1 * b3 + a2 * b6;
  out[1] = a0 * b1 + a1 * b4 + a2 * b7;
  out[2] = a0 * b2 + a1 * b5 + a2 * b8;
  out[3] = a3 * b0 + a4 * b3 + a5 * b6;
  out[4] = a3 * b1 + a4 * b4 + a5 * b7;
  out[5] = a3 * b2 + a4 * b5 + a5 * b8;
  out[6] = a6 * b0 + a7 * b3 + a8 * b6;
  out[7] = a6 * b1 + a7 * b4 + a8 * b7;
  out[8] = a6 * b2 + a7 * b5 + a8 * b8;
  return out;
}

/** out = a^T * b */
export function mat3MulAtB(a: ArrayLike<number>, b: ArrayLike<number>, out: Mat3): Mat3 {
  const a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7], a8 = a[8];
  const b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7], b8 = b[8];
  out[0] = a0 * b0 + a3 * b3 + a6 * b6;
  out[1] = a0 * b1 + a3 * b4 + a6 * b7;
  out[2] = a0 * b2 + a3 * b5 + a6 * b8;
  out[3] = a1 * b0 + a4 * b3 + a7 * b6;
  out[4] = a1 * b1 + a4 * b4 + a7 * b7;
  out[5] = a1 * b2 + a4 * b5 + a7 * b8;
  out[6] = a2 * b0 + a5 * b3 + a8 * b6;
  out[7] = a2 * b1 + a5 * b4 + a8 * b7;
  out[8] = a2 * b2 + a5 * b5 + a8 * b8;
  return out;
}

/** out = a * b^T */
export function mat3MulABt(a: ArrayLike<number>, b: ArrayLike<number>, out: Mat3): Mat3 {
  const a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7], a8 = a[8];
  const b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7], b8 = b[8];
  out[0] = a0 * b0 + a1 * b1 + a2 * b2;
  out[1] = a0 * b3 + a1 * b4 + a2 * b5;
  out[2] = a0 * b6 + a1 * b7 + a2 * b8;
  out[3] = a3 * b0 + a4 * b1 + a5 * b2;
  out[4] = a3 * b3 + a4 * b4 + a5 * b5;
  out[5] = a3 * b6 + a4 * b7 + a5 * b8;
  out[6] = a6 * b0 + a7 * b1 + a8 * b2;
  out[7] = a6 * b3 + a7 * b4 + a8 * b5;
  out[8] = a6 * b6 + a7 * b7 + a8 * b8;
  return out;
}

/** General 3x3 inverse. Returns false if singular. */
export function mat3Invert(m: ArrayLike<number>, out: Float64Array): boolean {
  const a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-300) return false;
  const id = 1 / det;
  out[0] = A * id; out[1] = -(b * i - c * h) * id; out[2] = (b * f - c * e) * id;
  out[3] = B * id; out[4] = (a * i - c * g) * id; out[5] = -(a * f - c * d) * id;
  out[6] = C * id; out[7] = -(a * h - b * g) * id; out[8] = (a * e - b * d) * id;
  return true;
}

export function mat3Vec(m: ArrayLike<number>, x: number, y: number, z: number, out: Vec3): Vec3 {
  out[0] = m[0] * x + m[1] * y + m[2] * z;
  out[1] = m[3] * x + m[4] * y + m[5] * z;
  out[2] = m[6] * x + m[7] * y + m[8] * z;
  return out;
}

/** out = m^T * v */
export function mat3TVec(m: ArrayLike<number>, x: number, y: number, z: number, out: Vec3): Vec3 {
  out[0] = m[0] * x + m[3] * y + m[6] * z;
  out[1] = m[1] * x + m[4] * y + m[7] * z;
  out[2] = m[2] * x + m[5] * y + m[8] * z;
  return out;
}

/** Rodrigues: out = Exp([wx, wy, wz]) */
export function so3Exp(wx: number, wy: number, wz: number, out: Mat3): Mat3 {
  const th2 = wx * wx + wy * wy + wz * wz;
  let a: number, b: number;
  if (th2 < 1e-12) {
    a = 1 - th2 / 6;
    b = 0.5 - th2 / 24;
  } else {
    const th = Math.sqrt(th2);
    a = Math.sin(th) / th;
    b = (1 - Math.cos(th)) / th2;
  }
  out[0] = 1 + b * (wx * wx - th2);
  out[1] = -a * wz + b * wx * wy;
  out[2] = a * wy + b * wx * wz;
  out[3] = a * wz + b * wx * wy;
  out[4] = 1 + b * (wy * wy - th2);
  out[5] = -a * wx + b * wy * wz;
  out[6] = -a * wy + b * wx * wz;
  out[7] = a * wx + b * wy * wz;
  out[8] = 1 + b * (wz * wz - th2);
  return out;
}

export function so3Log(R: ArrayLike<number>, out: Vec3): Vec3 {
  const tr = R[0] + R[4] + R[8];
  let c = (tr - 1) * 0.5;
  if (c > 1) c = 1;
  else if (c < -1) c = -1;
  const th = Math.acos(c);
  const vx = R[7] - R[5], vy = R[2] - R[6], vz = R[3] - R[1];
  if (th < 1e-5) {
    const s = 0.5 * (1 + (th * th) / 6);
    out[0] = s * vx; out[1] = s * vy; out[2] = s * vz;
    return out;
  }
  if (Math.PI - th < 1e-3) {
    // Near pi: axis from the dominant column of (R + I) / 2 = a a^T.
    const d0 = R[0], d1 = R[4], d2 = R[8];
    let ax: number, ay: number, az: number;
    if (d0 >= d1 && d0 >= d2) {
      ax = d0 + 1; ay = (R[3] + R[1]) * 0.5; az = (R[6] + R[2]) * 0.5;
    } else if (d1 >= d2) {
      ax = (R[1] + R[3]) * 0.5; ay = d1 + 1; az = (R[7] + R[5]) * 0.5;
    } else {
      ax = (R[2] + R[6]) * 0.5; ay = (R[5] + R[7]) * 0.5; az = d2 + 1;
    }
    let n = Math.hypot(ax, ay, az);
    if (ax * vx + ay * vy + az * vz < 0) n = -n;
    out[0] = (ax / n) * th; out[1] = (ay / n) * th; out[2] = (az / n) * th;
    return out;
  }
  const s = th / (2 * Math.sin(th));
  out[0] = s * vx; out[1] = s * vy; out[2] = s * vz;
  return out;
}

export function quatFromMat3(R: ArrayLike<number>, out: Quat): Quat {
  const m00 = R[0], m11 = R[4], m22 = R[8];
  const tr = m00 + m11 + m22;
  let w: number, x: number, y: number, z: number;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (R[7] - R[5]) / s; x = 0.25 * s; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = 0.25 * s; z = (R[5] + R[7]) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = 0.25 * s;
  }
  const n = Math.hypot(w, x, y, z);
  const sgn = w < 0 ? -1 : 1;
  out[0] = (sgn * w) / n; out[1] = (sgn * x) / n; out[2] = (sgn * y) / n; out[3] = (sgn * z) / n;
  return out;
}

export function mat3FromQuat(q: ArrayLike<number>, out: Mat3): Mat3 {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y, wz = w * z;
  out[0] = 1 - 2 * (yy + zz); out[1] = 2 * (xy - wz); out[2] = 2 * (xz + wy);
  out[3] = 2 * (xy + wz); out[4] = 1 - 2 * (xx + zz); out[5] = 2 * (yz - wx);
  out[6] = 2 * (xz - wy); out[7] = 2 * (yz + wx); out[8] = 1 - 2 * (xx + yy);
  return out;
}

export function quatMul(a: ArrayLike<number>, b: ArrayLike<number>, out: Quat): Quat {
  const aw = a[0], ax = a[1], ay = a[2], az = a[3];
  const bw = b[0], bx = b[1], by = b[2], bz = b[3];
  out[0] = aw * bw - ax * bx - ay * by - az * bz;
  out[1] = aw * bx + ax * bw + ay * bz - az * by;
  out[2] = aw * by - ax * bz + ay * bw + az * bx;
  out[3] = aw * bz + ax * by - ay * bx + az * bw;
  return out;
}

export function quatNormalize(q: Quat): Quat {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  q[0] /= n; q[1] /= n; q[2] /= n; q[3] /= n;
  return q;
}

export function quatExp(wx: number, wy: number, wz: number, out: Quat): Quat {
  const th = Math.hypot(wx, wy, wz);
  const half = th * 0.5;
  const s = th < 1e-8 ? 0.5 - (th * th) / 48 : Math.sin(half) / th;
  out[0] = Math.cos(half); out[1] = wx * s; out[2] = wy * s; out[3] = wz * s;
  return out;
}

const _qd = quat();
/** q <- q * Exp(w) (right / body-frame increment). */
export function quatIntegrateRight(q: Quat, wx: number, wy: number, wz: number): Quat {
  quatExp(wx, wy, wz, _qd);
  quatMul(q, _qd, q);
  return quatNormalize(q);
}

/** q <- Exp(w) * q (left / global increment). */
export function quatIntegrateLeft(q: Quat, wx: number, wy: number, wz: number): Quat {
  quatExp(wx, wy, wz, _qd);
  quatMul(_qd, q, q);
  return quatNormalize(q);
}

const _ra = new Float64Array(9);
/** Angle in radians between two rotation matrices (atan2 form: exact for tiny angles). */
export function rotationAngle(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const M = mat3MulAtB(a, b, _ra);
  const s = 0.5 * Math.hypot(M[7] - M[5], M[2] - M[6], M[3] - M[1]);
  const c = (M[0] + M[4] + M[8] - 1) * 0.5;
  return Math.atan2(s, c);
}
