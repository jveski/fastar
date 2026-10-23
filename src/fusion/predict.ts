// Camera-pose prediction shared by the filter (worker) and the render-side
// predictor (main thread), so both integrate the gyro identically.
import { ImuBuffer } from './imu';
import { quatIntegrateRight, mat3FromQuat, mat3MulAtB, mat3TVec, mat3, quat, vec3 } from '../math/rot';

/** Gyro rate at time t, linearly interpolated between samples (held at the ends). */
export function rateAt(imu: ImuBuffer, t: number, out: Float64Array): Float64Array {
  const i = imu.firstAfter(t);
  if (i === 0 || i >= imu.count) {
    const s = imu.slot(i === 0 ? 0 : imu.count - 1);
    out[0] = imu.w[3 * s]; out[1] = imu.w[3 * s + 1]; out[2] = imu.w[3 * s + 2];
    return out;
  }
  const a = imu.slot(i - 1), b = imu.slot(i);
  const ta = imu.t[a], tb = imu.t[b];
  const f = tb > ta ? (t - ta) / (tb - ta) : 0;
  for (let k = 0; k < 3; k++) out[k] = imu.w[3 * a + k] + f * (imu.w[3 * b + k] - imu.w[3 * a + k]);
  return out;
}

/**
 * Integrate camera orientation q (world-from-camera) and position p from t0
 * to t1 with trapezoidal gyro rates. w0 is the raw rate at t0 and is updated
 * to the rate at t1. Without an IMU the raw rate is 0 (so -b acts as the
 * angular velocity, as in the constant-velocity filter).
 */
export function integrateCamera(
  q: Float64Array, p: Float64Array, v: ArrayLike<number>, b: ArrayLike<number>,
  t0: number, w0: Float64Array, t1: number, imu: ImuBuffer | null, wTmp: Float64Array,
): void {
  if (t1 === t0) return;
  if (t1 < t0) {
    // Short step back from a newer state (one trapezoid; used within one IMU interval).
    const dt = t1 - t0;
    const w = imu && imu.count > 0 ? rateAt(imu, t1, wTmp) : wTmp.fill(0);
    quatIntegrateRight(q, (0.5 * (w0[0] + w[0]) - b[0]) * dt, (0.5 * (w0[1] + w[1]) - b[1]) * dt, (0.5 * (w0[2] + w[2]) - b[2]) * dt);
    p[0] += v[0] * dt; p[1] += v[1] * dt; p[2] += v[2] * dt;
    w0[0] = w[0]; w0[1] = w[1]; w0[2] = w[2];
    return;
  }
  const b0 = b[0], b1 = b[1], b2 = b[2];
  let tc = t0;
  if (imu) {
    for (let i = imu.firstAfter(t0); i < imu.count; i++) {
      const s = imu.slot(i);
      const ts = imu.t[s];
      if (ts > t1) break;
      const dt = ts - tc;
      const wx = imu.w[3 * s], wy = imu.w[3 * s + 1], wz = imu.w[3 * s + 2];
      quatIntegrateRight(q, (0.5 * (w0[0] + wx) - b0) * dt, (0.5 * (w0[1] + wy) - b1) * dt, (0.5 * (w0[2] + wz) - b2) * dt);
      p[0] += v[0] * dt; p[1] += v[1] * dt; p[2] += v[2] * dt;
      w0[0] = wx; w0[1] = wy; w0[2] = wz;
      tc = ts;
    }
  }
  if (t1 > tc) {
    const dt = t1 - tc;
    const w = imu && imu.count > 0 ? rateAt(imu, t1, wTmp) : wTmp.fill(0);
    quatIntegrateRight(q, (0.5 * (w0[0] + w[0]) - b0) * dt, (0.5 * (w0[1] + w[1]) - b1) * dt, (0.5 * (w0[2] + w[2]) - b2) * dt);
    p[0] += v[0] * dt; p[1] += v[1] * dt; p[2] += v[2] * dt;
    w0[0] = w[0]; w0[1] = w[1]; w0[2] = w[2];
  }
}

// Snapshot layout (Float64Array):
//   [0] n entries, [1] t_d, [2..5] q_wt, [6..8] p_wt, [9..11] raw rate at the last entry,
//   then n x [t, q_wc(4), p_wc(3), v_wc(3), b_g(3)]  (14 values each, oldest first)
export const SNAP_HEADER = 12;
/** Max gap (s) bridged backwards from a newer history entry. */
export const NEAR_ENTRY = 0.004;
export const SNAP_ENTRY = 14;

/**
 * Render-side predictor: holds the latest filter snapshot from the worker
 * and answers poseAt(captureTime) using the main thread's own gyro buffer.
 */
export class PosePredictor {
  private snap: Float64Array | null = null;
  private readonly q = quat();
  private readonly p = vec3();
  private readonly w0 = vec3();
  private readonly wTmp = vec3();
  private readonly Rwc = mat3();
  private readonly Rwt = mat3();
  maxExtrapolate = 0.1;

  ingest(snap: Float64Array): void {
    this.snap = snap;
  }

  get ready(): boolean {
    return !!this.snap && this.snap[0] > 0;
  }

  get td(): number {
    return this.snap ? this.snap[1] : 0;
  }

  /** Target-in-camera pose for a frame captured at camera time tCam (s). */
  poseAt(tCam: number, imu: ImuBuffer | null, R: Float64Array, t: Float64Array): boolean {
    const s = this.snap;
    if (!s || s[0] < 1) return false;
    const n = s[0];
    const tau = tCam + s[1];
    // Newest entry with time <= tau (or the oldest), or the next one if that
    // is closer: a vision update that also nudged t_d leaves its post-update
    // entry just after tau, and the one before it is the bare prediction.
    let k = 0;
    for (let i = n - 1; i >= 0; i--) {
      if (s[SNAP_HEADER + i * SNAP_ENTRY] <= tau) { k = i; break; }
    }
    if (k < n - 1) {
      const ta = s[SNAP_HEADER + k * SNAP_ENTRY], tb = s[SNAP_HEADER + (k + 1) * SNAP_ENTRY];
      if (tb - tau < tau - ta && tb - tau < NEAR_ENTRY) k++;
    }
    const o = SNAP_HEADER + k * SNAP_ENTRY;
    const t0 = s[o];
    this.q.set(s.subarray(o + 1, o + 5));
    this.p.set(s.subarray(o + 5, o + 8));
    const v = s.subarray(o + 8, o + 11), b = s.subarray(o + 11, o + 14);
    const useImu = imu && imu.count > 0 && imu.oldestTime <= t0 ? imu : null;
    if (k === n - 1) this.w0.set(s.subarray(9, 12));
    else if (useImu) rateAt(useImu, t0, this.w0);
    else this.w0.fill(0);
    const tLast = s[SNAP_HEADER + (n - 1) * SNAP_ENTRY];
    // Before the oldest entry: hold it (as the filter does); step back only from a near newer entry.
    const tEnd = tau < t0 && t0 - tau >= NEAR_ENTRY ? t0 : Math.min(tau, Math.max(t0, tLast) + this.maxExtrapolate);
    integrateCamera(this.q, this.p, v, b, t0, this.w0, tEnd, useImu, this.wTmp);
    const Rwc = mat3FromQuat(this.q, this.Rwc);
    const Rwt = mat3FromQuat(s.subarray(2, 6), this.Rwt);
    mat3MulAtB(Rwc, Rwt, R);
    mat3TVec(Rwc, s[6] - this.p[0], s[7] - this.p[1], s[8] - this.p[2], t);
    return true;
  }
}
