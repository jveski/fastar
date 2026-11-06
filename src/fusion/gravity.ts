// Which way the accelerometer's axes point, worked out from the data.
// Platforms disagree on the sign of devicemotion's
// accelerationIncludingGravity, and axis corrections learned for the gyro
// need not apply to it; the filter never notices (it only needs gravity as a
// consistent reference), but anything showing gravity to the user does. Two
// checks:
//   - the gravity it reads must turn exactly as the gyro says the camera
//     turns: that fixes each axis's sign, up to all three at once;
//   - in AR the camera looks level or down with the picture upright, so
//     "up" points toward the top of the image and/or behind the camera:
//     that fixes the overall sign. (Pointing up past 45 degrees fools it.)
import type { ImuBuffer } from './imu';
import { mat3FromQuat, quatIntegrateRight } from '../math/rot';

/** Candidate sign patterns, up to an overall sign: none, or one axis flipped. */
const PATTERNS = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]];
/** Gap (s) between the two readings compared, and the least rotation between them that counts (rad). */
const SPAN = 0.15;
const MIN_TURN = 0.05;
/** Readings farther than this from 1 g (m/s^2) are not taken for gravity alone. */
const STILL = 1.5;
const G = 9.81;
/** Turning (rad) and upright readings needed before a decision. */
const TURN_EVIDENCE = 1.5;
const UPRIGHT_EVIDENCE = 30;

export class AccelSigns {
  /** Multiply accelerometer axes (camera axes, as stored) by these. */
  readonly signs = Float64Array.of(1, 1, 1);
  private readonly err = new Float64Array(PATTERNS.length);
  private readonly upright = new Float64Array(PATTERNS.length);
  private uprightCount = 0;
  private turned = 0;
  private pattern = 0;
  private lastTime = -Infinity;
  private readonly q = new Float64Array(4);
  private readonly R = new Float64Array(9);

  reset(): void {
    this.signs.fill(1);
    this.err.fill(0);
    this.upright.fill(0);
    this.uprightCount = 0;
    this.turned = 0;
    this.pattern = 0;
    this.lastTime = -Infinity;
  }

  /** Take in the samples that arrived since the last call. */
  observe(imu: ImuBuffer): void {
    for (let i = imu.firstAfter(this.lastTime); i < imu.count; i++) {
      const s1 = imu.slot(i), t1 = imu.t[s1];
      if (t1 - this.lastTime < 0.03) continue;
      this.lastTime = t1;
      if (!imu.hasAccel[s1]) continue;
      const a1x = imu.a[3 * s1], a1y = imu.a[3 * s1 + 1], a1z = imu.a[3 * s1 + 2];
      const m1 = Math.sqrt(a1x * a1x + a1y * a1y + a1z * a1z);
      if (Math.abs(m1 - G) > STILL) continue;
      // Upright check: "up" toward the image top (-y) and/or behind the camera (-z).
      for (let p = 0; p < PATTERNS.length; p++) {
        const P = PATTERNS[p];
        this.upright[p] -= (P[1] * a1y + P[2] * a1z) / m1;
      }
      this.uprightCount++;
      // Turning check against the reading SPAN earlier.
      const i0 = imu.firstAfter(t1 - SPAN) - 1;
      if (i0 < 0) continue;
      const s0 = imu.slot(i0);
      if (!imu.hasAccel[s0]) continue;
      const a0x = imu.a[3 * s0], a0y = imu.a[3 * s0 + 1], a0z = imu.a[3 * s0 + 2];
      const m0 = Math.sqrt(a0x * a0x + a0y * a0y + a0z * a0z);
      if (Math.abs(m0 - G) > STILL) continue;
      // Camera rotation from t0 to t1 (camera-t1 vectors to camera-t0 axes).
      const q = this.q;
      q[0] = 1; q[1] = q[2] = q[3] = 0;
      let turn = 0;
      for (let k = i0 + 1; k <= i; k++) {
        const sa = imu.slot(k - 1), sb = imu.slot(k), dt = imu.t[sb] - imu.t[sa];
        const wx = 0.5 * (imu.w[3 * sa] + imu.w[3 * sb]) * dt, wy = 0.5 * (imu.w[3 * sa + 1] + imu.w[3 * sb + 1]) * dt, wz = 0.5 * (imu.w[3 * sa + 2] + imu.w[3 * sb + 2]) * dt;
        quatIntegrateRight(q, wx, wy, wz);
        turn += Math.sqrt(wx * wx + wy * wy + wz * wz);
      }
      if (turn < MIN_TURN) continue;
      const R = mat3FromQuat(q, this.R);
      // A fixed world direction seen at t0 as u0 is seen at t1 as R^T u0.
      for (let p = 0; p < PATTERNS.length; p++) {
        const P = PATTERNS[p];
        const ux = (P[0] * a0x) / m0, uy = (P[1] * a0y) / m0, uz = (P[2] * a0z) / m0;
        const px = R[0] * ux + R[3] * uy + R[6] * uz, py = R[1] * ux + R[4] * uy + R[7] * uz, pz = R[2] * ux + R[5] * uy + R[8] * uz;
        const dx = px - (P[0] * a1x) / m1, dy = py - (P[1] * a1y) / m1, dz = pz - (P[2] * a1z) / m1;
        this.err[p] += dx * dx + dy * dy + dz * dz;
      }
      this.turned += turn;
    }
    this.decide();
  }

  private decide(): void {
    if (this.turned >= TURN_EVIDENCE) {
      let best = 0, second = Infinity;
      for (let p = 1; p < PATTERNS.length; p++) if (this.err[p] < this.err[best]) best = p;
      for (let p = 0; p < PATTERNS.length; p++) if (p !== best && this.err[p] < second) second = this.err[p];
      if (this.err[best] < 0.5 * second) this.pattern = best;
    }
    const P = PATTERNS[this.pattern];
    const flip = this.uprightCount >= UPRIGHT_EVIDENCE && this.upright[this.pattern] < 0 ? -1 : 1;
    for (let k = 0; k < 3; k++) this.signs[k] = flip * P[k];
  }
}
