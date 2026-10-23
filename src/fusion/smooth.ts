// Render-side smoothing of pose corrections. The filter's pose for a frame
// can jump when a vision update corrects it (a re-anchor, a delayed update,
// a frame shown before its own result). Camera motion itself is smooth and
// known from the gyro, so the displayed pose carries the previous display
// forward by the camera motion between the two frames, then eases toward the
// filter's pose: sudden jumps spread over a few frames while motion is not
// delayed. Small corrections pass through untouched.
import { mat3, mat3Mul, mat3MulAtB, mat3TVec, mat3Vec, so3Exp, so3Log, vec3 } from '../math/rot';

export interface SmoothingOptions {
  /** Fraction of a sudden correction still pending after one frame (0 = off). */
  keep: number;
  /**
   * Corrections are measured in metres at the target (rotation counted at
   * 10 cm, depth at 30%: it only scales the overlay). Below `minError` they pass straight through (the ordinary
   * frame-to-frame refinement that keeps the overlay on the picture); from
   * there to `fullError` smoothing ramps up to `keep`; beyond `snapError`
   * (a re-acquisition) the new pose is taken at once.
   */
  minError: number;
  fullError: number;
  snapError: number;
  /** Frames further apart than this (s) restart from the filter's pose. */
  maxGap: number;
}

export const DEFAULT_SMOOTHING: SmoothingOptions = { keep: 0.8, minError: 0.002, fullError: 0.006, snapError: 0.15, maxGap: 0.25 };

export class PoseSmoother {
  readonly o: SmoothingOptions;
  private has = false;
  private time = 0;
  private readonly R = mat3();
  private readonly t = vec3();
  private readonly Ra = mat3();
  private readonly Rb = mat3();
  private readonly ta = vec3();
  private readonly tb = vec3();
  private readonly w = vec3();

  constructor(o: Partial<SmoothingOptions> = {}) {
    this.o = { ...DEFAULT_SMOOTHING, ...o };
  }

  reset(): void {
    this.has = false;
  }

  /** Time (s) of the previously displayed pose, or NaN if there is none. */
  get previousTime(): number {
    return this.has ? this.time : NaN;
  }

  /**
   * Displayed pose for the frame at time `time` (target in camera, R row-major).
   * (R, t): the filter's pose for this frame. (Rp, tp): the filter's pose, from
   * the same filter state, at previousTime; their difference is camera motion
   * only. Writes the smoothed pose back into (R, t).
   */
  update(time: number, R: Float64Array, t: Float64Array, Rp: ArrayLike<number> | null, tp: ArrayLike<number> | null): void {
    const o = this.o;
    if (!this.has || !Rp || !tp || o.keep <= 0 || !(time - this.time >= 0 && time - this.time <= o.maxGap)) {
      this.store(time, R, t);
      return;
    }
    // Carry the last display forward: P = T_k inv(T_prev) D_prev.
    // inv(T_prev) D_prev = (Rp^T R_D, Rp^T (t_D - tp)).
    const A = mat3MulAtB(Rp, this.R, this.Ra);
    const a = mat3TVec(Rp, this.t[0] - tp[0], this.t[1] - tp[1], this.t[2] - tp[2], this.ta);
    // Offset in target axes: O = inv(T_k) P = (A, a) since P = T_k (A, a).
    const th = so3Log(A, this.w);
    const e = Math.hypot(a[0], a[1], 0.3 * a[2]) + 0.1 * Math.hypot(th[0], th[1], th[2]);
    if (e > o.snapError) {
      this.store(time, R, t);
      return;
    }
    const k = o.keep * Math.min(1, Math.max(0, (e - o.minError) / (o.fullError - o.minError)));
    if (k === 0) {
      this.store(time, R, t);
      return;
    }
    const Ro = so3Exp(th[0] * k, th[1] * k, th[2] * k, this.Rb);
    const to = this.tb;
    to[0] = a[0] * k; to[1] = a[1] * k; to[2] = a[2] * k;
    // D = T_k O.
    const d = mat3Vec(R, to[0], to[1], to[2], this.ta);
    t[0] += d[0]; t[1] += d[1]; t[2] += d[2];
    mat3Mul(R, Ro, this.Ra);
    R.set(this.Ra);
    this.store(time, R, t);
  }

  private store(time: number, R: ArrayLike<number>, t: ArrayLike<number>): void {
    this.has = true;
    this.time = time;
    this.R.set(R);
    this.t.set(t);
  }
}
