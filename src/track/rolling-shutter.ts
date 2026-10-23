// Online rolling-shutter readout estimate. Phone cameras expose rows one
// after another, so while the phone turns the target is sheared by
// (angular velocity x readout time). The tracker removes that with the gyro
// (TargetTracker.deskew) once the readout is known; this estimates it.
//
// Per tracked frame with enough rotation, the pose is re-solved with the
// readout nudged up and down along each image axis. The change in the robust
// reprojection cost gives a local quadratic in the readout, and those are
// accumulated (with slow forgetting) into a running minimizer. Both axes are
// estimated because portrait video reads out along x and landscape along y,
// in a direction that depends on the device.
import type { TargetTracker } from './tracker';
import { PoseSolver } from '../vision/pose';

export interface RollingShutterOptions {
  /** Least angular speed (rad/s) for a frame to say anything about the readout. */
  minRate: number;
  /** Least inliers for a frame to be used. */
  minInliers: number;
  /** Readout probe step (s). */
  step: number;
  /** Readout is clamped to +-this (s). */
  maxReadout: number;
  /** Per-frame forgetting factor of the accumulated evidence. */
  forget: number;
  /** Frames of evidence before the estimate is used. */
  minFrames: number;
  /** Residuals are clipped at this (px) in the cost. */
  clip: number;
}

export const DEFAULT_ROLLING_SHUTTER: RollingShutterOptions = {
  minRate: 0.12,
  minInliers: 30,
  step: 0.006,
  maxReadout: 0.05,
  forget: 0.995,
  minFrames: 20,
  clip: 2,
};

export class RollingShutterEstimator {
  readonly o: RollingShutterOptions;
  /** Current estimate (s), see TargetTracker.rsX / rsY. */
  rsX = 0;
  rsY = 0;
  frames = 0;
  // Accumulated quadratic per axis: sum h, sum (h s - g).
  private readonly A = new Float64Array(2);
  private readonly B = new Float64Array(2);
  private solver: PoseSolver;
  private obj: Float32Array;
  private img: Float32Array;
  private readonly R = new Float64Array(9);
  private readonly t = new Float64Array(3);

  constructor(opts: Partial<RollingShutterOptions> = {}) {
    this.o = { ...DEFAULT_ROLLING_SHUTTER, ...opts };
    this.solver = new PoseSolver(256);
    this.obj = new Float32Array(512);
    this.img = new Float32Array(512);
  }

  get ready(): boolean {
    return this.frames >= this.o.minFrames;
  }

  reset(): void {
    this.A.fill(0);
    this.B.fill(0);
    this.frames = 0;
    this.rsX = this.rsY = 0;
  }

  /** Use a frame the tracker has just solved (its omega must be set). */
  observe(tr: TargetTracker): void {
    const w = tr.omega, o = this.o;
    if (Math.hypot(w[0], w[1], w[2]) < o.minRate || !tr.ok || tr.inliers < o.minInliers) return;
    const n = tr.n;
    if (this.obj.length < 2 * n) {
      this.obj = new Float32Array(2 * n);
      this.img = new Float32Array(2 * n);
      this.solver = new PoseSolver(n);
    }
    for (let i = 0; i < n; i++) {
      const j = tr.idx[i];
      this.obj[2 * i] = tr.obj[2 * j];
      this.obj[2 * i + 1] = tr.obj[2 * j + 1];
    }
    const sx = this.rsX, sy = this.rsY, d = o.step;
    const c0 = this.cost(tr, sx, sy);
    if (!Number.isFinite(c0)) return;
    let used = false;
    for (let a = 0; a < 2; a++) {
      const cp = this.cost(tr, a === 0 ? sx + d : sx, a === 0 ? sy : sy + d);
      const cm = this.cost(tr, a === 0 ? sx - d : sx, a === 0 ? sy : sy - d);
      if (!Number.isFinite(cp) || !Number.isFinite(cm)) continue;
      const h = (cp - 2 * c0 + cm) / (d * d);
      if (!(h > 0)) continue;
      const g = (cp - cm) / (2 * d);
      const s = a === 0 ? sx : sy;
      this.A[a] = this.A[a] * o.forget + h;
      this.B[a] = this.B[a] * o.forget + h * s - g;
      used = true;
    }
    if (!used) return;
    this.frames++;
    const m = o.maxReadout;
    if (this.A[0] > 0) this.rsX = Math.max(-m, Math.min(m, this.B[0] / this.A[0]));
    if (this.A[1] > 0) this.rsY = Math.max(-m, Math.min(m, this.B[1] / this.A[1]));
  }

  /** Robust reprojection cost after re-solving the pose with readout (rx, ry). */
  private cost(tr: TargetTracker, rx: number, ry: number): number {
    const n = tr.n;
    tr.deskew(tr.pos, this.img, n, rx, ry);
    this.R.set(tr.R);
    this.t.set(tr.t);
    if (!this.solver.solve(tr.K, this.obj, this.img, n, this.R, this.t)) return NaN;
    const r = this.solver.residuals, c2 = this.o.clip * this.o.clip;
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.min(r[i] * r[i], c2);
    return s;
  }
}
