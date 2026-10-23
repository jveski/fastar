import type { Pyramid } from '../image/pyramid';
import { createGray, type GrayImage } from '../image/gray';
import type { TargetData } from '../target/format';
import type { Intrinsics } from '../vision/camera';
import { Klt } from '../vision/klt';
import { NccMatcher } from '../vision/ncc';
import { PoseSolver, poseToHomography } from '../vision/pose';
import { mat3Invert } from '../math/rot';

export interface TrackerOptions {
  /** Desired number of active points. */
  maxPoints: number;
  /** Drift-corrected points per frame (rotating subset). */
  nccPerFrame: number;
  /** NCC acceptance threshold. */
  minNcc: number;
  /**
   * Tracked points are dropped only below this NCC (occluded, or KLT slid
   * onto something else). Between it and minNcc (typically motion blur or a
   * screen's moire) the KLT position is kept, just not corrected.
   */
  dropNcc: number;
  /** Below this many inliers the target is lost. */
  minInliers: number;
  /** Max replenish attempts per frame. */
  replenishAttempts: number;
  /** Keep points this far from the frame border (px). */
  margin: number;
  /** Forward-backward KLT threshold (px). */
  fbThreshold: number;
}

export const DEFAULT_TRACKER: TrackerOptions = {
  maxPoints: 150,
  nccPerFrame: 50,
  minNcc: 0.7,
  dropNcc: 0.4,
  minInliers: 8,
  replenishAttempts: 40,
  margin: 12,
  fbThreshold: 1.5,
};

export interface TrackTimings {
  klt: number;
  solve: number;
  ncc: number;
  replenish: number;
}

/**
 * Frame-to-frame tracker for one target: gyro-seeded pyramidal KLT on target
 * points, robust pose, NCC drift correction against the target image on a
 * rotating subset, and replenishment from the target's tracking points.
 */
export class TargetTracker {
  readonly o: TrackerOptions;
  /** Tracking points in target metres. */
  readonly obj: Float32Array;
  readonly numTrackPoints: number;
  private readonly images: GrayImage[];
  private readonly imgPixelM: number[]; // metres per image pixel
  // Active set.
  readonly idx: Int32Array;
  readonly pos: Float32Array;
  n = 0;
  private readonly active: Uint8Array;
  // Pose.
  readonly R = new Float64Array(9);
  readonly t = new Float64Array(3);
  readonly cov = new Float64Array(36);
  inliers = 0;
  rms = 0;
  ok = false;
  readonly timings: TrackTimings = { klt: 0, solve: 0, ncc: 0, replenish: 0 };
  /**
   * Rolling shutter: a pixel (u, v) is exposed rsX * (u - cx) / width +
   * rsY * (v - cy) / height seconds after the frame time (readout along x in
   * portrait video, y in landscape; negative when reversed).
   */
  rsX = 0;
  rsY = 0;
  /** Camera angular (rad/s) and linear (m/s) velocity in camera axes during this frame, for de-skewing. */
  readonly omega = new Float64Array(3);
  readonly velocity = new Float64Array(3);

  private readonly klt = new Klt();
  private readonly ncc = new NccMatcher(4, 4);
  private readonly solver: PoseSolver;
  private readonly guess: Float32Array;
  private readonly prevPts: Float32Array;
  private readonly status: Uint8Array;
  private readonly sObj: Float32Array;
  private readonly sImg: Float32Array;
  private readonly H = new Float64Array(9);
  private readonly Hinv = new Float64Array(9);
  private readonly p2 = new Float32Array(2);
  private nccCursor = 0;
  private replCursor = 0;

  constructor(readonly target: TargetData, public K: Intrinsics, opts: Partial<TrackerOptions> = {}) {
    this.o = { ...DEFAULT_TRACKER, ...opts };
    const m = target.meta;
    const s = m.physicalWidth / m.width;
    const cu = (m.width - 1) / 2, cv = (m.height - 1) / 2;
    this.numTrackPoints = target.trackPoints.length / 2;
    this.obj = new Float32Array(2 * this.numTrackPoints);
    for (let i = 0; i < this.numTrackPoints; i++) {
      this.obj[2 * i] = (target.trackPoints[2 * i] - cu) * s;
      this.obj[2 * i + 1] = (target.trackPoints[2 * i + 1] - cv) * s;
    }
    this.images = m.images.map((im, k) => createGray(im.width, im.height, target.images[k]));
    this.imgPixelM = m.images.map((im) => m.physicalWidth / im.width);
    const cap = Math.max(this.o.maxPoints, this.numTrackPoints);
    this.idx = new Int32Array(cap);
    this.pos = new Float32Array(2 * cap);
    this.active = new Uint8Array(this.numTrackPoints);
    this.guess = new Float32Array(2 * cap);
    this.prevPts = new Float32Array(2 * cap);
    this.status = new Uint8Array(cap);
    this.sObj = new Float32Array(2 * cap);
    this.sImg = new Float32Array(2 * cap);
    this.solver = new PoseSolver(cap);
  }

  reset(): void {
    this.n = 0;
    this.active.fill(0);
    this.ok = false;
  }

  /**
   * Seed points in `frame` from a known pose (acquisition or coasting
   * prediction) using NCC against the target image. Returns success.
   */
  init(frame: Pyramid, R: ArrayLike<number>, t: ArrayLike<number>, maxAttempts = this.numTrackPoints, minPoints = this.o.minInliers + 4): boolean {
    this.reset();
    for (let i = 0; i < 9; i++) this.R[i] = R[i];
    for (let i = 0; i < 3; i++) this.t[i] = t[i];
    this.replCursor = 0;
    this.replenish(frame.levels[0], maxAttempts, this.o.maxPoints);
    if (this.n < minPoints) { this.reset(); return false; }
    this.ok = this.solve(this.R, this.t);
    if (this.ok) this.removeOutliers();
    return this.ok && this.inliers >= this.o.minInliers;
  }

  /** Track from `prev` to `cur`. (predR, predT) is the predicted pose at `cur` (e.g. gyro-propagated). */
  track(prev: Pyramid, cur: Pyramid, predR: ArrayLike<number>, predT: ArrayLike<number>): boolean {
    const o = this.o, K = this.K;
    const n = this.n;
    const tm = this.timings;
    let t0 = now();
    // 1. Gyro-seeded KLT with forward-backward check.
    const guess = this.guess, prevPts = this.prevPts, status = this.status;
    for (let i = 0; i < n; i++) {
      prevPts[2 * i] = this.pos[2 * i];
      prevPts[2 * i + 1] = this.pos[2 * i + 1];
      const j = this.idx[i];
      if (!project(K, predR, predT, this.obj[2 * j], this.obj[2 * j + 1], guess, 2 * i)) {
        guess[2 * i] = prevPts[2 * i];
        guess[2 * i + 1] = prevPts[2 * i + 1];
      }
    }
    this.klt.trackBatch(prev, cur, prevPts, guess, n, status, o.fbThreshold);
    // Compact survivors that stay inside the frame.
    let m = 0;
    const W = K.width - o.margin, Hh = K.height - o.margin;
    for (let i = 0; i < n; i++) {
      const x = guess[2 * i], y = guess[2 * i + 1];
      if (!status[i] || x < o.margin || y < o.margin || x >= W || y >= Hh) {
        this.active[this.idx[i]] = 0;
        continue;
      }
      this.idx[m] = this.idx[i];
      this.pos[2 * m] = x;
      this.pos[2 * m + 1] = y;
      m++;
    }
    this.n = m;
    let t1 = now();
    tm.klt = t1 - t0;
    // 2. Robust pose from the prediction.
    for (let i = 0; i < 9; i++) this.R[i] = predR[i];
    for (let i = 0; i < 3; i++) this.t[i] = predT[i];
    if (m < o.minInliers || !this.solve(this.R, this.t)) return this.fail();
    this.removeOutliers();
    t0 = now();
    tm.solve = t0 - t1;
    // 3. NCC drift correction on a rotating subset.
    const corrected = this.driftCorrect(cur.levels[0]);
    t1 = now();
    tm.ncc = t1 - t0;
    if (corrected > 0) {
      if (this.n < o.minInliers || !this.solve(this.R, this.t)) return this.fail();
      this.removeOutliers();
    }
    t0 = now();
    tm.solve += t0 - t1;
    // 4. Replenish lost points from the target's tracking points.
    if (this.n < o.maxPoints) this.replenish(cur.levels[0], o.replenishAttempts, o.maxPoints);
    tm.replenish = now() - t0;
    if (this.inliers < o.minInliers) return this.fail();
    this.ok = true;
    return true;
  }

  private fail(): boolean {
    this.ok = false;
    this.inliers = 0;
    return false;
  }

  /** Solve the pose from active points; fills inliers, cov, rms. */
  private solve(R: Float64Array, t: Float64Array): boolean {
    const n = this.n;
    for (let i = 0; i < n; i++) {
      const j = this.idx[i];
      this.sObj[2 * i] = this.obj[2 * j];
      this.sObj[2 * i + 1] = this.obj[2 * j + 1];
    }
    this.deskew(this.pos, this.sImg, n, this.rsX, this.rsY);
    if (!this.solver.solve(this.K, this.sObj, this.sImg, n, R, t)) return false;
    this.inliers = this.solver.inliers;
    this.rms = this.solver.rmsInlier;
    this.cov.set(this.solver.cov);
    return t[2] > 0;
  }

  /**
   * Move rolling-shutter observations to where they would be at the frame
   * time: each point moves with its image velocity (from the camera's angular
   * and linear velocity and the point's depth at the current pose) over its
   * exposure offset.
   */
  deskew(src: Float32Array, dst: Float32Array, n: number, rsX: number, rsY: number): void {
    const K = this.K, w = this.omega, v = this.velocity, R = this.R, t = this.t;
    if (rsX === 0 && rsY === 0) {
      dst.set(src.subarray(0, 2 * n));
      return;
    }
    const ax = rsX / K.width, ay = rsY / K.height;
    for (let i = 0; i < n; i++) {
      const u = src[2 * i], vv = src[2 * i + 1];
      const dt = ax * (u - K.cx) + ay * (vv - K.cy);
      const j = this.idx[i];
      const X = this.obj[2 * j], Y = this.obj[2 * j + 1];
      const px = R[0] * X + R[1] * Y + t[0], py = R[3] * X + R[4] * Y + t[1], pz = R[6] * X + R[7] * Y + t[2];
      if (pz <= 1e-6) { dst[2 * i] = u; dst[2 * i + 1] = vv; continue; }
      // Point velocity in camera axes: -(w x P + v).
      const qx = -(w[1] * pz - w[2] * py + v[0]);
      const qy = -(w[2] * px - w[0] * pz + v[1]);
      const qz = -(w[0] * py - w[1] * px + v[2]);
      const iz = 1 / pz;
      dst[2 * i] = u - dt * K.fx * (qx - px * iz * qz) * iz;
      dst[2 * i + 1] = vv - dt * K.fy * (qy - py * iz * qz) * iz;
    }
  }

  private removeOutliers(): void {
    const w = this.solver.weights;
    let m = 0;
    for (let i = 0; i < this.n; i++) {
      if (w[i] <= 0) { this.active[this.idx[i]] = 0; continue; }
      this.idx[m] = this.idx[i];
      this.pos[2 * m] = this.pos[2 * i];
      this.pos[2 * m + 1] = this.pos[2 * i + 1];
      m++;
    }
    this.n = m;
  }

  /** Pick the target image whose pixels are closest to one frame pixel. */
  private pickImage(): number {
    const pxPerM = this.K.fx / Math.max(1e-3, this.t[2]);
    let best = 0, bestErr = Infinity;
    for (let k = 0; k < this.images.length; k++) {
      const e = Math.abs(Math.log(this.imgPixelM[k] * pxPerM));
      if (e < bestErr) { bestErr = e; best = k; }
    }
    return best;
  }

  /** Set up H (target image k px -> frame px) and its inverse for the current pose. */
  private setupWarp(k: number): boolean {
    const im = this.images[k];
    poseToHomography(this.K, this.R, this.t, this.imgPixelM[k], (im.width - 1) / 2, (im.height - 1) / 2, this.H);
    return mat3Invert(this.H, this.Hinv);
  }

  private driftCorrect(frame: GrayImage): number {
    const n = this.n;
    if (n === 0) return 0;
    const k = this.pickImage();
    if (!this.setupWarp(k)) return 0;
    const img = this.images[k];
    const cnt = Math.min(this.o.nccPerFrame, n);
    let corrected = 0;
    const drop = this.status; // reuse as a drop mask
    drop.fill(0, 0, n);
    for (let c = 0; c < cnt; c++) {
      const i = (this.nccCursor + c) % n;
      const j = this.idx[i];
      if (!project(this.K, this.R, this.t, this.obj[2 * j], this.obj[2 * j + 1], this.p2, 0)) continue;
      const ncc = this.ncc;
      if (!ncc.buildTemplate(img, this.Hinv, this.p2[0], this.p2[1])) continue;
      if (ncc.match(frame, this.o.minNcc)) {
        this.pos[2 * i] = ncc.x;
        this.pos[2 * i + 1] = ncc.y;
        corrected++;
      } else if (ncc.score < this.o.dropNcc) {
        drop[i] = 1; // occluded or drifted onto something else
      }
    }
    this.nccCursor = (this.nccCursor + cnt) % Math.max(1, n);
    // Remove dropped points.
    let m = 0;
    for (let i = 0; i < n; i++) {
      if (drop[i]) { this.active[this.idx[i]] = 0; continue; }
      this.idx[m] = this.idx[i];
      this.pos[2 * m] = this.pos[2 * i];
      this.pos[2 * m + 1] = this.pos[2 * i + 1];
      m++;
    }
    this.n = m;
    return corrected;
  }

  /** Add inactive tracking points that project into the frame, located by NCC. */
  private replenish(frame: GrayImage, attempts: number, maxPoints: number): void {
    const total = this.numTrackPoints;
    if (total === 0) return;
    const k = this.pickImage();
    if (!this.setupWarp(k)) return;
    const img = this.images[k];
    const o = this.o, K = this.K;
    let tried = 0, c = 0;
    for (; c < total && tried < attempts && this.n < maxPoints; c++) {
      const j = (this.replCursor + c) % total;
      if (this.active[j]) continue;
      if (!project(K, this.R, this.t, this.obj[2 * j], this.obj[2 * j + 1], this.p2, 0)) continue;
      const x = this.p2[0], y = this.p2[1];
      if (x < o.margin || y < o.margin || x >= K.width - o.margin || y >= K.height - o.margin) continue;
      tried++;
      if (!this.ncc.buildTemplate(img, this.Hinv, x, y) || !this.ncc.match(frame, o.minNcc)) continue;
      this.idx[this.n] = j;
      this.pos[2 * this.n] = this.ncc.x;
      this.pos[2 * this.n + 1] = this.ncc.y;
      this.active[j] = 1;
      this.n++;
    }
    this.replCursor = (this.replCursor + Math.max(1, c)) % total;
  }
}

function project(K: Intrinsics, R: ArrayLike<number>, t: ArrayLike<number>, X: number, Y: number, out: Float32Array, o: number): boolean {
  const z = R[6] * X + R[7] * Y + t[2];
  if (z <= 1e-6) return false;
  out[o] = (K.fx * (R[0] * X + R[1] * Y + t[0])) / z + K.cx;
  out[o + 1] = (K.fy * (R[3] * X + R[4] * Y + t[1])) / z + K.cy;
  return true;
}

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
