import type { Pyramid } from '../image/pyramid';
import { createGray, type GrayImage } from '../image/gray';
import { VPT_STATIONARY, type TargetData } from '../target/format';
import { VideoModel } from '../target/video';
import type { Intrinsics } from '../vision/camera';
import { Klt } from '../vision/klt';
import { NccMatcher } from '../vision/ncc';
import { DEFAULT_POSE_OPTIONS, PoseSolver, poseToHomography } from '../vision/pose';
import { ippe } from '../vision/ippe';
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

/** Video time for a frame, in video frame coordinates (video time * fps). */
export interface VideoWindow {
  /** Best estimate. */
  x: number;
  /** Credible interval: points must stay on screen (and still, unless locked) across it. */
  lo: number;
  hi: number;
  /** The estimate is good to a fraction of a frame: moving points are usable. */
  locked: boolean;
}

/** Whether coarse re-alignment also tries the other planar tilt (see coarseAlign). */
export const Flips = { None: 0, Free: 1, Careful: 2 } as const;
export type Flips = (typeof Flips)[keyof typeof Flips];

/** Coarse re-alignment: pyramid level, search radius there (px), NCC threshold, points tried. */
const COARSE_LEVEL = 2;
const COARSE_SEARCH = 10;
const COARSE_NCC = 0.7;
const COARSE_ATTEMPTS = 120;
/** Points tried per candidate time when searching the time and pose together. */
const SEARCH_ATTEMPTS = 60;
/** Points checked at full resolution to choose between candidate poses. */
const FINE_SAMPLE = 32;
/** Most rounds of that search (each from the best pose of the last). */
const SEARCH_ROUNDS = 3;
/**
 * Time evidence from the tracked points: log-likelihood per px^2 of
 * (capped) squared residual, the cap (px), and the most one frame's
 * evidence may move from the mean (the points' errors are not independent).
 */
const GEO_GAIN = 0.5;
const GEO_RES_CAP = 3;
const GEO_CAP = 6;
/** Frames of starved replenishment before looking for a better pose. */
const DOUBT_FRAMES = 4;

/**
 * Log-likelihood of a probe's NCC: above PROBE_MID favours the predicted
 * content being on screen, below it the opposite; capped so one occluded or
 * blurred probe cannot outvote the rest.
 */
const PROBE_MID = 0.6;
const PROBE_GAIN = 3;
function probeLlr(ncc: number): number {
  const v = PROBE_GAIN * (ncc - PROBE_MID);
  return v > 1.2 ? 1.2 : v < -1.2 ? -1.2 : v;
}

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

  /** Video targets: baked points over time, and which points are usable at the current time. */
  readonly video: VideoModel | null;
  readonly usable: Uint8Array;
  /** Video frame coordinate the point positions are set for. */
  frameX = 0;
  private readonly metresPerPx: number;
  private readonly Hc = new Float64Array(9);
  private readonly pb = new Float64Array(2);
  private readonly probeNcc = new NccMatcher(4, 1);
  private probeCursor = 0;
  private probeFrame: GrayImage | null = null;
  private readonly probeSpans = new Int32Array(64);
  private probeMark = new Uint8Array(0);
  private probeVals = new Float64Array(256);
  private probeSel = new Int32Array(24);
  /** Per-candidate poses while probing the time. */
  private readonly timeSolver: PoseSolver;
  private readonly Rp = new Float64Array(9);
  private readonly tp = new Float64Array(3);
  private tImg = new Float32Array(0);
  private geoCost = new Float64Array(0);
  /** Cost of the last poseForTime fit (NaN if it kept the current pose). */
  private fitCost = NaN;
  /** Wide, coarse re-alignment (video). */
  private readonly coarseNcc = new NccMatcher(4, COARSE_SEARCH);
  private readonly cObj: Float32Array;
  private readonly cImg: Float32Array;
  private coarseCursor = 0;
  private readonly cNorm: Float64Array;
  private readonly Rbest = new Float64Array(9);
  private readonly Rs0 = new Float64Array(9);
  private readonly ts0 = new Float64Array(3);
  private readonly tbest = new Float64Array(3);
  private readonly Rq = new Float64Array(9);
  private readonly tq = new Float64Array(3);
  private doubt = 0;
  /** Video: replenishing has kept failing; the time and pose are suspect (see checkAlignment). */
  starved = false;
  private replTried = 0;
  private replFound = 0;

  constructor(readonly target: TargetData, public K: Intrinsics, opts: Partial<TrackerOptions> = {}) {
    this.o = { ...DEFAULT_TRACKER, ...opts };
    const m = target.meta;
    const s = (this.metresPerPx = m.physicalWidth / m.width);
    const cu = (m.width - 1) / 2, cv = (m.height - 1) / 2;
    if (target.video) {
      const V = (this.video = new VideoModel(target));
      this.numTrackPoints = V.numPoints;
      this.obj = new Float32Array(2 * this.numTrackPoints);
      this.usable = new Uint8Array(this.numTrackPoints);
      this.probeMark = new Uint8Array(this.numTrackPoints);
      this.images = [];
      this.imgPixelM = V.patchScales.map((k) => s * k);
    } else {
      this.video = null;
      this.numTrackPoints = target.trackPoints.length / 2;
      this.obj = new Float32Array(2 * this.numTrackPoints);
      for (let i = 0; i < this.numTrackPoints; i++) {
        this.obj[2 * i] = (target.trackPoints[2 * i] - cu) * s;
        this.obj[2 * i + 1] = (target.trackPoints[2 * i + 1] - cv) * s;
      }
      this.usable = new Uint8Array(this.numTrackPoints).fill(1);
      this.images = m.images.map((im, k) => createGray(im.width, im.height, target.images[k]));
      this.imgPixelM = m.images.map((im) => m.physicalWidth / im.width);
    }
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
    this.timeSolver = new PoseSolver(cap, { ...DEFAULT_POSE_OPTIONS, maxIter: 3 });
    this.cObj = new Float32Array(2 * Math.max(cap, COARSE_ATTEMPTS));
    this.cImg = new Float32Array(2 * Math.max(cap, COARSE_ATTEMPTS));
    this.cNorm = new Float64Array(2 * Math.max(cap, COARSE_ATTEMPTS));
  }

  reset(): void {
    this.n = 0;
    this.doubt = 0;
    this.starved = false;
    this.active.fill(0);
    this.ok = false;
  }

  /**
   * Video targets: set the video time of the next frame. Points must be on
   * screen across the window; unless the time is locked they must also hold
   * still across it. Active points that stop being usable are dropped.
   */
  setTime(w: VideoWindow): void {
    const V = this.video;
    if (!V) return;
    this.frameX = w.x;
    const m = this.target.meta, s = this.metresPerPx;
    const cu = (m.width - 1) / 2, cv = (m.height - 1) / 2;
    // Locked: on screen at the estimate (near a cut, the wrong side just fails to track for a frame).
    const a = w.locked ? w.x : w.lo, b = w.locked ? w.x : w.hi;
    const p = this.pb, usable = this.usable, obj = this.obj;
    for (let j = 0; j < this.numTrackPoints; j++) {
      let ok = V.covers(j, a, b);
      if (ok && !w.locked) ok = V.motion(j, a, b, w.x) <= 0.5;
      usable[j] = ok ? 1 : 0;
      if (!ok) continue;
      V.positionAt(j, w.x, p);
      obj[2 * j] = (p[0] - cu) * s;
      obj[2 * j + 1] = (p[1] - cv) * s;
    }
    let k = 0;
    for (let i = 0; i < this.n; i++) {
      const j = this.idx[i];
      if (!usable[j]) { this.active[j] = 0; continue; }
      this.idx[k] = j;
      this.pos[2 * k] = this.pos[2 * i];
      this.pos[2 * k + 1] = this.pos[2 * i + 1];
      k++;
    }
    this.n = k;
  }

  /** Points usable at the current video time (all, for image targets). */
  usableCount(): number {
    let c = 0;
    for (let j = 0; j < this.numTrackPoints; j++) c += this.usable[j];
    return c;
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
    // Video: the pose may be well off (an acquisition matched against a
    // keyframe whose content has moved on since); line it up coarsely first.
    if (this.video) this.coarseAlign(frame, this.R, this.t);
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
    this.replTried = this.replFound = 0;
    if (this.n < o.maxPoints) this.replenish(cur.levels[0], o.replenishAttempts, o.maxPoints);
    if (this.video) this.checkAlignment();
    tm.replenish = now() - t0;
    if (this.inliers < o.minInliers) return this.fail();
    this.ok = true;
    return true;
  }

  /**
   * Video targets: a pose can hold itself up on a few points when much of
   * the picture moves as one (a layer sliding across looks like the screen
   * moving, so a wrong time and a matching wrong pose fit together): the
   * points fit, but hardly any other point is where this pose says. When
   * replenishing keeps failing like that, `starved` is raised for the
   * caller to search the time and pose again (see searchTime).
   */
  private checkAlignment(): void {
    const o = this.o;
    const starved = this.n < 0.5 * o.maxPoints && this.replTried >= 10 && this.replFound < 0.25 * this.replTried;
    this.doubt = starved ? this.doubt + 1 : 0;
    this.starved = this.doubt >= DOUBT_FRAMES;
    if (this.starved) this.doubt = 0;
  }

  /**
   * Video targets: for each candidate frame coordinate, the pose found by
   * coarseAlign from (R, t) and how many points agree with it (into
   * `inliers`; `flips` as for coarseAlign). Returns the index of the candidate most points agree with
   * (its pose in Rbest, tbest), or -1. Changes nothing else.
   */
  searchTime(frame: Pyramid, R: ArrayLike<number>, t: ArrayLike<number>, frames: ArrayLike<number>, count: number,
    inliers: Float64Array, Rbest: Float64Array, tbest: Float64Array, flips: Flips = Flips.None, attempts = SEARCH_ATTEMPTS): number {
    let best = -1;
    const Rc = this.Rq, tc = this.tq, R0 = this.Rs0, t0 = this.ts0;
    for (let i = 0; i < 9; i++) R0[i] = R[i];
    for (let i = 0; i < 3; i++) t0[i] = t[i];
    // Again from the best pose found, while that helps: a start far off
    // (tens of pixels) finds few points at any time, the right one included.
    for (let round = 0; round < SEARCH_ROUNDS; round++) {
      const prev = best < 0 ? 0 : inliers[best];
      let bi = -1;
      for (let ci = 0; ci < count; ci++) {
        Rc.set(R0);
        tc.set(t0);
        const inl = (inliers[ci] = this.coarseAlign(frame, Rc, tc, attempts, frames[ci] + 0.5, flips));
        if (inl > 0 && (bi < 0 || inl > inliers[bi])) {
          bi = ci;
          Rbest.set(Rc);
          tbest.set(tc);
        }
      }
      if (bi < 0) { best = -1; break; }
      best = bi;
      if (inliers[bi] < 1.2 * prev + 2 || inliers[bi] >= 0.8 * attempts) break;
      R0.set(Rbest);
      t0.set(tbest);
    }
    return best;
  }

  /**
   * Video targets: refine (R, t) in place from points matched at a coarse
   * pyramid level over a wide window (tens of pixels at full resolution),
   * with a robust solve. Points are placed for the current video time, or
   * for frame coordinate x if given. With `flips`, also tries both planar
   * solutions of the matches, for a start that may have tilted the wrong
   * way (a target seen nearly head-on can tilt either way): Free picks
   * whichever fits best at full resolution (after an acquisition), Careful
   * keeps the start unless another fits clearly better (while tracking).
   * Returns the number of agreeing points, or 0 (pose unchanged) if too few
   * agree.
   */
  coarseAlign(frame: Pyramid, R: Float64Array, t: Float64Array, attempts = COARSE_ATTEMPTS, x = NaN, flips: Flips = Flips.None): number {
    const V = this.video;
    if (!V || this.numTrackPoints === 0) return 0;
    const L = Math.min(COARSE_LEVEL, frame.levels.length - 1);
    if (L < 1) return 0;
    const img = frame.levels[L], sc = 1 << L, K = this.K, o = this.o;
    // Patch level closest to the pixels of this pyramid level.
    const pxPerM = K.fx / Math.max(1e-3, t[2]) / sc;
    let k = 0, bestErr = Infinity;
    for (let i = 0; i < this.imgPixelM.length; i++) {
      const e = Math.abs(Math.log(this.imgPixelM[i] * pxPerM));
      if (e < bestErr) { bestErr = e; k = i; }
    }
    if (!this.setupWarp(k, R, t)) return 0;
    // Frame pixels of level L -> base pixels.
    const Hi = this.Hinv;
    Hi[0] *= sc; Hi[1] *= sc; Hi[3] *= sc; Hi[4] *= sc; Hi[6] *= sc; Hi[7] *= sc;
    const ncc = this.coarseNcc, obj = this.cObj, im = this.cImg, p = this.p2;
    const edge = ncc.half + ncc.search + 1;
    const total = this.numTrackPoints;
    let n = 0, tried = 0;
    const at = x === x, m = this.target.meta, s = this.metresPerPx, pb = this.pb;
    const cu = (m.width - 1) / 2, cv = (m.height - 1) / 2;
    for (let c = 0; c < total && tried < attempts; c++) {
      const j = (this.coarseCursor + c) % total;
      let X: number, Y: number;
      if (at) {
        if (!V.validAt(j, x)) continue;
        V.positionAt(j, x, pb);
        X = (pb[0] - cu) * s; Y = (pb[1] - cv) * s;
      } else {
        if (!this.usable[j]) continue;
        X = this.obj[2 * j]; Y = this.obj[2 * j + 1];
      }
      if (!project(K, R, t, X, Y, p, 0)) continue;
      const u = p[0] / sc, v = p[1] / sc;
      if (u < edge || v < edge || u >= img.width - edge || v >= img.height - edge) continue;
      tried++;
      if (!this.buildTemplate(ncc, j, k, u, v, at ? x : this.frameX) || !ncc.match(img, COARSE_NCC)) continue;
      obj[2 * n] = X;
      obj[2 * n + 1] = Y;
      im[2 * n] = ncc.x * sc;
      im[2 * n + 1] = ncc.y * sc;
      n++;
    }
    const need = o.minInliers + 4;
    if (n < need) return 0;
    let best = 0;
    const Rs = this.Rp, ts = this.tp, nim = this.cNorm;
    for (let i = 0; i < n; i++) {
      nim[2 * i] = (im[2 * i] - K.cx) / K.fx;
      nim[2 * i + 1] = (im[2 * i + 1] - K.cy) / K.fy;
    }
    // Coarse matches cannot tell the two tilts apart; full-resolution ones can.
    const sols = flips !== Flips.None ? ippe(obj, nim, n) : [];
    let bestFine = -1;
    for (let c = -1; c < sols.length; c++) {
      Rs.set(c < 0 ? R : sols[c].R);
      ts.set(c < 0 ? t : sols[c].t);
      // Twice: the start may be far off.
      if (!this.timeSolver.solve(K, obj, im, n, Rs, ts) || !(ts[2] > 0)) continue;
      if (!this.timeSolver.solve(K, obj, im, n, Rs, ts) || !(ts[2] > 0)) continue;
      const inl = this.timeSolver.inliers;
      if (inl < need || inl < 0.4 * n) continue;
      const fine = sols.length ? this.fineScore(frame.levels[0], Rs, ts, x) : 0;
      // The check samples few points: a careful switch needs a clear margin.
      const margin = flips === Flips.Careful && best > 0 ? Math.max(3, 0.25 * bestFine) : 0;
      if (best === 0 || fine > bestFine + margin || (margin === 0 && fine === bestFine && inl > best)) {
        bestFine = fine;
        best = inl;
        this.Rbest.set(Rs);
        this.tbest.set(ts);
      }
    }
    this.coarseCursor = (this.coarseCursor + 7919) % Math.max(1, total);
    if (best === 0) return 0;
    R.set(this.Rbest);
    t.set(this.tbest);
    return best;
  }

  /**
   * How many of a fixed sample of points match at full resolution (NCC
   * within a few pixels) at pose (R, t), placed for frame coordinate x (or
   * the current time). Same sample every call, so poses compare fairly.
   */
  private fineScore(frame: GrayImage, R: Float64Array, t: Float64Array, x: number): number {
    const V = this.video!, K = this.K, o = this.o, m = this.target.meta, s = this.metresPerPx;
    const cu = (m.width - 1) / 2, cv = (m.height - 1) / 2;
    const pxPerM = K.fx / Math.max(1e-3, t[2]);
    let k = 0, bestErr = Infinity;
    for (let i = 0; i < this.imgPixelM.length; i++) {
      const e = Math.abs(Math.log(this.imgPixelM[i] * pxPerM));
      if (e < bestErr) { bestErr = e; k = i; }
    }
    if (!this.setupWarp(k, R, t)) return 0;
    const at = x === x, total = this.numTrackPoints, p = this.p2, pb = this.pb;
    let tried = 0, found = 0;
    for (let c = 0; c < total && tried < FINE_SAMPLE; c++) {
      const j = (this.coarseCursor + c * 7) % total;
      let X: number, Y: number;
      if (at) {
        if (!V.validAt(j, x)) continue;
        V.positionAt(j, x, pb);
        X = (pb[0] - cu) * s; Y = (pb[1] - cv) * s;
      } else {
        if (!this.usable[j]) continue;
        X = this.obj[2 * j]; Y = this.obj[2 * j + 1];
      }
      if (!project(K, R, t, X, Y, p, 0)) continue;
      if (p[0] < o.margin || p[1] < o.margin || p[0] >= K.width - o.margin || p[1] >= K.height - o.margin) continue;
      tried++;
      if (this.buildTemplate(this.ncc, j, k, p[0], p[1], at ? x : this.frameX) && this.ncc.match(frame, o.minNcc)) found++;
    }
    return found;
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
    for (let k = 0; k < this.imgPixelM.length; k++) {
      const e = Math.abs(Math.log(this.imgPixelM[k] * pxPerM));
      if (e < bestErr) { bestErr = e; best = k; }
    }
    return best;
  }

  /**
   * Set up H (target image k px -> frame px) and its inverse for the current
   * pose. Video targets: base px -> frame px (patches are placed per point).
   */
  private setupWarp(k: number, R: ArrayLike<number> = this.R, t: ArrayLike<number> = this.t): boolean {
    if (this.video) {
      const m = this.target.meta;
      poseToHomography(this.K, R, t, this.metresPerPx, (m.width - 1) / 2, (m.height - 1) / 2, this.H);
    } else {
      const im = this.images[k];
      poseToHomography(this.K, R, t, this.imgPixelM[k], (im.width - 1) / 2, (im.height - 1) / 2, this.H);
    }
    return mat3Invert(this.H, this.Hinv);
  }

  /** NCC template for point j predicted at frame (x, y), from image k or (video) the point's patch at level k. */
  private buildTemplate(ncc: NccMatcher, j: number, k: number, x: number, y: number, frameX = this.frameX): boolean {
    const V = this.video;
    if (!V) return ncc.buildTemplate(this.images[k], this.Hinv, x, y);
    // Patch pixel = (base px - point position) / scale + centre.
    const sc = V.patchScales[k], a = 1 / sc, c = (V.patchSize - 1) / 2;
    V.positionAt(j, frameX, this.pb);
    const bx = c - this.pb[0] * a, by = c - this.pb[1] * a;
    const Hi = this.Hinv, Hc = this.Hc;
    Hc[0] = a * Hi[0] + bx * Hi[6]; Hc[1] = a * Hi[1] + bx * Hi[7]; Hc[2] = a * Hi[2] + bx * Hi[8];
    Hc[3] = a * Hi[3] + by * Hi[6]; Hc[4] = a * Hi[4] + by * Hi[7]; Hc[5] = a * Hi[5] + by * Hi[8];
    Hc[6] = Hi[6]; Hc[7] = Hi[7]; Hc[8] = Hi[8];
    return ncc.buildTemplate(V.patch(V.snapshotAt(j, frameX), k), Hc, x, y);
  }

  /**
   * Video targets: evidence about the video time from the current frame at
   * the current pose, as a log-likelihood per video frame added into `llr`
   * (length frameCount, zeroed here). Each probe point says where it should
   * be, and how it should look, at each frame:
   *   - a point that holds still with one appearance through its time on
   *     screen is checked once: its content is visible or not, which favours
   *     or disfavours the frames it is on screen for;
   *   - a moving or changing point is checked at the candidate frames
   *     (unwrapped indices) where it is on screen, relative to its mean.
   * Frames a probe says nothing about stay at 0: the same content often
   * shows at other times too (a point lost by the compiler and picked up
   * again is another point), so absence is not evidence.
   * Mostly points that move, change or appear within the candidates are
   * used; a few still points with a limited time on screen keep the
   * hypothesis on the right shot. Returns the number of probes used.
   */
  probeTime(frame: GrayImage, frames: ArrayLike<number>, count: number, llr: Float64Array, maxProbes = 24): number {
    const V = this.video;
    if (!V) return 0;
    const N = V.frameCount;
    llr.fill(0, 0, N);
    this.probeFrame = frame;
    if (count < 1) return 0;
    const k = this.pickImage();
    if (!this.setupWarp(k)) return 0;
    const a = frames[0], b = frames[count - 1] + 1, mid = 0.5 * (a + b);
    const total = this.numTrackPoints;
    const ncc = this.probeNcc;
    // First pass: points that keep moving through the candidates (they say
    // which frame is on screen). Second: still points (a few per span, a
    // third of the budget at most: they all say the same thing about their
    // span), and points that move or change only somewhere in the window.
    const maxStill = Math.max(2, Math.floor(maxProbes / 3));
    const spans = this.probeSpans;
    let nSpans = 0;
    let used = 0, still = 0, c = 0;
    const mark = this.probeMark;
    mark.fill(0, 0, total);
    if (this.probeSel.length < maxProbes) this.probeSel = new Int32Array(maxProbes);
    const sel = this.probeSel;
    let ns = 0;
    for (let pass = 0; pass < 2 && used + ns < maxProbes; pass++) {
      for (c = 0; c < total && used + ns < maxProbes; c++) {
        const j = (this.probeCursor + c) % total;
        if (mark[j]) continue;
        const stationary = (V.flags[j] & VPT_STATIONARY) !== 0 && V.snapshotCount(j) === 1;
        if (stationary && V.always(j)) continue;
        if (pass === 0) {
          if (stationary || V.movingFraction(j, a, b) < 0.5) continue;
        } else if (stationary) {
          if (still >= maxStill) continue;
          let same = 0;
          for (let q = 0; q < nSpans; q++) if (spans[2 * q] === V.start[j] && spans[2 * q + 1] === V.end[j]) same++;
          if (same >= 2) continue;
          const sc = this.probeAt(ncc, j, k, V.start[j] + 0.5);
          if (Number.isNaN(sc)) continue;
          this.addSpan(llr, j, probeLlr(sc), 0);
          if (nSpans < spans.length / 2) { spans[2 * nSpans] = V.start[j]; spans[2 * nSpans + 1] = V.end[j]; nSpans++; }
          used++;
          still++;
          continue;
        } else if (V.covers(j, a, b - 1e-6) && V.motion(j, a, b, mid) <= 1 && !V.changesWithin(j, a, b)) {
          continue;
        }
        let valid = 0;
        for (let ci = 0; ci < count && valid < 2; ci++) if (V.validAt(j, frames[ci] + 0.5)) valid++;
        if (valid < 2) continue;
        mark[j] = 1;
        sel[ns++] = j;
      }
    }
    this.probeCursor = (this.probeCursor + Math.max(1, c)) % Math.max(1, total);
    if (ns === 0) return used;
    // Each candidate frame is checked at the pose the tracked points give for
    // it. Content moving as one (a pan, a zoom, a layer sliding across) looks
    // just like the screen moving: at the pose fitted for one time, the
    // content of a nearby time can line up too, and the time would hold
    // itself up. Re-fitted per candidate, only the right time lines up
    // everything.
    if (this.probeVals.length < ns * count) this.probeVals = new Float64Array(ns * count);
    const vals = this.probeVals;
    vals.fill(NaN, 0, ns * count);
    this.prepareTimePoses();
    if (this.geoCost.length < count) this.geoCost = new Float64Array(count);
    const geo = this.geoCost;
    for (let ci = 0; ci < count; ci++) {
      const x = frames[ci] + 0.5;
      const Rp = this.poseForTime(x);
      geo[ci] = this.fitCost;
      if (!this.setupWarp(k, Rp, this.tp)) continue;
      for (let q = 0; q < ns; q++) {
        const j = sel[q];
        if (!V.validAt(j, x)) continue;
        const sc = this.probeAt(ncc, j, k, x, Rp, this.tp);
        if (!Number.isNaN(sc)) vals[q * count + ci] = probeLlr(sc);
      }
    }
    for (let q = 0; q < ns; q++) {
      // Centred on the probe's own mean: content that matches about as
      // well everywhere (repeating glyphs, flat areas) says nothing; only
      // the frames it matches better than its other candidates gain.
      let ne = 0, mean = 0;
      for (let ci = 0; ci < count; ci++) {
        const v = vals[q * count + ci];
        if (v === v) { mean += v; ne++; }
      }
      if (ne < 2) continue;
      mean /= ne;
      for (let ci = 0; ci < count; ci++) {
        const v = vals[q * count + ci];
        if (v === v) llr[((frames[ci] % N) + N) % N] += v - mean;
      }
      used++;
    }
    // The tracked points themselves: how well the best pose for each time
    // fits them. Their sub-pixel positions tell apart times whose content
    // differs by less than the probes resolve, and a time error that a
    // slightly wrong pose would otherwise absorb.
    let ng = 0, gm = 0;
    for (let ci = 0; ci < count; ci++) if (geo[ci] === geo[ci]) { gm += geo[ci]; ng++; }
    if (ng >= 2) {
      gm /= ng;
      for (let ci = 0; ci < count; ci++) {
        if (geo[ci] !== geo[ci]) continue;
        const v = GEO_GAIN * (gm - geo[ci]);
        llr[((frames[ci] % N) + N) % N] += v > GEO_CAP ? GEO_CAP : v < -GEO_CAP ? -GEO_CAP : v;
      }
      used++;
    }
    return used;
  }

  /** Rolling-shutter-corrected positions of the active points, for poseForTime. */
  private prepareTimePoses(): void {
    if (this.tImg.length < 2 * this.n) this.tImg = new Float32Array(2 * this.n);
    this.deskew(this.pos, this.tImg, this.n, this.rsX, this.rsY);
  }

  /**
   * The pose (returned R, and this.tp) that fits the active points' tracked
   * positions to where the video puts them at frame coordinate x; the
   * current pose if too few of them are on screen then.
   */
  private poseForTime(x: number): Float64Array {
    const V = this.video!, m = this.target.meta, s = this.metresPerPx;
    const cu = (m.width - 1) / 2, cv = (m.height - 1) / 2;
    const Rp = this.Rp, tp = this.tp, obj = this.sObj, img = this.sImg, p = this.pb;
    Rp.set(this.R);
    tp.set(this.t);
    let n = 0;
    for (let i = 0; i < this.n; i++) {
      const j = this.idx[i];
      if (!V.validAt(j, x)) continue;
      V.positionAt(j, x, p);
      obj[2 * n] = (p[0] - cu) * s;
      obj[2 * n + 1] = (p[1] - cv) * s;
      img[2 * n] = this.tImg[2 * i];
      img[2 * n + 1] = this.tImg[2 * i + 1];
      n++;
    }
    this.fitCost = NaN;
    if (n < this.o.minInliers + 4) return Rp;
    if (!this.timeSolver.solve(this.K, obj, img, n, Rp, tp) || !(tp[2] > 0) || this.timeSolver.inliers < this.o.minInliers) {
      Rp.set(this.R);
      tp.set(this.t);
      return Rp;
    }
    // Capped squared residuals, scaled to the active points (some may not be
    // on screen at x).
    const res = this.timeSolver.residuals;
    let c = 0;
    for (let i = 0; i < n; i++) c += Math.min(res[i] * res[i], GEO_RES_CAP * GEO_RES_CAP);
    this.fitCost = (c / n) * this.n;
    return Rp;
  }

  /** Probe NCC of point j as shown at frame coordinate x, or NaN if it cannot be checked here. */
  private probeAt(ncc: NccMatcher, j: number, k: number, x: number, R: ArrayLike<number> = this.R, t: ArrayLike<number> = this.t): number {
    const V = this.video!, m = this.target.meta, s = this.metresPerPx, K = this.K, mg = this.o.margin;
    const p = this.pb, q = this.p2;
    V.positionAt(j, x, p);
    if (!project(K, R, t, (p[0] - (m.width - 1) / 2) * s, (p[1] - (m.height - 1) / 2) * s, q, 0)) return NaN;
    if (q[0] < mg || q[1] < mg || q[0] >= K.width - mg || q[1] >= K.height - mg) return NaN;
    if (!this.buildTemplate(ncc, j, k, q[0], q[1], x)) return NaN;
    ncc.match(this.probeFrame!, -2);
    return ncc.score;
  }

  /** Add `inside` over the frames point j is on screen and `outside` over the rest. */
  private addSpan(llr: Float64Array, j: number, inside: number, outside: number): void {
    const V = this.video!, N = V.frameCount;
    if (outside !== 0) for (let f = 0; f < N; f++) llr[f] += outside;
    const d = inside - outside;
    if (d === 0) return;
    for (let f = V.start[j]; f <= V.end[j] && f < V.start[j] + N; f++) llr[f % N] += d;
  }

  private driftCorrect(frame: GrayImage): number {
    const n = this.n;
    if (n === 0) return 0;
    const k = this.pickImage();
    if (!this.setupWarp(k)) return 0;
    const cnt = Math.min(this.o.nccPerFrame, n);
    let corrected = 0;
    const drop = this.status; // reuse as a drop mask
    drop.fill(0, 0, n);
    for (let c = 0; c < cnt; c++) {
      const i = (this.nccCursor + c) % n;
      const j = this.idx[i];
      if (!project(this.K, this.R, this.t, this.obj[2 * j], this.obj[2 * j + 1], this.p2, 0)) continue;
      const ncc = this.ncc;
      if (!this.buildTemplate(ncc, j, k, this.p2[0], this.p2[1])) continue;
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
    const o = this.o, K = this.K;
    let tried = 0, c = 0;
    for (; c < total && tried < attempts && this.n < maxPoints; c++) {
      const j = (this.replCursor + c) % total;
      if (this.active[j] || !this.usable[j]) continue;
      if (!project(K, this.R, this.t, this.obj[2 * j], this.obj[2 * j + 1], this.p2, 0)) continue;
      const x = this.p2[0], y = this.p2[1];
      if (x < o.margin || y < o.margin || x >= K.width - o.margin || y >= K.height - o.margin) continue;
      tried++;
      this.replTried++;
      if (!this.buildTemplate(this.ncc, j, k, x, y) || !this.ncc.match(frame, o.minNcc)) continue;
      this.replFound++;
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
