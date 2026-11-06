// Tracker for a target that bends (printed paper): the rigid tracker's
// pipeline (gyro-seeded KLT, NCC drift correction against the target image,
// replenishment) with points placed on a deforming mesh instead of a plane.
// Each frame fits the mesh to the tracked points (SurfaceSolver), then
// expresses it in a target frame that follows the sheet as a whole (the best
// rigid fit of the flat mesh onto it): that pose goes to the filter like any
// target's, and the shape in it changes only as the paper bends.
import type { Pyramid } from '../image/pyramid';
import { createGray, type GrayImage } from '../image/gray';
import type { TargetData } from '../target/format';
import { selectTrackingPoints } from '../target/compiler';
import type { Intrinsics } from '../vision/camera';
import { Klt } from '../vision/klt';
import { NccMatcher } from '../vision/ncc';
import { PoseSolver, poseToHomography } from '../vision/pose';
import { absoluteOrientation, mat3Invert } from '../math/rot';
import { DEFAULT_TRACKER, type TrackerOptions, type TrackTimings } from '../track/tracker';
import { SurfaceMesh } from './mesh';
import { SurfaceSolver, type SurfaceSolverOptions } from './solver';
import { sampleGrid } from './grid';
import { keepMask, setLoad } from '../track/tracker';

export interface SurfaceOptions {
  /** Mesh vertices along the target's long side (cells stay roughly square). */
  grid: number;
  /** Tracking points chosen from the target image (spread evenly over it). */
  trackPoints: number;
  solver?: Partial<SurfaceSolverOptions>;
}

export const DEFAULT_SURFACE: SurfaceOptions = { grid: 9, trackPoints: 500 };

/** Tracker defaults for bending targets: more points, so every cell has some. */
export const SURFACE_TRACKER_DEFAULTS: Partial<TrackerOptions> = { maxPoints: 250, nccPerFrame: 60, replenishAttempts: 60 };

/**
 * Initialization passes: pyramid level matched over a wide window, and the
 * sheet's stiffness for the fit (stiff first, so coarse matches only shape it
 * broadly, as a curl or a lifted side).
 */
const INIT_PASSES: Array<[level: number, stiffness: number]> = [[2, 20], [2, 5], [1, 1]];
/** Wide matching: search radius at the matched level (px) and NCC threshold. */
const COARSE_SEARCH = 10;
const COARSE_NCC = 0.7;
/** Gauss-Newton steps per shape fit while initializing (the start may be far off). */
const INIT_ITERATIONS = 4;
/** Points matched per level while initializing. */
const INIT_POINTS = 160;
/** A fit stretching any edge by more than this is rejected (and the sheet assumed flat again). */
const MAX_STRAIN = 0.2;
/** Wide matches per frame for cells without any tracked point. */
const RESCUE_ATTEMPTS = 8;

export class SurfaceTracker {
  readonly o: TrackerOptions;
  readonly s: SurfaceOptions;
  readonly mesh: SurfaceMesh;
  /** Not a video target (shared shape with TargetTracker). */
  readonly video = null;
  readonly starved = false;
  readonly numTrackPoints: number;
  /** Tracking points in flat target metres [X, Y]. */
  readonly obj: Float32Array;
  /** Per tracking point: face vertices, barycentric weights, face and mesh cell. */
  private readonly bIdx: Int32Array;
  private readonly bW: Float32Array;
  private readonly face: Int32Array;
  private readonly cell: Int32Array;
  // Active set.
  readonly idx: Int32Array;
  readonly pos: Float32Array;
  n = 0;
  private readonly active: Uint8Array;
  // Pose of the sheet as a whole, and its shape in that frame.
  readonly R = new Float64Array(9);
  readonly t = new Float64Array(3);
  readonly cov = new Float64Array(36);
  /** Mesh vertices in target metres (x right, y down, z into the target). */
  readonly shape: Float64Array;
  inliers = 0;
  rms = 0;
  ok = false;
  readonly timings: TrackTimings = { klt: 0, solve: 0, ncc: 0, replenish: 0 };
  // Set by the pipeline; rolling shutter is not corrected on bending targets.
  rsX = 0;
  rsY = 0;
  readonly omega = new Float64Array(3);
  readonly velocity = new Float64Array(3);

  private readonly images: GrayImage[];
  private readonly imgPixelM: number[];
  private readonly klt = new Klt();
  private readonly ncc = new NccMatcher(4, 4);
  private readonly coarseNcc = new NccMatcher(4, COARSE_SEARCH);
  private readonly pose: PoseSolver;
  private readonly solver: SurfaceSolver;
  // Scratch.
  private readonly Vc: Float64Array;
  private readonly Vp: Float64Array;
  private readonly guess: Float32Array;
  private readonly prevPts: Float32Array;
  private readonly status: Uint8Array;
  private readonly sObj: Float32Array;
  private readonly sZ: Float32Array;
  private readonly sIdx: Int32Array;
  private readonly sW: Float32Array;
  private readonly cIds: Int32Array;
  private readonly cImg: Float32Array;
  private readonly cellCount: Int32Array;
  private readonly faceHinv: Float64Array;
  private readonly faceOk: Uint8Array;
  private readonly H = new Float64Array(9);
  private readonly Hl = new Float64Array(9);
  private readonly Rp = new Float64Array(9);
  private readonly tp = new Float64Array(3);
  private readonly p2 = new Float32Array(2);
  private readonly nrm = new Float64Array(3);
  private nccCursor = 0;
  private replCursor = 0;
  private coarseCursor = 0;
  /** Per-frame work limits, set by the tracking budget (setLoad); the options by default. */
  readonly load: Pick<TrackerOptions, 'maxPoints' | 'nccPerFrame' | 'replenishAttempts'>;
  private readonly wsort: Float64Array;

  constructor(readonly target: TargetData, public K: Intrinsics, opts: Partial<TrackerOptions> = {}, surface: Partial<SurfaceOptions> = {}) {
    if (target.video) throw new Error('fastar: video targets cannot be tracked as bending surfaces');
    this.o = { ...DEFAULT_TRACKER, ...SURFACE_TRACKER_DEFAULTS, ...opts };
    this.s = { ...DEFAULT_SURFACE, ...surface };
    const m = target.meta;
    this.mesh = SurfaceMesh.forTarget(m, this.s.grid);
    this.images = m.images.map((im, k) => createGray(im.width, im.height, target.images[k]));
    this.imgPixelM = m.images.map((im) => m.physicalWidth / im.width);
    // Tracking points spread over the whole target, from its ~512 px image.
    const ti = Math.min(1, this.images.length - 1);
    const tp = selectTrackingPoints(this.images[ti], this.s.trackPoints);
    const N = (this.numTrackPoints = tp.length / 2);
    const sc = m.images[ti].scale, s = m.physicalWidth / m.width;
    const cu = (m.width - 1) / 2, cv = (m.height - 1) / 2;
    this.obj = new Float32Array(2 * N);
    this.bIdx = new Int32Array(3 * N);
    this.bW = new Float32Array(3 * N);
    this.face = new Int32Array(N);
    this.cell = new Int32Array(N);
    for (let j = 0; j < N; j++) {
      const X = (this.obj[2 * j] = ((tp[2 * j] + 0.5) * sc - 0.5 - cu) * s);
      const Y = (this.obj[2 * j + 1] = ((tp[2 * j + 1] + 0.5) * sc - 0.5 - cv) * s);
      this.face[j] = this.mesh.bind(X, Y, this.bIdx, this.bW, 3 * j);
      this.cell[j] = this.face[j] >> 1;
    }
    const cap = Math.max(this.o.maxPoints, INIT_POINTS);
    this.idx = new Int32Array(cap);
    this.pos = new Float32Array(2 * cap);
    this.active = new Uint8Array(N);
    this.guess = new Float32Array(2 * cap);
    this.prevPts = new Float32Array(2 * cap);
    this.status = new Uint8Array(cap);
    this.sObj = new Float32Array(2 * cap);
    this.sZ = new Float32Array(cap);
    this.sIdx = new Int32Array(3 * cap);
    this.sW = new Float32Array(3 * cap);
    this.cIds = new Int32Array(cap);
    this.cImg = new Float32Array(2 * cap);
    this.pose = new PoseSolver(cap);
    this.solver = new SurfaceSolver(this.mesh, cap, this.s.solver);
    this.shape = this.mesh.rest.slice();
    this.Vc = new Float64Array(3 * this.mesh.count);
    this.Vp = new Float64Array(3 * this.mesh.count);
    this.cellCount = new Int32Array(this.mesh.faceCount >> 1);
    this.faceHinv = new Float64Array(9 * this.mesh.faceCount);
    this.faceOk = new Uint8Array(this.mesh.faceCount);
    this.wsort = new Float64Array(cap);
    this.load = { maxPoints: this.o.maxPoints, nccPerFrame: this.o.nccPerFrame, replenishAttempts: this.o.replenishAttempts };
  }

  /** Mesh cells (each wants a few points). */
  get cells(): number {
    return this.cellCount.length;
  }

  /** See TargetTracker.setLoad. */
  setLoad(points: number): void {
    setLoad(this.o, this.load, points);
  }

  reset(): void {
    this.n = 0;
    this.active.fill(0);
    this.ok = false;
  }

  /** Forget the shape: the sheet is flat again. */
  flatten(): void {
    this.shape.set(this.mesh.rest);
  }

  /** Point of the (smoothly interpolated) sheet at flat target point (X, Y), in target metres. */
  surfacePoint(X: number, Y: number, out: Float64Array): void {
    const m = this.mesh;
    sampleGrid(this.shape, m.nx, m.ny, m.gridX(X), m.gridY(Y), out);
  }

  /**
   * Seed points in `frame` from a pose (acquisition, or the filter's
   * prediction while coasting), keeping the current shape: match points over
   * wide windows on coarse pyramid levels, fitting pose and shape at each
   * with a shrinking robust scale, then at full resolution. Returns success.
   */
  init(frame: Pyramid, R: ArrayLike<number>, t: ArrayLike<number>, maxAttempts = this.numTrackPoints, minPoints = this.o.minInliers + 4): boolean {
    this.reset();
    for (let i = 0; i < 9; i++) this.R[i] = R[i];
    for (let i = 0; i < 3; i++) this.t[i] = t[i];
    const attempts = Math.min(maxAttempts, INIT_POINTS);
    for (const [L, stiff] of INIT_PASSES) {
      if (L >= frame.levels.length) continue;
      const m = this.matchWide(frame, L, attempts);
      if (m < minPoints) continue;
      this.fit(this.cIds, this.cImg, m, 2 * (1 << L), stiff, INIT_ITERATIONS);
    }
    this.replenish(frame, maxAttempts, this.load.maxPoints, 0);
    if (this.n < minPoints) { this.reset(); return false; }
    this.ok = this.fit(this.idx, this.pos, this.n, 4, 1, INIT_ITERATIONS) && this.fit(this.idx, this.pos, this.n);
    if (this.ok) this.removeOutliers();
    return this.ok && this.inliers >= this.o.minInliers;
  }

  /** Track from `prev` to `cur`. (predR, predT) is the predicted pose at `cur` (e.g. gyro-propagated). */
  track(prev: Pyramid, cur: Pyramid, predR: ArrayLike<number>, predT: ArrayLike<number>): boolean {
    const o = this.o, K = this.K, tm = this.timings;
    let t0 = now();
    // 1. KLT seeded where the predicted pose puts the points on the current shape.
    for (let i = 0; i < 9; i++) this.R[i] = predR[i];
    for (let i = 0; i < 3; i++) this.t[i] = predT[i];
    this.toCamera(this.R, this.t);
    const n = this.n, guess = this.guess, prevPts = this.prevPts, status = this.status;
    for (let i = 0; i < n; i++) {
      prevPts[2 * i] = this.pos[2 * i];
      prevPts[2 * i + 1] = this.pos[2 * i + 1];
      if (!this.project(this.idx[i], guess, 2 * i)) {
        guess[2 * i] = prevPts[2 * i];
        guess[2 * i + 1] = prevPts[2 * i + 1];
      }
    }
    this.klt.trackBatch(prev, cur, prevPts, guess, n, status, o.fbThreshold);
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
    // 2. Pose with the last shape: outliers are left for the shape fit, which
    // can explain points that moved because the paper bent.
    if (m < o.minInliers || !this.solveRigid(this.idx, this.pos, m)) return this.fail();
    t0 = now();
    tm.solve = t0 - t1;
    // 3. NCC drift correction on a rotating subset.
    this.driftCorrect(cur.levels[0]);
    t1 = now();
    tm.ncc = t1 - t0;
    // 4. Pose and shape.
    if (this.n < o.minInliers || !this.fit(this.idx, this.pos, this.n)) return this.fail();
    this.removeOutliers();
    t0 = now();
    tm.solve += t0 - t1;
    // 5. Replenish, filling the emptiest cells first.
    const L = this.load;
    if (this.n < L.maxPoints) this.replenish(cur, L.replenishAttempts, L.maxPoints, RESCUE_ATTEMPTS);
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

  /** Camera-frame mesh vertices for pose (R, t) into Vc. */
  private toCamera(R: ArrayLike<number>, t: ArrayLike<number>): void {
    const D = this.shape, V = this.Vc;
    for (let v = 0; v < this.mesh.count; v++) {
      const x = D[3 * v], y = D[3 * v + 1], z = D[3 * v + 2];
      V[3 * v] = R[0] * x + R[1] * y + R[2] * z + t[0];
      V[3 * v + 1] = R[3] * x + R[4] * y + R[5] * z + t[1];
      V[3 * v + 2] = R[6] * x + R[7] * y + R[8] * z + t[2];
    }
  }

  /** Project tracking point j through the camera-frame vertices Vc. */
  private project(j: number, out: Float32Array, o: number): boolean {
    const V = this.Vc, b = this.bIdx, w = this.bW;
    const a = 3 * b[3 * j], c = 3 * b[3 * j + 1], d = 3 * b[3 * j + 2];
    const wa = w[3 * j], wc = w[3 * j + 1], wd = w[3 * j + 2];
    const z = wa * V[a + 2] + wc * V[c + 2] + wd * V[d + 2];
    if (z <= 1e-6) return false;
    const K = this.K;
    out[o] = (K.fx * (wa * V[a] + wc * V[c] + wd * V[d])) / z + K.cx;
    out[o + 1] = (K.fy * (wa * V[a + 1] + wc * V[c + 1] + wd * V[d + 1])) / z + K.cy;
    return true;
  }

  /** Rigid pose (this.R, this.t) of points `ids` seen at `img`, on the current shape. */
  private solveRigid(ids: Int32Array, img: Float32Array, n: number): boolean {
    const D = this.shape, b = this.bIdx, w = this.bW;
    for (let i = 0; i < n; i++) {
      const j = ids[i];
      const a = 3 * b[3 * j], c = 3 * b[3 * j + 1], d = 3 * b[3 * j + 2];
      const wa = w[3 * j], wc = w[3 * j + 1], wd = w[3 * j + 2];
      this.sObj[2 * i] = wa * D[a] + wc * D[c] + wd * D[d];
      this.sObj[2 * i + 1] = wa * D[a + 1] + wc * D[c + 1] + wd * D[d + 1];
      this.sZ[i] = wa * D[a + 2] + wc * D[c + 2] + wd * D[d + 2];
    }
    if (!this.pose.solve(this.K, this.sObj, img, n, this.R, this.t, this.sZ) || !(this.t[2] > 0)) return false;
    this.inliers = this.pose.inliers;
    this.rms = this.pose.rmsInlier;
    this.cov.set(this.pose.cov);
    return true;
  }

  /**
   * Fit pose and shape to points `ids` seen at `img`: the shape in the
   * camera from the current pose, the pose as the rigid fit of the flat
   * mesh onto it, then the pose again on the new shape (for its
   * covariance). minSigma: floor of the shape fit's robust scale (px);
   * stiffness: multiplies its bend weight.
   */
  private fit(ids: Int32Array, img: Float32Array, n: number, minSigma?: number, stiffness = 1, iterations?: number): boolean {
    if (n < this.o.minInliers) return false;
    if (!this.solveRigid(ids, img, n)) return false;
    const b = this.bIdx, w = this.bW, sIdx = this.sIdx, sW = this.sW;
    for (let i = 0; i < n; i++) {
      const j = ids[i];
      for (let k = 0; k < 3; k++) { sIdx[3 * i + k] = b[3 * j + k]; sW[3 * i + k] = w[3 * j + k]; }
    }
    this.toCamera(this.R, this.t);
    this.Vp.set(this.Vc);
    const R = this.R, t = this.t, V = this.Vc;
    const nrm = this.nrm;
    nrm[0] = R[2]; nrm[1] = R[5]; nrm[2] = R[8];
    if (!this.solver.solve(this.K, V, this.Vp, nrm, sIdx, sW, img, n, minSigma, stiffness, iterations) || this.solver.strain(V) > MAX_STRAIN) {
      this.flatten();
      return false;
    }
    if (!absoluteOrientation(this.mesh.rest, V, this.mesh.count, null, R, t) || !(t[2] > 0)) return false;
    const D = this.shape;
    for (let v = 0; v < this.mesh.count; v++) {
      const x = V[3 * v] - t[0], y = V[3 * v + 1] - t[1], z = V[3 * v + 2] - t[2];
      D[3 * v] = R[0] * x + R[3] * y + R[6] * z;
      D[3 * v + 1] = R[1] * x + R[4] * y + R[7] * z;
      D[3 * v + 2] = R[2] * x + R[5] * y + R[8] * z;
    }
    return this.solveRigid(ids, img, n);
  }

  private removeOutliers(): void {
    const w = this.pose.weights;
    const keep = keepMask(w, this.n, this.load.maxPoints, this.wsort);
    let m = 0;
    for (let i = 0; i < this.n; i++) {
      if (!keep(i)) { this.active[this.idx[i]] = 0; continue; }
      this.idx[m] = this.idx[i];
      this.pos[2 * m] = this.pos[2 * i];
      this.pos[2 * m + 1] = this.pos[2 * i + 1];
      m++;
    }
    this.n = m;
    if (this.inliers > m) this.inliers = m;
  }

  /** Target image whose pixels are closest to one frame pixel at pyramid scale sc. */
  private pickImage(sc = 1): number {
    const pxPerM = this.K.fx / Math.max(1e-3, this.t[2]) / sc;
    let best = 0, bestErr = Infinity;
    for (let k = 0; k < this.imgPixelM.length; k++) {
      const e = Math.abs(Math.log(this.imgPixelM[k] * pxPerM));
      if (e < bestErr) { bestErr = e; best = k; }
    }
    return best;
  }

  /**
   * Per face: the homography from frame pixels to target image k's pixels
   * for the current pose and shape (each face is planar).
   */
  private setupFaceWarps(k: number): void {
    this.toCamera(this.R, this.t);
    const m = this.mesh, V = this.Vc, r = m.rest, im = this.images[k];
    const Rp = this.Rp, tp = this.tp, H = this.H;
    const cu = (im.width - 1) / 2, cv = (im.height - 1) / 2;
    for (let f = 0; f < m.faceCount; f++) {
      const a = 3 * m.faces[3 * f], b = 3 * m.faces[3 * f + 1], c = 3 * m.faces[3 * f + 2];
      const i00 = m.faceInv[4 * f], i01 = m.faceInv[4 * f + 1], i10 = m.faceInv[4 * f + 2], i11 = m.faceInv[4 * f + 3];
      // Face plane: P(X, Y) = M [X, Y] + (V_a - M r_a), with M = [E1 E2] inv.
      for (let q = 0; q < 3; q++) {
        const e1 = V[b + q] - V[a + q], e2 = V[c + q] - V[a + q];
        const m0 = e1 * i00 + e2 * i10, m1 = e1 * i01 + e2 * i11;
        Rp[3 * q] = m0;
        Rp[3 * q + 1] = m1;
        tp[q] = V[a + q] - m0 * r[a] - m1 * r[a + 1];
      }
      poseToHomography(this.K, Rp, tp, this.imgPixelM[k], cu, cv, H);
      const ok = mat3Invert(H, this.Hl);
      this.faceOk[f] = ok ? 1 : 0;
      for (let i = 0; i < 9; i++) this.faceHinv[9 * f + i] = this.Hl[i];
    }
  }

  /** NCC template for point j predicted at level-L frame position (x, y), from image k. */
  private template(ncc: NccMatcher, j: number, k: number, x: number, y: number, L = 0): boolean {
    const f = this.face[j];
    if (!this.faceOk[f]) return false;
    // Frame pixels of level L -> level 0: scale the first two columns.
    const sc = 1 << L, Hl = this.Hl, Hf = this.faceHinv;
    for (let i = 0; i < 9; i++) Hl[i] = Hf[9 * f + i] * (i % 3 === 2 ? 1 : sc);
    return ncc.buildTemplate(this.images[k], Hl, x, y);
  }

  private driftCorrect(frame: GrayImage): number {
    const n = this.n;
    if (n === 0) return 0;
    const k = this.pickImage();
    this.setupFaceWarps(k);
    const cnt = Math.min(this.load.nccPerFrame, n);
    let corrected = 0;
    const drop = this.status;
    drop.fill(0, 0, n);
    const p = this.p2, ncc = this.ncc;
    for (let c = 0; c < cnt; c++) {
      const i = (this.nccCursor + c) % n;
      const j = this.idx[i];
      if (!this.project(j, p, 0) || !this.template(ncc, j, k, p[0], p[1])) continue;
      if (ncc.match(frame, this.o.minNcc)) {
        this.pos[2 * i] = ncc.x;
        this.pos[2 * i + 1] = ncc.y;
        corrected++;
      } else if (ncc.score < this.o.dropNcc) {
        drop[i] = 1;
      }
    }
    this.nccCursor = (this.nccCursor + cnt) % Math.max(1, n);
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

  /**
   * Match points over a wide window at pyramid level L (into cIds/cImg,
   * level-0 pixels), spread over the target. Returns the number matched.
   */
  private matchWide(frame: Pyramid, L: number, attempts: number): number {
    const sc = 1 << L, img = frame.levels[L], ncc = this.coarseNcc;
    const k = this.pickImage(sc);
    this.setupFaceWarps(k);
    const edge = ncc.half + ncc.search + 1, p = this.p2, total = this.numTrackPoints;
    // A stride coprime with the point count visits the target evenly.
    const stride = coprimeStride(total);
    let n = 0, tried = 0;
    for (let c = 0; c < total && tried < attempts && n < this.cIds.length; c++) {
      const j = (this.coarseCursor + c * stride) % total;
      if (!this.project(j, p, 0)) continue;
      const u = p[0] / sc, v = p[1] / sc;
      if (u < edge || v < edge || u >= img.width - edge || v >= img.height - edge) continue;
      tried++;
      if (!this.template(ncc, j, k, u, v, L) || !ncc.match(img, COARSE_NCC)) continue;
      this.cIds[n] = j;
      this.cImg[2 * n] = ncc.x * sc;
      this.cImg[2 * n + 1] = ncc.y * sc;
      n++;
    }
    this.coarseCursor = (this.coarseCursor + 7919) % Math.max(1, total);
    return n;
  }

  /**
   * Add inactive tracking points that project into the frame, located by
   * NCC, cells with the fewest points first. Up to `rescue` points in cells
   * with none are searched over a wide window first: where the paper bent
   * since its points were lost, the shape there is a guess.
   */
  private replenish(frame: Pyramid, attempts: number, maxPoints: number, rescue: number): void {
    const total = this.numTrackPoints;
    if (total === 0) return;
    const o = this.o, K = this.K, base = frame.levels[0];
    const counts = this.cellCount;
    counts.fill(0);
    for (let i = 0; i < this.n; i++) counts[this.cell[this.idx[i]]]++;
    const cap = Math.max(2, Math.ceil((1.5 * maxPoints) / counts.length));
    const k = this.pickImage();
    this.setupFaceWarps(k);
    const p = this.p2, ncc = this.ncc, wide = this.coarseNcc, L = Math.min(1, frame.levels.length - 1);
    const lev = frame.levels[L], sc = 1 << L, edge = wide.half + wide.search + 1;
    let kw = -1;
    let tried = 0, c = 0;
    for (; c < total && tried < attempts && this.n < maxPoints; c++) {
      const j = (this.replCursor + c) % total;
      if (this.active[j]) continue;
      const cell = this.cell[j];
      if (counts[cell] >= cap) continue;
      if (!this.project(j, p, 0)) continue;
      let x = p[0], y = p[1];
      if (x < o.margin || y < o.margin || x >= K.width - o.margin || y >= K.height - o.margin) continue;
      tried++;
      if (counts[cell] === 0 && rescue > 0 && L > 0) {
        rescue--;
        if (kw < 0) kw = this.pickImage(sc);
        const u = x / sc, v = y / sc;
        if (u < edge || v < edge || u >= lev.width - edge || v >= lev.height - edge) continue;
        // The coarse image level is right for this level's pixels; warps are per image.
        if (kw !== k) this.setupFaceWarps(kw);
        const okWide = this.template(wide, j, kw, u, v, L) && wide.match(lev, COARSE_NCC);
        if (kw !== k) this.setupFaceWarps(k);
        if (!okWide) continue;
        x = wide.x * sc;
        y = wide.y * sc;
      }
      if (!this.template(ncc, j, k, x, y) || !ncc.match(base, o.minNcc)) continue;
      this.idx[this.n] = j;
      this.pos[2 * this.n] = ncc.x;
      this.pos[2 * this.n + 1] = ncc.y;
      this.active[j] = 1;
      counts[cell]++;
      this.n++;
    }
    this.replCursor = (this.replCursor + Math.max(1, c)) % total;
  }
}

/** A prime step that visits all n points (n is not a multiple of it). */
function coprimeStride(n: number): number {
  for (const s of [97, 89, 83, 79, 73, 71]) if (n % s !== 0) return s;
  return 1;
}

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
