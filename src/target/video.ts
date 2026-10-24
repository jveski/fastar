// Runtime view of a video target's baked points: validity, paths and patch
// snapshots over video time.
//
// Conventions: video frame i is on screen for video time tv in
// [i / fps, (i + 1) / fps). Times are handled as frame coordinates
// x = tv * fps. A point's frames run from `start` to `end` inclusive, with
// `end` unwrapped past the clip end (end >= frameCount) for points that
// continue through the loop seam; its path vertices use the same unwrapped
// frames. A camera sees frame floor(x)'s positions while that frame is on
// screen, and a blend of two frames when its exposure spans a frame change:
// positions hold through each frame and blend over EXPOSURE_HALF either
// side of the change.
import { createGray, type GrayImage } from '../image/gray';
import { VPT_STATIONARY, VPT_STRIDE, type TargetData, type VideoMeta } from './format';

/**
 * Half the exposure assumed around a frame change (frames): between instant
 * exposures and whole-frame ones, about what a phone camera filming a
 * screen uses (1/60 s at 30 fps).
 */
const EXPOSURE_HALF = 0.25;

export class VideoModel {
  readonly meta: VideoMeta;
  readonly frameCount: number;
  readonly fps: number;
  readonly duration: number;
  readonly numPoints: number;
  readonly start: Int32Array;
  readonly end: Int32Array;
  readonly flags: Uint32Array;
  readonly patchSize: number;
  readonly patchScales: number[];
  private readonly pathOff: Int32Array;
  private readonly pathCnt: Int32Array;
  private readonly snapOff: Int32Array;
  private readonly snapCnt: Int32Array;
  private readonly path: Float32Array;
  private readonly patchIndex: Uint32Array;
  /** Patch images per snapshot and level. */
  private readonly patchImages: GrayImage[];
  private readonly levels: number;

  constructor(readonly target: TargetData) {
    const v = target.video, m = target.meta.video;
    if (!v || !m) throw new Error('fastar: not a video target');
    this.meta = m;
    this.frameCount = m.frameCount;
    this.fps = m.fps;
    this.duration = m.duration;
    const n = (this.numPoints = v.points.length / VPT_STRIDE);
    this.start = new Int32Array(n); this.end = new Int32Array(n); this.flags = new Uint32Array(n);
    this.pathOff = new Int32Array(n); this.pathCnt = new Int32Array(n);
    this.snapOff = new Int32Array(n); this.snapCnt = new Int32Array(n);
    for (let j = 0; j < n; j++) {
      const o = j * VPT_STRIDE, p = v.points;
      this.start[j] = p[o]; this.end[j] = p[o + 1];
      this.pathOff[j] = p[o + 2]; this.pathCnt[j] = p[o + 3];
      this.snapOff[j] = p[o + 4]; this.snapCnt[j] = p[o + 5];
      this.flags[j] = p[o + 6];
    }
    this.path = v.path;
    this.patchIndex = v.patchIndex;
    this.patchSize = m.patchSize;
    this.patchScales = m.patchScales;
    this.levels = m.patchScales.length;
    const P = m.patchSize, L = this.levels;
    const ns = v.patchIndex.length / 2;
    this.patchImages = new Array(ns * L);
    for (let s = 0; s < ns; s++) {
      const off = v.patchIndex[2 * s + 1];
      for (let l = 0; l < L; l++) {
        this.patchImages[s * L + l] = createGray(P, P, v.patches.subarray(off + l * P * P, off + (l + 1) * P * P));
      }
    }
  }

  /** True if the point spans the whole loop. */
  always(j: number): boolean {
    return this.end[j] - this.start[j] + 1 >= this.frameCount;
  }

  /** Frame coordinate x moved into point j's unwrapped frame range (it may still lie past `end`). */
  unwrap(j: number, x: number): number {
    const N = this.frameCount;
    let r = x - N * Math.floor(x / N);
    if (r < this.start[j]) r += N;
    return r;
  }

  /** Point j is on screen at frame coordinate x. */
  validAt(j: number, x: number): boolean {
    if (this.always(j)) return true;
    return Math.floor(this.unwrap(j, x)) <= this.end[j];
  }

  /** Point j is on screen for the whole of [a, b] (frame coordinates, b - a < frameCount). */
  covers(j: number, a: number, b: number): boolean {
    // A stationary whole-loop point has no seam; any other point's span ends somewhere.
    if (this.always(j) && (this.flags[j] & VPT_STATIONARY)) return true;
    const r = this.unwrap(j, a);
    return Math.floor(r + (b - a)) <= this.end[j];
  }

  /**
   * Frame (fractional between frames) whose positions a camera sees at
   * frame coordinate x: frame floor(x) while it is on screen, blending into
   * the next one over an exposure's worth of time around each frame change.
   */
  frameAt(x: number): number {
    const h = EXPOSURE_HALF, b = Math.round(x), d = x - b;
    return d >= h ? b : d <= -h ? b - 1 : b - 0.5 + d / (2 * h);
  }

  /** Base-pixel position of point j at frame coordinate x (unwrapped for the point). */
  positionAt(j: number, x: number, out: Float32Array | Float64Array, o = 0): void {
    const po = this.pathOff[j], pc = this.pathCnt[j], P = this.path;
    const u = this.frameAt(this.unwrap(j, x));
    const f0 = P[3 * po];
    if (pc === 1 || u <= f0) { out[o] = P[3 * po + 1]; out[o + 1] = P[3 * po + 2]; return; }
    const last = po + pc - 1;
    if (u >= P[3 * last]) { out[o] = P[3 * last + 1]; out[o + 1] = P[3 * last + 2]; return; }
    let lo = po, hi = last; // P[3 lo] <= u < P[3 hi]
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (P[3 * mid] <= u) lo = mid; else hi = mid;
    }
    const a = (u - P[3 * lo]) / (P[3 * hi] - P[3 * lo]);
    out[o] = P[3 * lo + 1] + a * (P[3 * hi + 1] - P[3 * lo + 1]);
    out[o + 1] = P[3 * lo + 2] + a * (P[3 * hi + 2] - P[3 * lo + 2]);
  }

  /** Largest distance (base px) point j moves from its position at x over [a, b]. */
  motion(j: number, a: number, b: number, x: number): number {
    const pc = this.pathCnt[j];
    if (pc === 1) return 0;
    const p = this.tmp, q = this.tmp2;
    this.positionAt(j, x, p);
    let m = 0;
    this.positionAt(j, a, q);
    m = Math.max(m, Math.hypot(q[0] - p[0], q[1] - p[1]));
    this.positionAt(j, b, q);
    m = Math.max(m, Math.hypot(q[0] - p[0], q[1] - p[1]));
    // Path vertices inside the interval.
    const ua = this.unwrap(j, a) - 0.5, ub = ua + (b - a);
    const po = this.pathOff[j], P = this.path;
    for (let k = po; k < po + pc; k++) {
      const f = P[3 * k];
      if (f > ua && f < ub) m = Math.max(m, Math.hypot(P[3 * k + 1] - p[0], P[3 * k + 2] - p[1]));
    }
    return m;
  }

  /**
   * Fraction of [a, b] (frame coordinates) during which point j moves at
   * least `minSpeed` base px per frame: 1 for something that keeps moving,
   * small for something that only slides away once.
   */
  movingFraction(j: number, a: number, b: number, minSpeed = 0.5): number {
    const pc = this.pathCnt[j];
    if (pc < 2 || b <= a) return 0;
    const ua = this.unwrap(j, a) - 0.5, ub = ua + (b - a);
    const po = this.pathOff[j], P = this.path;
    let moving = 0;
    for (let k = po; k + 1 < po + pc; k++) {
      const f0 = P[3 * k], f1 = P[3 * k + 3];
      const lo = Math.max(f0, ua), hi = Math.min(f1, ub);
      if (hi <= lo) continue;
      const v = Math.hypot(P[3 * k + 4] - P[3 * k + 1], P[3 * k + 5] - P[3 * k + 2]) / Math.max(1e-6, f1 - f0);
      if (v >= minSpeed) moving += hi - lo;
    }
    return moving / (b - a);
  }

  private readonly tmp = new Float64Array(2);
  private readonly tmp2 = new Float64Array(2);

  /** Index of the patch snapshot of point j shown at frame coordinate x. */
  snapshotAt(j: number, x: number): number {
    const so = this.snapOff[j], sc = this.snapCnt[j], I = this.patchIndex;
    const f = Math.floor(this.unwrap(j, x));
    let k = so;
    while (k + 1 < so + sc && I[2 * (k + 1)] <= f) k++;
    return k;
  }

  snapshotCount(j: number): number {
    return this.snapCnt[j];
  }

  /** The appearance of point j changes within [a, b] (another snapshot starts inside). */
  changesWithin(j: number, a: number, b: number): boolean {
    const so = this.snapOff[j], sc = this.snapCnt[j];
    if (sc <= 1) return false;
    const fa = Math.floor(this.unwrap(j, a)), fb = fa + Math.floor(b - a) + 1;
    const I = this.patchIndex;
    for (let k = so + 1; k < so + sc; k++) if (I[2 * k] > fa && I[2 * k] <= fb) return true;
    return false;
  }

  patch(snapshot: number, level: number): GrayImage {
    return this.patchImages[snapshot * this.levels + level];
  }

  /**
   * Video frames during which keyframe range [k0, k1] (k1 unwrapped) is on
   * screen: from halfway after the previous keyframe (or exactly at a cut)
   * to halfway before the next one. Returned unwrapped: [first, last].
   */
  keyframeSpan(k0: number, k1: number): [number, number] {
    return keyframeSpan(this.meta, k0, k1);
  }
}

export function keyframeSpan(m: VideoMeta, k0: number, k1: number): [number, number] {
  const K = m.keyframes, nk = K.length, N = m.frameCount;
  const frameOf = (k: number) => K[((k % nk) + nk) % nk] + N * Math.floor(k / nk);
  const isChange = (k: number) => {
    const f = K[((k % nk) + nk) % nk];
    return f === 0 || m.changes.includes(f);
  };
  const f0 = frameOf(k0), f1 = frameOf(k1);
  const start = isChange(k0) ? f0 : Math.ceil((frameOf(k0 - 1) + f0) / 2);
  const end = isChange(k1 + 1) ? frameOf(k1 + 1) - 1 : Math.floor((f1 + frameOf(k1 + 1)) / 2);
  return [start, Math.max(start, end)];
}
