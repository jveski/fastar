import { Pyramid } from '../image/pyramid';
import { blur5, createGray, type GrayImage } from '../image/gray';
import { FastDetector } from '../vision/fast';
import { FeatureSet, describeLevel, hamming, ORB_BORDER } from '../vision/orb';
import { KmTreeSearcher } from '../vision/kmtree';
import { prosacHomography, homographyDLT, applyH } from '../vision/homography';
import { ippe, type IppeSolution } from '../vision/ippe';
import { PoseSolver } from '../vision/pose';
import type { Intrinsics } from '../vision/camera';
import type { TargetData } from '../target/format';
import { keyframeSpan } from '../target/video';
import { Rng } from '../math/rng';

export interface AcquireResult {
  target: number;
  /** Target base pixels -> frame pixels. */
  H: Float64Array;
  R: Float64Array;
  t: Float64Array;
  inliers: number;
  /** Inlier correspondences: frame [x, y] and target base [u, v]. */
  frame: Float32Array;
  tgt: Float32Array;
  /** Pose covariance from the solver (6x6, [dtheta, dt]). */
  cov: Float64Array;
  timings: { features: number; match: number; total: number };
  /** Target feature index of each inlier. */
  tid?: Int32Array;
  /** Video targets: when the matched content is on screen. */
  time?: AcquireTime;
}

export interface AcquireTime {
  /** Keyframe the match was made against. */
  keyframe: number;
  /** Video-time intervals [start, end) in seconds, flattened; end may pass the clip duration (loop). */
  ranges: number[];
}

export interface AcquireOptions {
  exclude?: ReadonlyArray<number>;
  /** Unit "up" direction (opposite gravity) in camera coordinates, if known. */
  up?: ArrayLike<number> | null;
}

const LEVEL_POINTS = [400, 300, 200, 100];
const RATIO = 0.7;
const MAX_DIST = 80;
const MIN_INLIERS = 12;

export class Acquirer {
  readonly features = new FeatureSet(2048);
  private readonly pyr: Pyramid;
  private readonly smooth: GrayImage[];
  private readonly blurTmp: Uint16Array;
  private readonly det: FastDetector;
  private readonly searcher = new KmTreeSearcher(40);
  /** Video targets hold features from many keyframes: search a little deeper. */
  private readonly videoSearcher = new KmTreeSearcher(80);
  private readonly covisibleFns = new WeakMap<TargetData, (a: number, b: number) => boolean>();
  private readonly rng = new Rng(77);
  private readonly solver = new PoseSolver(4096);
  // Match buffers.
  private readonly mq = new Int32Array(4096);
  private readonly mt = new Int32Array(4096);
  private readonly mquality = new Float32Array(4096);
  private readonly allQ = new Int32Array(4096);
  private readonly allT = new Int32Array(4096);
  private readonly allQuality = new Float32Array(4096);
  /** Per-level FAST thresholds, adapted toward 3-10x the needed corner count. */
  readonly thresholds = new Int32Array([20, 20, 20, 20]);

  constructor(readonly targets: TargetData[], readonly width: number, readonly height: number, public K: Intrinsics) {
    this.pyr = new Pyramid(width, height, 4);
    this.smooth = this.pyr.levels.map((l) => createGray(l.width, l.height));
    this.blurTmp = new Uint16Array(width * height);
    this.det = new FastDetector(width, height, 30000, 600);
  }

  /** Detect FAST + ORB on a 4-level pyramid over the full frame. */
  extract(luma: Uint8Array): FeatureSet {
    this.pyr.build(luma);
    const f = this.features;
    f.clear();
    for (let l = 0; l < this.pyr.numLevels; l++) {
      const lev = this.pyr.levels[l];
      if (lev.width < 2 * ORB_BORDER + 8 || lev.height < 2 * ORB_BORDER + 8) break;
      blur5(lev, this.smooth[l], this.blurTmp);
      const want = LEVEL_POINTS[l] ?? 100;
      const n = this.det.detect(lev, {
        threshold: this.thresholds[l],
        maxPoints: want,
        border: ORB_BORDER,
        cell: Math.max(16, Math.round(Math.max(lev.width, lev.height) / 10)),
      });
      const c = this.det.candidates;
      if (c > 10 * want) this.thresholds[l] = Math.min(40, this.thresholds[l] + 2);
      else if (c < 3 * want) this.thresholds[l] = Math.max(8, this.thresholds[l] - 2);
      describeLevel(lev, this.smooth[l], this.det.x, this.det.y, n, 1 << l, 0, f);
    }
    return f;
  }

  acquire(luma: Uint8Array, o: AcquireOptions = {}): AcquireResult | null {
    const t0 = now();
    this.extract(luma);
    const t1 = now();
    let best: AcquireResult | null = null;
    for (let ti = 0; ti < this.targets.length; ti++) {
      if (o.exclude?.includes(ti)) continue;
      const r = this.matchTarget(ti, o);
      if (r && (!best || r.inliers > best.inliers)) best = r;
    }
    const t2 = now();
    if (best) best.timings = { features: t1 - t0, match: t2 - t1, total: t2 - t0 };
    return best;
  }

  matchTarget(ti: number, o: AcquireOptions = {}): AcquireResult | null {
    const T = this.targets[ti];
    const f = this.features;
    const fdesc = f.desc;
    const { mq, mt, mquality } = this;
    // 1. Tree search + ratio test. A video's look-alike from another time
    // is usually the same content after it moved, not an ambiguity: keyframe
    // voting sorts out which time matches, so only features that can be on
    // screen together compete in the ratio test.
    const searcher = T.video ? this.videoSearcher : this.searcher;
    const covisible = T.video ? this.covisibility(T) : undefined;
    let nm = 0;
    for (let q = 0; q < f.count && nm < mq.length; q++) {
      const b = searcher.search(T.tree, T.desc, T.kp, fdesc, 8 * q, undefined, covisible);
      if (b < 0) continue;
      const d1 = searcher.dist1, d2 = searcher.dist2;
      if (d1 > MAX_DIST || d1 >= RATIO * d2) continue;
      mq[nm] = q; mt[nm] = b; mquality[nm] = d1 / Math.max(1, d2);
      nm++;
    }
    if (nm < MIN_INLIERS) return null;
    if (T.video) return this.matchVideo(ti, T, nm, o);
    return this.fromMatches(ti, T, nm, o, -1);
  }

  /**
   * Video: vote for the keyframe the matches agree on (each feature votes for
   * every keyframe it was seen in), then match against that keyframe's
   * features only. Tries the runner-up keyframe if the best one fails.
   */
  private matchVideo(ti: number, T: TargetData, nm: number, o: AcquireOptions): AcquireResult | null {
    const meta = T.meta.video!, nk = meta.keyframes.length, range = T.video!.featureRange;
    const votes = new Int32Array(nk + 1);
    for (let m = 0; m < nm; m++) addRange(votes, nk, range[2 * this.mt[m]], range[2 * this.mt[m] + 1], 1);
    for (let k = 1; k < nk; k++) votes[k] += votes[k - 1];
    let kb = 0;
    for (let k = 1; k < nk; k++) if (votes[k] > votes[kb]) kb = k;
    let kb2 = -1;
    for (let k = 0; k < nk; k++) {
      const d = Math.min(Math.abs(k - kb), nk - Math.abs(k - kb));
      if (d > 2 && votes[k] >= 0.5 * votes[kb] && (kb2 < 0 || votes[k] > votes[kb2])) kb2 = k;
    }
    this.allQ.set(this.mq.subarray(0, nm));
    this.allT.set(this.mt.subarray(0, nm));
    this.allQuality.set(this.mquality.subarray(0, nm));
    for (const k of kb2 >= 0 ? [kb, kb2] : [kb]) {
      if (votes[k] < MIN_INLIERS) continue;
      let n = 0;
      for (let m = 0; m < nm; m++) {
        const t = this.allT[m];
        if (!inRange(k, range[2 * t], range[2 * t + 1], nk)) continue;
        this.mq[n] = this.allQ[m]; this.mt[n] = t; this.mquality[n] = this.allQuality[m];
        n++;
      }
      const r = this.fromMatches(ti, T, n, o, k);
      if (!r) continue;
      r.time = { keyframe: k, ranges: this.timeRanges(T, r.tid!, k) };
      return r;
    }
    return null;
  }

  /** Whether two features of video target T share a keyframe (cached per target). */
  private covisibility(T: TargetData): (a: number, b: number) => boolean {
    let fn = this.covisibleFns.get(T);
    if (!fn) {
      const range = T.video!.featureRange, nk = T.meta.video!.keyframes.length;
      fn = (a, b) => rangesOverlap(range[2 * a], range[2 * a + 1], range[2 * b], range[2 * b + 1], nk);
      this.covisibleFns.set(T, fn);
    }
    return fn;
  }

  /** Video time where most inlier features are on screen, around keyframe k. */
  private timeRanges(T: TargetData, tid: Int32Array, k: number): number[] {
    const meta = T.meta.video!, nk = meta.keyframes.length, range = T.video!.featureRange;
    const cnt = new Int32Array(nk + 1);
    for (let i = 0; i < tid.length; i++) addRange(cnt, nk, range[2 * tid[i]], range[2 * tid[i] + 1], 1);
    for (let j = 1; j < nk; j++) cnt[j] += cnt[j - 1];
    const thr = 0.6 * cnt[k];
    const out: number[] = [];
    if (Array.from(cnt.subarray(0, nk)).every((c) => c >= thr)) return [0, meta.duration];
    // Maximal circular runs above the threshold.
    let s0 = 0;
    while (cnt[s0] >= thr) s0++; // a keyframe below the threshold exists
    for (let j = 1; j <= nk; j++) {
      const a = (s0 + j) % nk;
      if (cnt[a] < thr) continue;
      const prev = (a + nk - 1) % nk;
      if (cnt[prev] >= thr) continue;
      let e = a;
      while (cnt[(e + 1) % nk] >= thr) e++;
      // Content that slides along can match a neighbouring keyframe about as
      // well (shifted, it looks like the screen moved): allow the
      // neighbouring keyframes too, short of cuts.
      const [f0, f1] = keyframeSpan(meta, isCut(meta, a) ? a : a - 1, isCut(meta, e + 1) ? e : e + 1);
      out.push(f0 / meta.fps, (f1 + 1) / meta.fps);
    }
    return out;
  }

  private fromMatches(ti: number, T: TargetData, nm: number, o: AcquireOptions, keyframe: number): AcquireResult | null {
    const { mq, mt, mquality } = this;
    const fkp = this.features.kp;
    if (nm < MIN_INLIERS) return null;
    // 2. Hough vote on (centre x, centre y, angle, log-scale).
    const sel = this.hough(T, nm);
    if (sel.length < 6) return null;
    sel.sort((a, b) => mquality[a] - mquality[b]);
    const src = new Float32Array(2 * sel.length), dst = new Float32Array(2 * sel.length);
    sel.forEach((m, i) => {
      src[2 * i] = T.kp[4 * mt[m]]; src[2 * i + 1] = T.kp[4 * mt[m] + 1];
      dst[2 * i] = fkp[4 * mq[m]]; dst[2 * i + 1] = fkp[4 * mq[m] + 1];
    });
    // 3. PROSAC homography.
    const H = new Float64Array(9);
    const mask = new Uint8Array(sel.length);
    const inl = prosacHomography(src, dst, sel.length, H, mask, this.rng, { maxIter: 300, threshold: 5, confidence: 0.995 });
    if (inl < 6) return null;
    // 4. Guided matching with the homography.
    const g = this.guided(T, H, keyframe);
    if (!g || g.count < MIN_INLIERS) return null;
    // 5. Pose: IPPE then robust Gauss-Newton.
    const r = this.solvePose(ti, T, g.H, g.frame, g.tgt, g.count, o.up ?? null);
    if (r) r.tid = g.tid;
    return r;
  }

  private hough(T: TargetData, nm: number): number[] {
    const { mq, mt } = this;
    const fkp = this.features.kp, tkp = T.kp;
    const cu = (T.meta.width - 1) / 2, cv = (T.meta.height - 1) / 2;
    const bin = Math.max(this.width, this.height) / 8;
    const keys = new Int32Array(nm * 4);
    const votes = new Map<number, number>();
    for (let m = 0; m < nm; m++) {
      const q = mq[m], t = mt[m];
      const s = fkp[4 * q + 2] / tkp[4 * t + 2];
      const da = fkp[4 * q + 3] - tkp[4 * t + 3];
      const c = Math.cos(da), sn = Math.sin(da);
      const du = cu - tkp[4 * t], dv = cv - tkp[4 * t + 1];
      const x = fkp[4 * q] + s * (c * du - sn * dv);
      const y = fkp[4 * q + 1] + s * (sn * du + c * dv);
      const bx = Math.floor(x / bin) + 16, by = Math.floor(y / bin) + 16;
      let ba = Math.floor(((da / (2 * Math.PI)) * 12) % 12);
      if (ba < 0) ba += 12;
      const bs = Math.floor(Math.log2(s)) + 16;
      if (bx < 0 || bx > 63 || by < 0 || by > 63 || bs < 0 || bs > 31) { keys[4 * m] = -1; continue; }
      keys[4 * m] = bx; keys[4 * m + 1] = by; keys[4 * m + 2] = ba; keys[4 * m + 3] = bs;
      const key = ((bx * 64 + by) * 12 + ba) * 32 + bs;
      votes.set(key, (votes.get(key) ?? 0) + 1);
    }
    let bestKey = -1, bestVotes = 0;
    for (const [k, v] of votes) if (v > bestVotes) { bestVotes = v; bestKey = k; }
    if (bestKey < 0) return [];
    const bs = bestKey % 32, ba = Math.floor(bestKey / 32) % 12;
    const by = Math.floor(bestKey / (32 * 12)) % 64, bx = Math.floor(bestKey / (32 * 12 * 64));
    const out: number[] = [];
    for (let m = 0; m < nm; m++) {
      if (keys[4 * m] < 0) continue;
      const dA = Math.abs(keys[4 * m + 2] - ba);
      if (Math.abs(keys[4 * m] - bx) <= 1 && Math.abs(keys[4 * m + 1] - by) <= 1 &&
          Math.min(dA, 12 - dA) <= 1 && Math.abs(keys[4 * m + 3] - bs) <= 1) out.push(m);
    }
    return out;
  }

  /** Re-match all target features near their projections; returns refined H and inliers. */
  private guided(T: TargetData, H0: Float64Array, keyframe: number) {
    const range = T.video?.featureRange, nk = T.meta.video?.keyframes.length ?? 0;
    const f = this.features, fkp = f.kp, fdesc = f.desc;
    const nT = T.kp.length / 4;
    const cell = 16;
    const gw = Math.ceil(this.width / cell), gh = Math.ceil(this.height / cell);
    const start = new Int32Array(gw * gh + 1), items = new Int32Array(f.count);
    for (let q = 0; q < f.count; q++) start[cellOf(fkp, q, cell, gw, gh) + 1]++;
    for (let c = 0; c < gw * gh; c++) start[c + 1] += start[c];
    const fill = start.slice(0, gw * gh);
    for (let q = 0; q < f.count; q++) items[fill[cellOf(fkp, q, cell, gw, gh)]++] = q;
    const bestD = new Int32Array(f.count).fill(1 << 20), bestT = new Int32Array(f.count).fill(-1);
    const p = new Float64Array(2), p2 = new Float64Array(2);
    const H = H0;
    const radius = 8;
    for (let i = 0; i < nT; i++) {
      if (range && !inRange(keyframe, range[2 * i], range[2 * i + 1], nk)) continue;
      const u = T.kp[4 * i], v = T.kp[4 * i + 1];
      if (!applyH(H, u, v, p)) continue;
      const x = p[0], y = p[1];
      if (x < 0 || y < 0 || x >= this.width || y >= this.height) continue;
      applyH(H, u + 1, v, p2);
      const ls = Math.hypot(p2[0] - x, p2[1] - y); // local frame px per base px
      const expScale = T.kp[4 * i + 2] * ls;
      const cx0 = Math.max(0, Math.floor((x - radius) / cell)), cx1 = Math.min(gw - 1, Math.floor((x + radius) / cell));
      const cy0 = Math.max(0, Math.floor((y - radius) / cell)), cy1 = Math.min(gh - 1, Math.floor((y + radius) / cell));
      let b1 = 1 << 20, bq = -1, b2 = 1 << 20;
      for (let cy = cy0; cy <= cy1; cy++) {
        for (let cx = cx0; cx <= cx1; cx++) {
          const c = cy * gw + cx;
          for (let k = start[c]; k < start[c + 1]; k++) {
            const q = items[k];
            const dx = fkp[4 * q] - x, dy = fkp[4 * q + 1] - y;
            if (dx * dx + dy * dy > radius * radius) continue;
            const r = fkp[4 * q + 2] / expScale;
            if (r > 2.2 || r < 0.45) continue;
            const dd = hamming(T.desc, 8 * i, fdesc, 8 * q);
            if (dd < b1) { b2 = b1; b1 = dd; bq = q; } else if (dd < b2) b2 = dd;
          }
        }
      }
      if (bq < 0 || b1 > 64 || b1 >= 0.85 * b2) continue;
      if (b1 < bestD[bq]) { bestD[bq] = b1; bestT[bq] = i; }
    }
    const pairs: number[] = [];
    for (let q = 0; q < f.count; q++) if (bestT[q] >= 0) pairs.push(q);
    if (pairs.length < MIN_INLIERS) return null;
    pairs.sort((a, b) => bestD[a] - bestD[b]);
    const n = pairs.length;
    const src = new Float32Array(2 * n), dst = new Float32Array(2 * n);
    pairs.forEach((q, k) => {
      const i = bestT[q];
      src[2 * k] = T.kp[4 * i]; src[2 * k + 1] = T.kp[4 * i + 1];
      dst[2 * k] = fkp[4 * q]; dst[2 * k + 1] = fkp[4 * q + 1];
    });
    const H1 = new Float64Array(9);
    const mask = new Uint8Array(n);
    const inl = prosacHomography(src, dst, n, H1, mask, this.rng, { maxIter: 100, threshold: 3, confidence: 0.995 });
    if (inl < MIN_INLIERS) return null;
    homographyDLT(src, dst, n, H1, mask);
    const frame = new Float32Array(2 * inl), tgt = new Float32Array(2 * inl), tid = new Int32Array(inl);
    let k = 0;
    for (let i = 0; i < n && k < inl; i++) {
      if (!mask[i]) continue;
      frame[2 * k] = dst[2 * i]; frame[2 * k + 1] = dst[2 * i + 1];
      tgt[2 * k] = src[2 * i]; tgt[2 * k + 1] = src[2 * i + 1];
      tid[k] = bestT[pairs[i]];
      k++;
    }
    return { H: H1, frame, tgt, tid, count: k };
  }

  private solvePose(ti: number, T: TargetData, H: Float64Array, frame: Float32Array, tgt: Float32Array, n: number, up: ArrayLike<number> | null): AcquireResult | null {
    const K = this.K;
    const s = T.meta.physicalWidth / T.meta.width;
    const cu = (T.meta.width - 1) / 2, cv = (T.meta.height - 1) / 2;
    const obj = new Float32Array(2 * n), nimg = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      obj[2 * i] = (tgt[2 * i] - cu) * s;
      obj[2 * i + 1] = (tgt[2 * i + 1] - cv) * s;
      nimg[2 * i] = (frame[2 * i] - K.cx) / K.fx;
      nimg[2 * i + 1] = (frame[2 * i + 1] - K.cy) / K.fy;
    }
    const sols = ippe(obj, nimg, n);
    if (sols.length === 0) return null;
    const pick = chooseSolution(sols, up);
    const R = pick.R.slice(), t = pick.t.slice();
    if (!this.solver.solve(K, obj, frame, n, R, t)) return null;
    if (this.solver.inliers < MIN_INLIERS || t[2] <= 0) return null;
    return {
      target: ti, H, R, t, inliers: this.solver.inliers, frame, tgt,
      cov: this.solver.cov.slice(),
      timings: { features: 0, match: 0, total: 0 },
    };
  }
}

/**
 * Lower reprojection error wins unless the two are ambiguous, in which case
 * prefer the plane orientation that is closer to vertical or horizontal
 * relative to gravity (posters on walls, pages on tables).
 */
export function chooseSolution(sols: IppeSolution[], up: ArrayLike<number> | null): IppeSolution {
  if (sols.length < 2 || !up) return sols[0];
  const [a, b] = sols;
  if (b.err > 1.5 * a.err) return a;
  const tilt = (S: IppeSolution) => {
    // Target z axis (normal) in camera coordinates is the third column of R.
    const c = Math.abs(S.R[2] * up[0] + S.R[5] * up[1] + S.R[8] * up[2]);
    return Math.min(c, Math.sqrt(Math.max(0, 1 - c * c)));
  };
  return tilt(b) < tilt(a) ? b : a;
}

/** Keyframe k (any integer, wrapped) starts a new shot. */
function isCut(meta: { keyframes: number[]; changes: number[] }, k: number): boolean {
  const nk = meta.keyframes.length, f = meta.keyframes[((k % nk) + nk) % nk];
  return f === 0 || meta.changes.includes(f);
}

/** Add v over keyframes [a, b] (b unwrapped past nk for ranges through the loop seam) to a difference array. */
function addRange(d: Int32Array, nk: number, a: number, b: number, v: number): void {
  if (b - a + 1 >= nk) { d[0] += v; d[nk] -= v; return; }
  if (b < nk) { d[a] += v; d[b + 1] -= v; return; }
  d[a] += v; d[nk] -= v;
  d[0] += v; d[b - nk + 1] -= v;
}

function inRange(k: number, a: number, b: number, nk: number): boolean {
  return (k >= a && k <= b) || (k + nk >= a && k + nk <= b);
}

/** Keyframe ranges [a0, a1] and [b0, b1] (ends unwrapped past nk through the loop seam) share a keyframe. */
function rangesOverlap(a0: number, a1: number, b0: number, b1: number, nk: number): boolean {
  if (a1 - a0 + 1 >= nk || b1 - b0 + 1 >= nk) return true;
  return (a0 <= b1 && b0 <= a1) || (a0 + nk <= b1 && b0 <= a1 + nk) || (a0 <= b1 + nk && b0 + nk <= a1);
}

function cellOf(kp: Float32Array, q: number, cell: number, gw: number, gh: number): number {
  const cx = Math.min(gw - 1, Math.max(0, Math.floor(kp[4 * q] / cell)));
  const cy = Math.min(gh - 1, Math.max(0, Math.floor(kp[4 * q + 1] / cell)));
  return cy * gw + cx;
}

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
