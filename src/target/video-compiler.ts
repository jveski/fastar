// Video target compiler: follows points through the clip with KLT, keeps
// patch snapshots wherever their appearance changes, and extracts ORB
// features on keyframes (merging features that stay put across keyframes).
// Streaming: frames are added one at a time, so memory does not grow with
// the frame size times the clip length. No platform APIs, so the CLI (via
// ffmpeg) and the browser (via a <video> element) feed the same builder.
import { blur5, createGray, resizeArea, type GrayImage } from '../image/gray';
import { Pyramid } from '../image/pyramid';
import { Klt } from '../vision/klt';
import { FastDetector } from '../vision/fast';
import { FeatureSet, describeLevel, hamming, ORB_BORDER } from '../vision/orb';
import { buildKmTree } from '../vision/kmtree';
import { cornerCandidates, selfSimilar, selfSimilarity } from './compiler';
import { VPT_STATIONARY, VPT_STRIDE, type QualityReport, type TargetData, type VideoMeta } from './format';

export interface VideoCompileOptions {
  name?: string;
  /** Physical width of the displayed video in metres. */
  physicalWidth: number;
  /** Frame rate of the frames passed in. */
  fps: number;
  /** The clip plays in a loop (default true): points and features continue across the seam. */
  loop?: boolean;
  /** Long side of the analysed frames (larger inputs are downscaled). */
  maxDim?: number;
  /** Seconds between keyframes (more are added at cuts). */
  keyframeInterval?: number;
  /** Tracking points to keep on screen at any moment. */
  pointsPerFrame?: number;
  maxActiveTracks?: number;
  patchSize?: number;
  maxPerLevel?: number;
  minLevelSize?: number;
  fastThreshold?: number;
  /** Cap on stored features (longest-lived kept). */
  maxFeatures?: number;
  /** New features kept per keyframe between cuts. */
  maxNewPerKeyframe?: number;
  /**
   * Features each keyframe holds at least, topping up with more new ones
   * when few carry over from earlier keyframes (clips where everything moves).
   */
  minKeyframeFeatures?: number;
}

const LEVEL_STEP = Math.pow(2, 1 / 3);
const PATCH_LEVELS = 3; // patch scales 1, 2, 4 (pyramid levels)
const BORDER = 10;
const FB_MAX = 0.3; // forward-backward KLT error (px)
const STEP_NCC = 0.85; // appearance between consecutive frames
const SNAP_NCC = 0.9; // appearance against the last snapshot
const STATIONARY_PX = 0.3;
const PATH_TOL = 0.25;
const MERGE_GAP = 4; // keyframes
const SEAM_S = 1; // looping clips: points are followed this far past the seam
const CHANGE_HISTORY = 8; // frames of recent change a cut is measured against
const TURNOVER = 0.5; // share of tracks ended since the last keyframe that calls for a new one

interface Track {
  start: number;
  xs: number[];
  ys: number[];
  snapFrames: number[];
  snaps: Uint8Array[];
  /** 9x9 base patch at the last snapshot, and at the start. */
  ref: Float32Array;
  firstRef: Float32Array;
  score: number;
  vx: number;
  vy: number;
  alive: boolean;
  /** Keyframe this track was alive at (turnover between keyframes). */
  kf: number;
}

export class VideoTargetBuilder {
  readonly o: Required<Omit<VideoCompileOptions, 'name'>> & { name: string };
  private W = 0;
  private H = 0;
  private f = 0;
  private pyr: Pyramid[] = [];
  private base: GrayImage | null = null;
  private thumb: GrayImage | null = null;
  private prevThumb: GrayImage | null = null;
  private readonly klt = new Klt({ halfWin: 4, levels: 4, maxIter: 20, eps: 0.01, minEig: 1e-4 });
  private tracks: Track[] = [];
  private alive: Track[] = [];
  private readonly s9a = new Float32Array(81);
  private readonly s9b = new Float32Array(81);
  // Keyframes and features.
  private readonly keyframes: number[] = [];
  private readonly changes: number[] = [];
  /** Recent frame-to-frame thumbnail differences and track death rates (cut detection). */
  private readonly diffHist: number[] = [];
  private readonly deathHist: number[] = [];
  private readonly all = new FeatureSet(8192);
  private readonly first: number[] = [];
  private readonly last: number[] = [];
  private readonly level: number[] = [];
  private readonly kfFeats = new FeatureSet(4096);
  private levelScales: number[] = [];
  private levelImgs: GrayImage[] = [];
  private levelSmooth: GrayImage[] = [];
  private det: FastDetector | null = null;
  private kfCoverage: number[] = [];
  private kfCounts: number[] = [];
  private kf0: { kp: Float32Array; desc: Uint32Array; n: number } | null = null;
  /** Tracks alive at the last keyframe, and how many of them have ended since. */
  private kfAlive = 0;
  private kfLost = 0;
  /** Looping clips: the first frames (analysed size), to follow points through the seam. */
  private head: Uint8Array[] = [];

  constructor(o: VideoCompileOptions) {
    if (!(o.physicalWidth > 0)) throw new Error('physicalWidth must be > 0 (metres)');
    if (!(o.fps > 0)) throw new Error('fps must be > 0');
    this.o = {
      name: o.name ?? 'video',
      physicalWidth: o.physicalWidth,
      fps: o.fps,
      loop: o.loop ?? true,
      maxDim: o.maxDim ?? 1024,
      keyframeInterval: o.keyframeInterval ?? 0.5,
      pointsPerFrame: o.pointsPerFrame ?? 160,
      maxActiveTracks: o.maxActiveTracks ?? 500,
      patchSize: o.patchSize ?? 20,
      maxPerLevel: o.maxPerLevel ?? 300,
      minLevelSize: o.minLevelSize ?? 100,
      fastThreshold: o.fastThreshold ?? 20,
      maxFeatures: o.maxFeatures ?? 32000,
      maxNewPerKeyframe: o.maxNewPerKeyframe ?? 300,
      minKeyframeFeatures: o.minKeyframeFeatures ?? 600,
    };
  }

  get frameCount(): number {
    return this.f;
  }

  addFrame(input: GrayImage): void {
    if (this.f === 0) this.setup(input);
    const W = this.W, H = this.H;
    if (Math.abs(input.width / input.height - W / H) > 0.01 * (W / H)) {
      throw new Error(`frame ${this.f} is ${input.width}x${input.height}; expected the aspect of ${W}x${H}`);
    }
    const base = input.width === W && input.height === H ? input : resizeArea(input, W, H, this.base!);
    const cur = this.pyr[this.f & 1], prev = this.pyr[(this.f + 1) & 1];
    cur.build(base.data);
    if (this.o.loop && this.f < Math.round(SEAM_S * this.o.fps)) this.head.push(base.data.slice());
    resizeArea(cur.levels[0], this.thumb!.width, this.thumb!.height, this.thumb!);
    const f = this.f;
    let change = false;
    if (f > 0) {
      const before = this.alive.length;
      const died = this.step(prev, cur, f);
      let diff = 0;
      const a = this.thumb!.data, b = this.prevThumb!.data;
      for (let i = 0; i < a.length; i++) diff += Math.abs(a[i] - b[i]);
      diff /= a.length;
      // A cut is a jump well beyond how much the picture has been changing
      // from frame to frame: content that churns all the time is not a cut
      // at every frame. (Until there is a history, assume moderate change.)
      const frac = before > 0 ? died / before : 0;
      const dBase = this.diffHist.length ? median(this.diffHist) : 6;
      const fBase = this.deathHist.length ? median(this.deathHist) : 0.05;
      change = (diff > 15 && diff > 2.5 * dBase + 3) || (before >= 8 && frac >= 0.35 && frac > 2 * fBase + 0.1);
      if (change) this.changes.push(f);
      // A cut's own jump does not count as the picture's usual change.
      this.diffHist.push(change ? dBase : diff);
      this.deathHist.push(change ? fBase : frac);
      if (this.diffHist.length > CHANGE_HISTORY) { this.diffHist.shift(); this.deathHist.shift(); }
    }
    this.prevThumb!.data.set(this.thumb!.data);
    this.spawn(cur, f);
    const kfFrames = Math.max(1, Math.round(this.o.keyframeInterval * this.o.fps));
    const lastKf = this.keyframes.length ? this.keyframes[this.keyframes.length - 1] : -Infinity;
    // Content that turns over quickly gets keyframes sooner: once half of
    // what was tracked at the last keyframe is gone, its features are stale.
    const turnover = this.kfAlive >= 20 && this.kfLost >= TURNOVER * this.kfAlive && f - lastKf >= 2;
    if (f === 0 || change || turnover || f - lastKf >= kfFrames) this.keyframe(cur.levels[0], f);
    this.f++;
  }

  private setup(input: GrayImage): void {
    const long0 = Math.max(input.width, input.height), maxDim = this.o.maxDim;
    const s = long0 > maxDim ? maxDim / long0 : 1;
    this.W = Math.round(input.width * s);
    this.H = Math.round(input.height * s);
    const W = this.W, H = this.H;
    if (Math.min(W, H) < 64) throw new Error(`video too small (${W}x${H})`);
    this.base = createGray(W, H);
    this.pyr = [new Pyramid(W, H, 4), new Pyramid(W, H, 4)];
    const tw = 48, th = Math.max(8, Math.round((48 * H) / W));
    this.thumb = createGray(tw, th);
    this.prevThumb = createGray(tw, th);
    for (let k = 0; ; k++) {
      const sc = Math.pow(LEVEL_STEP, -k);
      const lw = Math.round(W * sc), lh = Math.round(H * sc);
      if (Math.max(lw, lh) < this.o.minLevelSize || Math.min(lw, lh) < 2 * ORB_BORDER + 8) break;
      this.levelImgs.push(createGray(lw, lh));
      this.levelSmooth.push(createGray(lw, lh));
      this.levelScales.push(W / lw);
    }
    this.det = new FastDetector(W, H, 60000, this.o.maxPerLevel);
  }

  /** Follow alive tracks into frame f; returns how many ended. */
  private step(prev: Pyramid, cur: Pyramid, f: number): number {
    const klt = this.klt, W = this.W, H = this.H;
    const P0 = prev.levels[0], C0 = cur.levels[0];
    const still: Track[] = [];
    let died = 0;
    for (const t of this.alive) {
      const k = t.xs.length - 1;
      const x = t.xs[k], y = t.ys[k];
      let ok = klt.trackPoint(prev, cur, x, y, x + t.vx, y + t.vy);
      if (!ok && (t.vx || t.vy)) ok = klt.trackPoint(prev, cur, x, y, x, y);
      let nx = klt.x, ny = klt.y;
      if (ok) {
        ok = klt.trackPoint(cur, prev, nx, ny, x, y) && Math.hypot(klt.x - x, klt.y - y) <= FB_MAX;
      }
      if (ok) ok = nx >= BORDER && ny >= BORDER && nx <= W - 1 - BORDER && ny <= H - 1 - BORDER;
      if (ok) {
        sample9(P0, x, y, this.s9a);
        const sd = sample9(C0, nx, ny, this.s9b);
        ok = sd >= 2.5 && ncc81(this.s9a, this.s9b) >= STEP_NCC;
      }
      if (!ok) {
        t.alive = false;
        died++;
        if (t.kf === this.keyframes.length - 1) this.kfLost++;
        continue;
      }
      // Snap sub-0.02 px jitter so stationary content stays exactly still.
      if (Math.abs(nx - x) < 0.02 && Math.abs(ny - y) < 0.02) { nx = x; ny = y; }
      t.xs.push(nx); t.ys.push(ny);
      t.vx = nx - x; t.vy = ny - y;
      if (ncc81(t.ref, this.s9b) < SNAP_NCC) this.snapshot(t, cur, f, nx, ny);
      still.push(t);
    }
    this.alive = still;
    return died;
  }

  /** Start tracks on corners in cells with few alive tracks. */
  private spawn(cur: Pyramid, f: number): void {
    const W = this.W, H = this.H, o = this.o;
    if (this.alive.length >= o.maxActiveTracks) return;
    const half = cur.levels[1];
    const { cands, score } = cornerCandidates(half, Math.ceil(BORDER / 2) + 4);
    if (!cands.length) return;
    cands.sort((a, b) => score[b] - score[a]);
    const cell = Math.max(16, Math.round(Math.max(W, H) / 24));
    const gw = Math.ceil(W / cell), gh = Math.ceil(H / cell);
    const occ = new Uint8Array(gw * gh);
    const grid = new Map<number, Track[]>();
    const cellOf = (x: number, y: number) => Math.min(gh - 1, Math.max(0, (y / cell) | 0)) * gw + Math.min(gw - 1, Math.max(0, (x / cell) | 0));
    for (const t of this.alive) {
      const k = t.xs.length - 1, c = cellOf(t.xs[k], t.ys[k]);
      occ[c]++;
      let l = grid.get(c);
      if (!l) grid.set(c, (l = []));
      l.push(t);
    }
    const hw = half.width;
    const C0 = cur.levels[0];
    for (const p of cands) {
      if (this.alive.length >= o.maxActiveTracks) break;
      const x = 2 * (p % hw), y = 2 * ((p / hw) | 0);
      if (x < BORDER || y < BORDER || x > W - 1 - BORDER || y > H - 1 - BORDER) continue;
      const c = cellOf(x, y);
      if (occ[c] >= 2) continue;
      let near = false;
      const cx = c % gw, cy = (c / gw) | 0;
      for (let j = -1; j <= 1 && !near; j++) for (let i = -1; i <= 1 && !near; i++) {
        if (cx + i < 0 || cy + j < 0 || cx + i >= gw || cy + j >= gh) continue;
        for (const t of grid.get((cy + j) * gw + cx + i) ?? []) {
          const k = t.xs.length - 1;
          if (Math.abs(t.xs[k] - x) < 6 && Math.abs(t.ys[k] - y) < 6) { near = true; break; }
        }
      }
      if (near) continue;
      if (sample9(C0, x, y, this.s9b) < 3 || selfSimilar(C0, x, y)) continue;
      const t: Track = {
        start: f, xs: [x], ys: [y], snapFrames: [], snaps: [], ref: new Float32Array(81), firstRef: this.s9b.slice(),
        score: score[p], vx: 0, vy: 0, alive: true, kf: -1,
      };
      this.snapshot(t, cur, f, x, y);
      this.tracks.push(t);
      this.alive.push(t);
      occ[c]++;
      let l = grid.get(c);
      if (!l) grid.set(c, (l = []));
      l.push(t);
    }
  }

  /**
   * Track points alive at the last frame on into the first frames (unwrapped
   * frame numbers), short of their own start. They stop at staggered frames,
   * so the runtime hands over to the points that start there gradually.
   */
  private followPastSeam(ts: Track[]): void {
    const N = this.f, K = Math.min(this.head.length, N - 1);
    const stop = new Map(ts.map((t, i) => [t, Math.min(t.start, Math.ceil(K * (0.3 + 0.7 * ((i * 0.618034) % 1))))]));
    this.alive = ts;
    for (let k = 0; k < K && this.alive.length; k++) {
      this.alive = this.alive.filter((t) => stop.get(t)! > k);
      const f = N + k, cur = this.pyr[f & 1], prev = this.pyr[(f + 1) & 1];
      cur.build(this.head[k]);
      this.step(prev, cur, f);
    }
    this.alive = [];
    this.head = [];
  }

  private snapshot(t: Track, cur: Pyramid, f: number, x: number, y: number): void {
    const P = this.o.patchSize, c = (P - 1) / 2;
    const data = new Uint8Array(PATCH_LEVELS * P * P);
    for (let l = 0; l < PATCH_LEVELS; l++) {
      const img = cur.levels[l], s = 1 << l;
      const lx = x / s - c, ly = y / s - c;
      let o = l * P * P;
      for (let j = 0; j < P; j++) for (let i = 0; i < P; i++) data[o++] = Math.round(bilinear(img, lx + i, ly + j));
    }
    t.snapFrames.push(f);
    t.snaps.push(data);
    sample9(cur.levels[0], x, y, t.ref);
  }

  private keyframe(img: GrayImage, f: number): void {
    const kf = this.keyframes.length;
    this.keyframes.push(f);
    for (const t of this.alive) t.kf = kf;
    this.kfAlive = this.alive.length;
    this.kfLost = 0;
    const feats = this.kfFeats, det = this.det!, o = this.o;
    feats.clear();
    const levelOf: number[] = [];
    for (let k = 0; k < this.levelImgs.length; k++) {
      const lev = k === 0 ? img : resizeArea(img, this.levelImgs[k].width, this.levelImgs[k].height, this.levelImgs[k]);
      const smooth = blur5(lev, this.levelSmooth[k]);
      const n = det.detect(lev, {
        threshold: o.fastThreshold,
        maxPoints: o.maxPerLevel,
        border: ORB_BORDER,
        cell: Math.max(16, Math.round(Math.max(lev.width, lev.height) / 12)),
      });
      const before = feats.count;
      describeLevel(lev, smooth, det.x, det.y, n, this.levelScales[k], 0.5, feats);
      for (let i = before; i < feats.count; i++) levelOf.push(k);
    }
    if (kf === 0) this.kf0 = { kp: feats.kp.slice(0, 4 * feats.count), desc: feats.desc.slice(0, 8 * feats.count), n: feats.count };
    // Coverage of this keyframe on an 8x8 grid.
    const grid = new Uint8Array(64);
    for (let i = 0; i < feats.count; i++) {
      const gx = Math.min(7, Math.max(0, Math.floor((feats.kp[4 * i] / this.W) * 8)));
      const gy = Math.min(7, Math.max(0, Math.floor((feats.kp[4 * i + 1] / this.H) * 8)));
      grid[gy * 8 + gx] = 1;
    }
    this.kfCoverage.push(grid.reduce((a, b) => a + b, 0) / 64);
    this.kfCounts.push(feats.count);
    // Merge with features seen in the last few keyframes of this shot (a
    // static corner can drop out of a keyframe when moving content wins its
    // detector cell, and come back).
    const shotStart = this.changes.length ? this.changes[this.changes.length - 1] : 0;
    let since = kf - 1;
    while (since > 0 && kf - since < MERGE_GAP && this.keyframes[since - 1] >= shotStart) since--;
    const open = new Map<number, number[]>();
    const key = (lv: number, x: number, y: number) => (lv * 4096 + (y >> 2)) * 4096 + (x >> 2);
    const all = this.all;
    for (let i = 0; i < all.count; i++) {
      if (this.last[i] < since || this.last[i] === kf) continue;
      const k = key(this.level[i], all.kp[4 * i], all.kp[4 * i + 1]);
      let l = open.get(k);
      if (!l) open.set(k, (l = []));
      l.push(i);
    }
    const fresh: number[] = [];
    for (let q = 0; q < feats.count; q++) {
      const x = feats.kp[4 * q], y = feats.kp[4 * q + 1], lv = levelOf[q];
      const tol = Math.max(1, 0.75 * feats.kp[4 * q + 2]);
      let best = -1, bestD = 25;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        for (const i of open.get(key(lv, x + 4 * dx, y + 4 * dy)) ?? []) {
          if (this.last[i] === kf) continue;
          if (Math.abs(all.kp[4 * i] - x) > tol || Math.abs(all.kp[4 * i + 1] - y) > tol) continue;
          const d = hamming(all.desc, 8 * i, feats.desc, 8 * q);
          if (d < bestD) { bestD = d; best = i; }
        }
      }
      if (best >= 0) { this.last[best] = kf; continue; }
      fresh.push(q);
    }
    // New features between cuts are mostly moving content, which only
    // matches near this keyframe's time: keep an evenly spread subset, large
    // enough that the keyframe can be recognised when little else carries over.
    const isChange = kf === 0 || this.changes.includes(f);
    const cap = isChange ? fresh.length : Math.max(o.maxNewPerKeyframe, o.minKeyframeFeatures - (feats.count - fresh.length));
    const stride = Math.max(1, fresh.length / Math.max(1, cap));
    for (let r = 0; r < fresh.length; r += stride) {
      const q = fresh[Math.floor(r)], lv = levelOf[q];
      const i = all.count;
      all.reserve(i + 1);
      all.kp.set(feats.kp.subarray(4 * q, 4 * q + 4), 4 * i);
      all.desc.set(feats.desc.subarray(8 * q, 8 * q + 8), 8 * i);
      all.count++;
      this.first.push(kf); this.last.push(kf); this.level.push(lv);
    }
  }

  finish(): TargetData {
    const N = this.f, o = this.o, W = this.W, H = this.H;
    if (N < 2) throw new Error('need at least two frames');
    const ending = this.alive;
    for (const t of this.alive) t.alive = false;
    this.alive = [];
    const endOf = (t: Track) => t.start + t.xs.length - 1;
    let tracks = this.tracks;
    // Join tracks that continue through the loop seam.
    if (o.loop) {
      const starts = tracks.filter((t) => t.start === 0 && endOf(t) < N - 1);
      const used = new Set<Track>();
      for (const e of tracks) {
        if (endOf(e) !== N - 1 || e.start === 0) continue;
        const k = e.xs.length - 1;
        const px = e.xs[k] + e.vx, py = e.ys[k] + e.vy;
        let best: Track | null = null, bestD = 0.75;
        for (const s of starts) {
          if (used.has(s) || endOf(s) >= e.start) continue;
          const d = Math.hypot(s.xs[0] - px, s.ys[0] - py);
          if (d < bestD && ncc81(e.ref, s.firstRef) >= 0.8) { bestD = d; best = s; }
        }
        if (!best) continue;
        used.add(best);
        e.xs.push(...best.xs); e.ys.push(...best.ys);
        for (let i = 0; i < best.snaps.length; i++) {
          e.snapFrames.push(best.snapFrames[i] + N);
          e.snaps.push(best.snaps[i]);
        }
      }
      tracks = tracks.filter((t) => !used.has(t));
      // Moving content rarely lines up with a point started at frame 0, so
      // follow the rest into the clip's first frames again.
      this.followPastSeam(ending.filter((t) => endOf(t) === N - 1));
    }
    // Candidates: long enough; stationary or not.
    const minLen = Math.max(4, Math.round(0.15 * o.fps));
    interface Cand { t: Track; stationary: boolean; q: number }
    const cands: Cand[] = [];
    for (const t of tracks) {
      const n = t.xs.length;
      if (n < minLen && n < N) continue;
      let dev = 0;
      for (let i = 1; i < n; i++) dev = Math.max(dev, Math.abs(t.xs[i] - t.xs[0]), Math.abs(t.ys[i] - t.ys[0]));
      cands.push({ t, stationary: dev <= STATIONARY_PX, q: Math.min(n, N) * Math.sqrt(t.score) });
    }
    // Selection: keep each frame's points spread over a grid.
    const gw = 8, gh = Math.max(4, Math.round((8 * H) / W)), G = gw * gh;
    const cap = Math.max(2, Math.ceil((o.pointsPerFrame / G) * 1.5));
    const counts = new Uint8Array(N * G), total = new Uint16Array(N);
    const cellAt = (x: number, y: number) => Math.min(gh - 1, Math.max(0, Math.floor((y / H) * gh))) * gw + Math.min(gw - 1, Math.max(0, Math.floor((x / W) * gw)));
    const chosen: Cand[] = [];
    const tryAccept = (c: Cand, capFactor: number): boolean => {
      const t = c.t, n = Math.min(t.xs.length, N);
      let ok = 0;
      for (let i = 0; i < n; i++) {
        const fr = (t.start + i) % N;
        if (counts[fr * G + cellAt(t.xs[i], t.ys[i])] < cap * capFactor && total[fr] < o.pointsPerFrame * 1.25) ok++;
      }
      if (ok < 0.5 * n) return false;
      for (let i = 0; i < n; i++) {
        const fr = (t.start + i) % N;
        counts[fr * G + cellAt(t.xs[i], t.ys[i])]++;
        total[fr]++;
      }
      chosen.push(c);
      return true;
    };
    cands.sort((a, b) => b.q - a.q);
    // Moving points first (with a lower cap): they are rarer and carry the timing.
    const taken = new Set<Cand>();
    for (const c of cands) if (!c.stationary && tryAccept(c, 0.5)) taken.add(c);
    for (const c of cands) if (!taken.has(c)) tryAccept(c, 1);
    // Interleave by position so the runtime's rotating cursors spread out.
    chosen.sort((a, b) => spreadKey(a.t, W, H) - spreadKey(b.t, W, H));
    // Pack points, paths and patches.
    const P = o.patchSize, patchBytes = PATCH_LEVELS * P * P;
    const pts = new Uint32Array(chosen.length * VPT_STRIDE);
    const path: number[] = [];
    const pidx: number[] = [];
    let nSnaps = 0;
    for (const c of chosen) nSnaps += c.t.snaps.length;
    const patches = new Uint8Array(nSnaps * patchBytes);
    let snapAt = 0, moving = 0;
    const perFrame = new Uint16Array(N);
    const informative = new Uint8Array(N);
    chosen.forEach((c, j) => {
      const t = c.t, n = t.xs.length, o8 = j * VPT_STRIDE;
      const start = t.start, end = start + n - 1;
      pts[o8] = start;
      pts[o8 + 1] = end;
      pts[o8 + 2] = path.length / 3;
      if (c.stationary) {
        let mx = 0, my = 0;
        for (let i = 0; i < n; i++) { mx += t.xs[i]; my += t.ys[i]; }
        path.push(start, mx / n, my / n);
      } else {
        moving++;
        for (const i of simplifyPath(t.xs, t.ys, PATH_TOL)) path.push(start + i, t.xs[i], t.ys[i]);
      }
      pts[o8 + 3] = path.length / 3 - pts[o8 + 2];
      pts[o8 + 4] = pidx.length / 2;
      for (let s = 0; s < t.snaps.length; s++) {
        patches.set(t.snaps[s], snapAt * patchBytes);
        pidx.push(t.snapFrames[s], snapAt * patchBytes);
        snapAt++;
        if (s > 0) informative[t.snapFrames[s] % N] = 1;
      }
      pts[o8 + 5] = t.snaps.length;
      pts[o8 + 6] = c.stationary ? VPT_STATIONARY : 0;
      for (let i = 0; i < Math.min(n, N); i++) {
        perFrame[(start + i) % N]++;
        if (!c.stationary) informative[(start + i) % N] = 1;
      }
      if (n < N) { informative[start % N] = 1; informative[end % N] = 1; }
    });
    // Features: merge across the loop seam, cap the count, build the tree.
    const nk = this.keyframes.length;
    const all = this.all;
    const keep = new Uint8Array(all.count).fill(1);
    if (o.loop && nk > 1) {
      const heads = new Map<number, number[]>();
      const hkey = (lv: number, x: number, y: number) => (lv * 4096 + (y >> 3)) * 4096 + (x >> 3);
      for (let b = 0; b < all.count; b++) {
        if (this.first[b] !== 0 || this.last[b] === nk - 1) continue;
        const k = hkey(this.level[b], all.kp[4 * b], all.kp[4 * b + 1]);
        let l = heads.get(k);
        if (!l) heads.set(k, (l = []));
        l.push(b);
      }
      for (let a = 0; a < all.count; a++) {
        if (this.last[a] !== nk - 1 || this.first[a] === 0) continue;
        const near: number[] = [];
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          near.push(...(heads.get(hkey(this.level[a], all.kp[4 * a] + 8 * dx, all.kp[4 * a + 1] + 8 * dy)) ?? []));
        }
        for (const b of near) {
          if (!keep[b] || this.last[b] >= this.first[a]) continue;
          const tol = Math.max(1, 0.75 * all.kp[4 * a + 2]);
          if (Math.abs(all.kp[4 * a] - all.kp[4 * b]) > tol || Math.abs(all.kp[4 * a + 1] - all.kp[4 * b + 1]) > tol) continue;
          if (hamming(all.desc, 8 * a, all.desc, 8 * b) >= 25) continue;
          this.last[a] = nk + this.last[b];
          keep[b] = 0;
          break;
        }
      }
    }
    let order: number[] = [];
    for (let i = 0; i < all.count; i++) if (keep[i]) order.push(i);
    if (order.length > o.maxFeatures) {
      order.sort((a, b) => (this.last[b] - this.first[b]) - (this.last[a] - this.first[a]) || a - b);
      order = order.slice(0, o.maxFeatures).sort((a, b) => a - b);
    }
    const nf = order.length;
    const kp = new Float32Array(4 * nf), desc = new Uint32Array(8 * nf), range = new Uint16Array(2 * nf);
    const perLevel = new Array(this.levelScales.length).fill(0);
    order.forEach((i, k) => {
      kp.set(all.kp.subarray(4 * i, 4 * i + 4), 4 * k);
      desc.set(all.desc.subarray(8 * i, 8 * i + 8), 8 * k);
      range[2 * k] = this.first[i];
      range[2 * k + 1] = this.last[i];
      perLevel[this.level[i]]++;
    });
    const tree = buildKmTree(desc, nf);
    // Quality report.
    const secs = Math.ceil(N / o.fps);
    const pointsPerSecond: number[] = [], timingPerSecond: boolean[] = [];
    let minPoints = Infinity, minAt = 0;
    for (let s = 0; s < secs; s++) {
      let m = Infinity, inf = false;
      for (let fr = Math.round(s * o.fps); fr < Math.min(N, Math.round((s + 1) * o.fps)); fr++) {
        if (perFrame[fr] < m) m = perFrame[fr];
        if (perFrame[fr] < minPoints) { minPoints = perFrame[fr]; minAt = fr; }
        if (informative[fr]) inf = true;
      }
      pointsPerSecond.push(m === Infinity ? 0 : m);
      timingPerSecond.push(inf);
    }
    const coverage = this.kfCoverage.reduce((a, b) => a + b, 0) / Math.max(1, nk);
    const warnings: string[] = [];
    const fewest = Math.min(...this.kfCounts);
    if (fewest < 150) warnings.push(`few features (${fewest}) at ${(this.keyframes[this.kfCounts.indexOf(fewest)] / o.fps).toFixed(1)} s; use more detailed graphics`);
    if (coverage < 0.5) warnings.push(`features cover only ${(coverage * 100).toFixed(0)}% of the frame`);
    if (minPoints < 40) warnings.push(`only ${minPoints} tracking points at ${(minAt / o.fps).toFixed(1)} s; tracking may drop there`);
    const k0 = this.kf0!;
    const selfSim = selfSimilarity(k0.kp, k0.desc, k0.n, W, H);
    if (selfSim > 0.25) warnings.push(`repetitive texture (${(selfSim * 100).toFixed(0)}% of features have look-alikes)`);
    const quality: QualityReport = {
      features: nf, featuresPerLevel: perLevel, coverage, selfSimilarity: selfSim, trackingPoints: chosen.length, warnings,
    };
    const video: VideoMeta = {
      fps: o.fps, frameCount: N, duration: N / o.fps, loop: o.loop,
      keyframes: this.keyframes.slice(), changes: this.changes.slice(),
      patchSize: P, patchScales: [1, 2, 4].slice(0, PATCH_LEVELS),
      report: {
        points: chosen.length, movingPoints: moving, keyframes: nk, minPoints: minPoints === Infinity ? 0 : minPoints,
        pointsPerSecond, timingPerSecond,
      },
    };
    return {
      meta: { name: o.name, width: W, height: H, physicalWidth: o.physicalWidth, levelScales: this.levelScales.slice(), images: [], quality, video },
      kp, desc, tree, images: [], trackPoints: new Float32Array(0),
      video: {
        featureRange: range,
        points: pts,
        path: Float32Array.from(path),
        patchIndex: Uint32Array.from(pidx),
        patches,
      },
    };
  }
}

/** Compile a video target from its frames in order. */
export function compileVideoTarget(frames: Iterable<GrayImage>, o: VideoCompileOptions): TargetData {
  const b = new VideoTargetBuilder(o);
  for (const f of frames) b.addFrame(f);
  return b.finish();
}

function median(xs: number[]): number {
  if (!xs.length) return 0;
  const a = xs.slice().sort((x, y) => x - y), m = a.length >> 1;
  return a.length & 1 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

/** Bit-interleaved grid position, so consecutive points are far apart. */
function spreadKey(t: Track, W: number, H: number): number {
  const gx = Math.min(15, Math.floor((t.xs[0] / W) * 16)), gy = Math.min(15, Math.floor((t.ys[0] / H) * 16));
  let k = 0;
  for (let b = 0; b < 4; b++) k |= (((gx >> b) & 1) << (2 * b + 1)) | (((gy >> b) & 1) << (2 * b));
  // Reverse the bits so neighbours in order land in different quadrants.
  let r = 0;
  for (let b = 0; b < 8; b++) r |= ((k >> b) & 1) << (7 - b);
  return r * 1e6 + t.start;
}

/** Douglas-Peucker on a per-frame path; returns kept indices (first and last always). */
export function simplifyPath(xs: number[], ys: number[], tol: number): number[] {
  const n = xs.length;
  if (n <= 2) return n === 1 ? [0] : [0, 1];
  const keep = new Uint8Array(n);
  keep[0] = keep[n - 1] = 1;
  const stack: Array<[number, number]> = [[0, n - 1]];
  while (stack.length) {
    const [a, b] = stack.pop()!;
    let worst = -1, wd = tol;
    for (let i = a + 1; i < b; i++) {
      const s = (i - a) / (b - a);
      const d = Math.max(Math.abs(xs[a] + s * (xs[b] - xs[a]) - xs[i]), Math.abs(ys[a] + s * (ys[b] - ys[a]) - ys[i]));
      if (d > wd) { wd = d; worst = i; }
    }
    if (worst >= 0) {
      keep[worst] = 1;
      stack.push([a, worst], [worst, b]);
    }
  }
  const out: number[] = [];
  for (let i = 0; i < n; i++) if (keep[i]) out.push(i);
  return out;
}

function bilinear(img: GrayImage, x: number, y: number): number {
  const w = img.width, h = img.height, d = img.data;
  x = x < 0 ? 0 : x > w - 1 ? w - 1 : x;
  y = y < 0 ? 0 : y > h - 1 ? h - 1 : y;
  const x0 = Math.min(w - 2, Math.floor(x)), y0 = Math.min(h - 2, Math.floor(y));
  const ax = x - x0, ay = y - y0, i = y0 * w + x0;
  const top = d[i] + (d[i + 1] - d[i]) * ax, bot = d[i + w] + (d[i + w + 1] - d[i + w]) * ax;
  return top + (bot - top) * ay;
}

/** 9x9 bilinear patch around (x, y); returns its standard deviation. */
function sample9(img: GrayImage, x: number, y: number, out: Float32Array): number {
  let s = 0, s2 = 0, k = 0;
  for (let j = -4; j <= 4; j++) for (let i = -4; i <= 4; i++) {
    const v = bilinear(img, x + i, y + j);
    out[k++] = v; s += v; s2 += v * v;
  }
  const m = s / 81;
  return Math.sqrt(Math.max(0, s2 / 81 - m * m));
}

function ncc81(a: Float32Array, b: Float32Array): number {
  let sa = 0, sb = 0;
  for (let i = 0; i < 81; i++) { sa += a[i]; sb += b[i]; }
  const ma = sa / 81, mb = sb / 81;
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < 81; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    ab += x * y; aa += x * x; bb += y * y;
  }
  return aa > 1e-6 && bb > 1e-6 ? ab / Math.sqrt(aa * bb) : 0;
}
