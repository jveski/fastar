// Video clock: where a looping video is in its playback, from what the
// camera sees.
//
// The belief is a histogram over the playback offset b, with the video time
// at camera time t given by tv = (t - b) mod duration. b stays constant while
// the video plays, so time passing needs no prediction step beyond a little
// diffusion (clock drift, dropped frames), and the histogram can hold
// several hypotheses at once (content that repeats, or a held frame that
// only says "somewhere in these two seconds"). Measurements:
//   - intervals from acquisition (the matched features are on screen then),
//   - probes: how well the content predicted for each video frame matches
//     the camera image at the tracked pose (a log-likelihood per frame).
// Players often pause briefly at the loop point; at each seam the belief is
// widened toward later offsets, and the gap seen afterwards is learned.

export interface ClockOptions {
  /** Histogram bins per video frame. */
  binsPerFrame: number;
  /** Clock diffusion (s^2 per s). */
  diffusion: number;
  /** Floor of a single measurement's likelihood (robustness to outliers). */
  floor: number;
  /** Longest pause expected at the loop seam (s). */
  maxLoopGap: number;
  /**
   * Locked: main-mode std below this many frames. A camera running in step
   * with the video cannot see inside a frame (uniform over one frame: std
   * 0.29), so this must sit above that.
   */
  lockStdFrames: number;
  /** Likelihood outside an acquisition's time ranges (a wrong match). */
  rangeOutlier: number;
  /** A gap between updates longer than this (s), or time going back, resets the clock. */
  maxGap: number;
  /** A gap longer than this (s) halves the confidence. */
  softGap: number;
}

export const DEFAULT_CLOCK: ClockOptions = {
  binsPerFrame: 6,
  diffusion: 0.005 * 0.005,
  floor: 1e-3,
  maxLoopGap: 0.25,
  lockStdFrames: 0.5,
  rangeOutlier: 0.005,
  maxGap: 5,
  softGap: 0.5,
};

export interface ClockWindow {
  /** Estimated video time (s) in [0, duration). */
  tv: number;
  /** Credible interval of the main hypothesis (s); lo <= tv <= hi, possibly outside [0, duration). */
  lo: number;
  hi: number;
  std: number;
  /** Probability of the main hypothesis. */
  mass: number;
  locked: boolean;
}

export class VideoClock {
  readonly o: ClockOptions;
  readonly n: number;
  readonly binW: number;
  readonly p: Float64Array;
  private readonly L: Float64Array;
  private lastT = NaN;
  private pendingVar = 0;
  /** Video time of the main mode at the last update (seam detection). */
  private lastTv = NaN;
  private lastLocked = false;
  /** Offset before the last seam, and the pause measured since (refined while locked). */
  private seamOffset = NaN;
  private pendingGap = NaN;
  /** Offset while locked, away from the seam. */
  private steadyOffset = NaN;
  /** Pauses measured at earlier seams. */
  private readonly gaps: number[] = [];
  /** Measurements applied so far. */
  updates = 0;

  constructor(readonly duration: number, readonly fps: number, o: Partial<ClockOptions> = {}) {
    this.o = { ...DEFAULT_CLOCK, ...o };
    this.n = Math.max(8, Math.round(duration * fps * this.o.binsPerFrame));
    this.binW = duration / this.n;
    this.p = new Float64Array(this.n).fill(1 / this.n);
    this.L = new Float64Array(this.n);
  }

  /** Learned pause at the loop seam (s), NaN until seen. */
  get loopGap(): number {
    const g = this.gaps;
    if (!g.length) return this.pendingGap;
    let s = 0;
    for (const v of g) s += v;
    return s / g.length;
  }

  reset(): void {
    this.p.fill(1 / this.n);
    this.lastT = this.lastTv = this.seamOffset = this.pendingGap = this.steadyOffset = NaN;
    this.pendingVar = 0;
    this.updates = 0;
  }

  private mod(x: number): number {
    const D = this.duration;
    return x - D * Math.floor(x / D);
  }

  /** Video time of bin i at camera time t. */
  private tvOf(i: number, t: number): number {
    return this.mod(t - (i + 0.5) * this.binW);
  }

  /** Diffuse for the time since the last update, and widen at a loop seam. */
  advance(t: number): void {
    if (Number.isNaN(this.lastT)) { this.lastT = t; return; }
    const dt = t - this.lastT;
    if (dt === 0) return;
    this.lastT = t;
    if (dt < 0 || dt > this.o.maxGap) {
      // The camera clock jumped (or nothing was seen for a long time): start over.
      this.p.fill(1 / this.n);
      this.pendingVar = 0;
      this.lastTv = this.seamOffset = this.pendingGap = this.steadyOffset = NaN;
      this.lastLocked = false;
      return;
    }
    if (dt > this.o.softGap) {
      // A shorter gap: the app stalled (the estimate still holds) or the
      // camera's time base stepped (it does not). Keep half the belief.
      const u = 0.5 / this.n;
      for (let i = 0; i < this.n; i++) this.p[i] = 0.5 * this.p[i] + u;
    }
    this.pendingVar += this.o.diffusion * dt;
    const b2 = this.binW * this.binW;
    if (this.pendingVar > 25 * b2) {
      // A long gap (or a jump in the camera clock): one wide Gaussian, or nothing known.
      const sig = Math.sqrt(this.pendingVar);
      if (3 * sig >= 0.5 * this.duration) this.p.fill(1 / this.n);
      else this.gauss(sig / this.binW);
      this.pendingVar = 0;
    }
    while (this.pendingVar > 0.1 * b2) {
      const a = Math.min(0.25, this.pendingVar / (2 * b2));
      this.blur(a);
      this.pendingVar -= 2 * a * b2;
    }
    // Seam: the main mode's video time wrapped since the last update.
    if (this.lastLocked && !Number.isNaN(this.lastTv)) {
      const w = this.window(t);
      if (w.tv + 0.5 * this.duration < this.lastTv) {
        if (!Number.isNaN(this.pendingGap)) {
          this.gaps.push(this.pendingGap);
          if (this.gaps.length > 4) this.gaps.shift();
        }
        this.pendingGap = NaN;
        // From well before the seam: during the pause itself the probes
        // already start moving the offset.
        this.seamOffset = Number.isNaN(this.steadyOffset) ? this.offset() : this.steadyOffset;
        this.widenForGap();
      }
    }
  }

  private blur(a: number): void {
    const p = this.p, n = this.n, L = this.L;
    for (let i = 0; i < n; i++) L[i] = a * (p[(i + n - 1) % n] + p[(i + 1) % n]) + (1 - 2 * a) * p[i];
    p.set(L);
  }

  /** Circular Gaussian blur, sigma in bins. */
  private gauss(sig: number): void {
    const p = this.p, n = this.n, L = this.L;
    const R = Math.ceil(3 * sig), k = new Float64Array(2 * R + 1);
    let ks = 0;
    for (let i = -R; i <= R; i++) ks += (k[i + R] = Math.exp(-0.5 * (i / sig) ** 2));
    L.fill(0);
    for (let i = 0; i < n; i++) {
      const v = p[i] / ks;
      if (v < 1e-15) continue;
      for (let j = -R; j <= R; j++) L[(((i + j) % n) + n) % n] += v * k[j + R];
    }
    p.set(L);
  }

  /** A pause g at the seam moves the offset by +g. */
  private widenForGap(): void {
    const p = this.p, n = this.n, L = this.L;
    const maxB = Math.max(1, Math.round(this.o.maxLoopGap / this.binW));
    const kernel = new Float64Array(maxB + 1);
    const known = !Number.isNaN(this.loopGap);
    const g0 = known ? this.loopGap / this.binW : 0;
    const sig = Math.max(1, 0.01 / this.binW);
    for (let k = 0; k <= maxB; k++) kernel[k] = (known ? 0.3 : 0.5) / (maxB + 1) + (known ? 0.7 : 0.5) * Math.exp(-0.5 * ((k - g0) / sig) ** 2) / (sig * Math.sqrt(2 * Math.PI));
    let ks = 0;
    for (const v of kernel) ks += v;
    L.fill(0);
    for (let i = 0; i < n; i++) {
      const v = p[i];
      if (v < 1e-12) continue;
      for (let k = 0; k <= maxB; k++) L[(i + k) % n] += (v * kernel[k]) / ks;
    }
    p.set(L);
  }

  /** The video showed the matched content at camera time t during these video-time ranges ([start, end) pairs, s). */
  addRanges(t: number, ranges: ArrayLike<number>, margin = 1 / this.fps): void {
    this.advance(t);
    const L = this.L, eps = this.o.rangeOutlier;
    for (let i = 0; i < this.n; i++) {
      const tv = this.tvOf(i, t);
      let inside = false;
      for (let r = 0; r + 1 < ranges.length && !inside; r += 2) {
        const a = ranges[r] - margin, len = ranges[r + 1] - ranges[r] + 2 * margin;
        inside = len >= this.duration || this.mod(tv - a) <= len;
      }
      L[i] = inside ? 1 : eps;
    }
    this.apply(t);
  }

  /**
   * Evidence from probes for camera time t: a log-likelihood per video frame
   * (length frameCount; 0 = no information about that frame).
   */
  addFrameLikelihood(t: number, llr: ArrayLike<number>): void {
    const N = llr.length;
    let mx = -Infinity, mn = Infinity;
    for (let f = 0; f < N; f++) { if (llr[f] > mx) mx = llr[f]; if (llr[f] < mn) mn = llr[f]; }
    this.advance(t);
    if (!(mx - mn > 1e-9)) return;
    const L = this.L, fl = this.o.floor, fps = this.fps;
    for (let i = 0; i < this.n; i++) {
      let fi = Math.floor(this.tvOf(i, t) * fps);
      if (fi >= N) fi = N - 1;
      L[i] = Math.max(fl, Math.exp(llr[fi] - mx));
    }
    this.apply(t);
  }

  private apply(t: number): void {
    const p = this.p, L = this.L, n = this.n;
    let s = 0;
    for (let i = 0; i < n; i++) s += (p[i] *= L[i]);
    if (!(s > 0)) { p.fill(1 / n); return; }
    // Normalize, keeping a trace of probability everywhere so a seek or a
    // restarted player can still win later.
    const u = 1e-6 / n;
    for (let i = 0; i < n; i++) p[i] = p[i] / s + u;
    this.updates++;
    const w = this.window(t);
    this.lastTv = w.tv;
    this.lastLocked = w.locked;
    if (w.locked && w.tv < this.duration - this.o.maxLoopGap - 0.25) this.steadyOffset = this.offset();
    if (w.locked && !Number.isNaN(this.seamOffset)) {
      let g = this.offset() - this.seamOffset;
      g -= this.duration * Math.round(g / this.duration);
      if (g > -2 * this.binW && g < this.o.maxLoopGap + 2 * this.binW) this.pendingGap = Math.max(0, g);
    }
  }

  /** Mean offset b (s) of the main hypothesis. */
  offset(): number {
    return this.mode().mean;
  }

  /** Main mode: mean offset (wrapped), and mean/lo/hi on the same unwrapped axis. */
  private mode(): { mean: number; meanU: number; std: number; mass: number; lo: number; hi: number } {
    const p = this.p, n = this.n;
    let pmax = 0;
    for (let i = 0; i < n; i++) if (p[i] > pmax) pmax = p[i];
    const thr = 1e-3 * pmax;
    // Contiguous circular runs above the threshold; keep the heaviest.
    let start = -1;
    for (let i = 0; i < n; i++) if (p[i] < thr) { start = i; break; }
    let best = { a: 0, len: n, mass: 1 };
    if (start >= 0) {
      best = { a: 0, len: 0, mass: -1 };
      let i = 1;
      while (i <= n) {
        const k = (start + i) % n;
        if (p[k] < thr) { i++; continue; }
        let len = 0, mass = 0;
        while (len < n && p[(k + len) % n] >= thr) { mass += p[(k + len) % n]; len++; }
        if (mass > best.mass) best = { a: k, len, mass };
        i += len;
      }
    } else {
      // Above the threshold everywhere: centre the run on the peak.
      let im = 0;
      for (let i = 0; i < n; i++) if (p[i] > p[im]) im = i;
      best = { a: (im + Math.ceil(n / 2)) % n, len: n, mass: 1 };
    }
    let m = 0, m2 = 0, mass = 0;
    for (let j = 0; j < best.len; j++) {
      const v = p[(best.a + j) % n];
      m += v * j; m2 += v * j * j; mass += v;
    }
    m /= mass;
    const sd = Math.sqrt(Math.max(0, m2 / mass - m * m));
    // 2.5% tails.
    let acc = 0, lo = 0, hi = best.len - 1;
    for (let j = 0; j < best.len; j++) { acc += p[(best.a + j) % n]; if (acc >= 0.025 * mass) { lo = j; break; } }
    acc = 0;
    for (let j = best.len - 1; j >= 0; j--) { acc += p[(best.a + j) % n]; if (acc >= 0.025 * mass) { hi = j; break; } }
    const W = this.binW;
    return {
      mean: this.mod((best.a + m + 0.5) * W),
      meanU: (best.a + m + 0.5) * W,
      std: sd * W,
      mass,
      lo: (best.a + lo) * W,
      hi: (best.a + hi + 1) * W,
    };
  }

  /**
   * Candidate frames (ascending, unwrapped) covering the main hypothesis at
   * camera time t, at least `margin` frames either side of the estimate,
   * strided to at most `max` candidates. Vary `phase` between calls so
   * strided candidates cover every frame over a few calls.
   */
  candidates(t: number, out: Int32Array | Float64Array, max = out.length, margin = 2, phase = 0): number {
    const w = this.window(t), fps = this.fps;
    const fc = Math.floor(w.tv * fps);
    const f0 = Math.min(Math.floor(w.lo * fps), fc - margin), f1 = Math.max(Math.floor(w.hi * fps), fc + margin);
    const span = Math.min(f1 - f0, Math.round(this.duration * fps) - 1);
    const stride = Math.max(1, Math.ceil((span + 1) / max));
    let n = 0;
    for (let f = f0 + (phase % stride); f <= f0 + span && n < max; f += stride) out[n++] = f;
    return n;
  }

  /** Candidate frames over the whole clip, strided to at most `max`, starting at `phase`. */
  sweep(out: Int32Array | Float64Array, max = out.length, phase = 0): number {
    const N = Math.round(this.duration * this.fps);
    const stride = Math.max(1, Math.ceil(N / max));
    let n = 0;
    for (let f = phase % stride; f < N && n < max; f += stride) out[n++] = f;
    return n;
  }

  /** Estimate for camera time t (no update). */
  window(t: number): ClockWindow {
    const md = this.mode();
    const tv = this.mod(t - md.mean);
    // Later offsets mean earlier video times.
    const loTv = tv - (md.hi - md.meanU), hiTv = tv + (md.meanU - md.lo);
    return {
      tv, lo: Math.min(tv, loTv), hi: Math.max(tv, hiTv), std: md.std, mass: md.mass,
      locked: md.std < this.o.lockStdFrames / this.fps && md.mass > 0.8,
    };
  }
}

/** Video time (s, in [0, duration)) at time t (s) from a clock offset. */
export function videoTimeFromOffset(offset: number, duration: number, t: number): number {
  const x = t - offset;
  return x - duration * Math.floor(x / duration);
}
