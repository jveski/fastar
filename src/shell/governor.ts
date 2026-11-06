// Performance governor: watches how loaded the page and the tracker are,
// and steps quality down quickly when they cannot keep up, back up slowly
// once they can (and more slowly each time that turns out too soon). Its
// levels set how much time acquisition may take while something is tracked,
// and hints for the app's own rendering (pixel ratio, effects).
//
// Two ladders: one for the page (time spent handling frames, including the
// app's rendering in onFrame, and late display frames, which also catch
// GPU-bound rendering), one for tracking (dropped camera frames, tracking
// every 2nd frame). The hints follow the page's ladder; acquisition follows
// whichever is lower, since it competes with both for the CPU.

export type Bottleneck = 'none' | 'render' | 'tracking';
export type Effects = 'full' | 'reduced' | 'minimal';

export interface PerformanceState {
  /** 0: full quality; higher: more throttled (at most 3). */
  level: number;
  /** What is holding the frame rate back. */
  bottleneck: Bottleneck;
  /** Suggested device pixel ratio for rendering (already capped by the device's). */
  pixelRatio: number;
  /** Suggested level of optional effects (shadows, post-processing, particles...). */
  effects: Effects;
  /** Camera frame interval (ms). */
  frameInterval: number;
  /** Camera frames per tracked frame (2 when the device cannot track every frame). */
  stride: number;
  /** Share of the time the page spends handling camera frames (including onFrame listeners). */
  mainBusy: number;
  /** Share of display frames that came late. */
  jank: number;
  /** Share of recent tracked frames after which camera frames were dropped. */
  trackingOverload: number;
  /** Share of the time the acquisition worker is busy. */
  acquireBusy: number;
  /** syncDisplay: delay added to the display to wait for tracking results (ms), and share of frames shown before theirs arrived. */
  displayDelay: number;
  displayLate: number;
}

interface Level {
  pixelRatio: number;
  effects: Effects;
  acquireDuty: number;
}

export const LEVELS: readonly Level[] = [
  { pixelRatio: Infinity, effects: 'full', acquireDuty: 0.5 },
  { pixelRatio: 2, effects: 'full', acquireDuty: 0.35 },
  { pixelRatio: 1.5, effects: 'reduced', acquireDuty: 0.2 },
  { pixelRatio: 1, effects: 'minimal', acquireDuty: 0.1 },
];

export interface GovernorOptions {
  /** Measurement window (ms). */
  window: number;
  /** Ignore the first ms (warm-up: compiling, the first acquisitions). */
  warmup: number;
  /** Calm time (ms) before a level is raised back, its cap, and the time after one within which trouble doubles it. */
  hold: number;
  maxHold: number;
  relapse: number;
  /** Page busy share and late-frame share above which the page is overloaded, and below which it has room. */
  busyHigh: number;
  busyLow: number;
  jankHigh: number;
  jankLow: number;
  /** Tracking overload share above which tracking is overloaded, and below which it has room. */
  overloadHigh: number;
  overloadLow: number;
  devicePixelRatio: number;
  /** Starting level for both ladders. */
  startLevel: number;
}

export const DEFAULT_GOVERNOR: GovernorOptions = {
  window: 500, warmup: 2000, hold: 5000, maxHold: 60000, relapse: 20000,
  busyHigh: 0.5, busyLow: 0.3, jankHigh: 0.25, jankLow: 0.08, overloadHigh: 0.3, overloadLow: 0.1,
  devicePixelRatio: 1, startLevel: 0,
};

/** One quality ladder with asymmetric hysteresis: two bad windows step down, a calm hold steps up. */
export class Ladder {
  level: number;
  private bad = 0;
  private calmSince = NaN;
  private hold: number;
  private raisedAt = -Infinity;

  constructor(private readonly max: number, private readonly o: Pick<GovernorOptions, 'hold' | 'maxHold' | 'relapse'>, start = 0) {
    this.level = Math.min(max, Math.max(0, start));
    this.hold = o.hold;
  }

  /** One window at time `now` (ms). Returns whether the level changed. */
  step(now: number, pressure: boolean, calm: boolean): boolean {
    if (pressure) {
      this.calmSince = NaN;
      if (++this.bad < 2 || this.level >= this.max) return false;
      if (now - this.raisedAt < this.o.relapse) this.hold = Math.min(this.o.maxHold, 2 * this.hold);
      this.level++;
      this.bad = 0;
      return true;
    }
    this.bad = 0;
    if (!calm || this.level === 0) {
      this.calmSince = NaN;
      return false;
    }
    if (Number.isNaN(this.calmSince)) this.calmSince = now;
    if (now - this.calmSince < this.hold) return false;
    this.level--;
    this.raisedAt = this.calmSince = now;
    return true;
  }
}

/** Tracking-side measurements, from the latest results. */
export interface TrackingLoad {
  stride: number;
  overload: number;
  interval: number;
  acquireBusy: number;
}

export class Governor {
  readonly render: Ladder;
  readonly tracking: Ladder;
  private readonly o: GovernorOptions;
  private start = NaN;
  private windowStart = NaN;
  private mainMs = 0;
  private frames = 0;
  private lateFrames = 0;
  private prevFrame = NaN;
  private vsync = NaN;
  /** Recent display frame gaps (ms), for the vsync estimate. */
  private readonly gaps = new Float64Array(64);
  private gapN = 0;
  private load: TrackingLoad = { stride: 1, overload: 0, interval: 1000 / 30, acquireBusy: 0 };
  private bottleneck: Bottleneck = 'none';
  private mainBusy = 0;
  private jank = 0;
  display = { delay: 0, late: 0 };

  constructor(o: Partial<GovernorOptions> = {}) {
    this.o = { ...DEFAULT_GOVERNOR, ...o };
    const max = LEVELS.length - 1;
    this.render = new Ladder(max, this.o, this.o.startLevel);
    this.tracking = new Ladder(max, this.o, this.o.startLevel);
  }

  /** Time spent on the page handling a camera or display frame (ms). */
  addMain(ms: number): void {
    if (ms > 0) this.mainMs += ms;
  }

  /** A display frame (requestAnimationFrame) at time t (ms). */
  animationFrame(t: number): void {
    const d = t - this.prevFrame;
    this.prevFrame = t;
    if (!(d > 0 && d < 250)) return;
    this.gaps[this.gapN++ % this.gaps.length] = d;
    // A late frame is often followed by a short gap (the next one on time): count only the late one.
    if (!(this.vsync > 0)) return;
    this.frames++;
    if (d > 1.5 * this.vsync) this.lateFrames++;
  }

  /**
   * The display interval: a low quantile of the gaps between display
   * frames (robust to the short gap after a late frame, while most frames
   * may be late), the shortest seen (a loaded page has longer gaps, not a
   * slower display), rising only slowly (the browser capping it at 30 Hz,
   * as iOS Low Power Mode does, is the display's rate, not lateness).
   */
  private updateVsync(): void {
    const n = Math.min(this.gapN, this.gaps.length);
    if (n < 16) return;
    const s = this.gaps.slice(0, n).sort();
    const m = s[n >> 2];
    this.vsync = !(this.vsync > 0) || m < this.vsync ? m : this.vsync + 0.1 * (m - this.vsync);
  }

  setTrackingLoad(l: TrackingLoad): void {
    this.load = l;
  }

  get acquireDuty(): number {
    return LEVELS[Math.max(this.render.level, this.tracking.level)].acquireDuty;
  }

  /** Close a window if one has passed; returns whether a level changed. */
  update(now: number): boolean {
    if (Number.isNaN(this.start)) this.start = this.windowStart = now;
    const span = now - this.windowStart;
    if (span < this.o.window) return false;
    const o = this.o;
    this.mainBusy = this.mainMs / span;
    this.jank = this.frames ? this.lateFrames / this.frames : 0;
    this.updateVsync();
    const gap = span > 4 * o.window; // hidden page, or a long stall: not a measurement
    this.windowStart = now;
    this.mainMs = this.frames = this.lateFrames = 0;
    if (gap || now - this.start < o.warmup) return false;
    const L = this.load;
    const renderBad = this.mainBusy > o.busyHigh || this.jank > o.jankHigh;
    const renderOk = this.mainBusy < o.busyLow && this.jank < o.jankLow;
    const trackBad = L.stride > 1 || L.overload > o.overloadHigh;
    const trackOk = L.stride === 1 && L.overload < o.overloadLow;
    if (renderBad || trackBad) this.bottleneck = renderBad && (!trackBad || this.render.level <= this.tracking.level) ? 'render' : 'tracking';
    else if (this.render.level === 0 && this.tracking.level === 0) this.bottleneck = 'none';
    const a = this.render.step(now, renderBad, renderOk);
    const b = this.tracking.step(now, trackBad, trackOk);
    return a || b;
  }

  get state(): PerformanceState {
    const r = LEVELS[this.render.level];
    return {
      level: Math.max(this.render.level, this.tracking.level),
      bottleneck: this.bottleneck,
      pixelRatio: Math.min(this.o.devicePixelRatio, r.pixelRatio),
      effects: r.effects,
      frameInterval: this.load.interval,
      stride: this.load.stride,
      mainBusy: this.mainBusy,
      jank: this.jank,
      trackingOverload: this.load.overload,
      acquireBusy: this.load.acquireBusy,
      displayDelay: this.display.delay,
      displayLate: this.display.late,
    };
  }
}

/** Starting level from what the browser says about the device (only low-memory Android says much). */
export function deviceStartLevel(nav: { deviceMemory?: number } | undefined =
  typeof navigator !== 'undefined' ? (navigator as { deviceMemory?: number }) : undefined): number {
  const mem = nav?.deviceMemory;
  return mem !== undefined && mem <= 2 ? 1 : 0;
}
