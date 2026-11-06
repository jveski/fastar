// Display pacing for syncDisplay: every camera frame is shown, in order, a
// steady delay after it was captured, like a jitter buffer for audio.
// Showing each frame as soon as it was tracked instead ties the picture to
// the tracking rate: on a slow device the background itself then stutters.
//
// The delay follows how long tracking results take (most of them arrive in
// time), never more than maxDelay past the frame's arrival; a frame whose
// result is not in by then is shown anyway, with the filter's prediction.
// With nothing in view there is nothing to wait for, and frames are shown
// as they arrive. The delay changes a few ms per frame, never in jumps.

export interface PacedFrame<T> {
  /** Capture time (ms). */
  captureTime: number;
  /** When the frame reached the page (ms, same clock). */
  arrival: number;
  payload: T;
}

export interface PacerOptions {
  /** Most delay (ms) past a frame's arrival. */
  maxDelay: number;
  /** Delay changes per shown frame (ms): up, and down. */
  slewUp: number;
  slewDown: number;
  /** Share of results that should arrive in time. */
  quantile: number;
}

export const DEFAULT_PACER: PacerOptions = { maxDelay: 80, slewUp: 4, slewDown: 3, quantile: 0.9 };

const SAMPLES = 32;

export class DisplayPacer<T> {
  /** Current delay from capture to display (ms). */
  delay = NaN;
  /** Frames are waiting on results (a target is shown). */
  active = false;
  /** Frames held at most; the oldest is dropped beyond that. */
  capacity = 5;
  /** Frames shown, shown before their result arrived, and dropped. */
  shown = 0;
  late = 0;
  dropped = 0;
  private readonly o: PacerOptions;
  private readonly q: Array<PacedFrame<T>> = [];
  private lastResult = -Infinity;
  private readonly lat = new Float64Array(SAMPLES);
  private latN = 0;
  private readonly arr = new Float64Array(SAMPLES);
  private arrN = 0;
  private readonly scratch = new Float64Array(SAMPLES);

  constructor(o: Partial<PacerOptions> = {}) {
    this.o = { ...DEFAULT_PACER, ...o };
  }

  get maxDelay(): number {
    return this.o.maxDelay;
  }

  get length(): number {
    return this.q.length;
  }

  /** Delay past arrival (ms): what pacing adds to the display latency. */
  get extraDelay(): number {
    const a = this.arrivalDelay();
    return Number.isNaN(this.delay) ? 0 : Math.max(0, this.delay - a);
  }

  push(f: PacedFrame<T>): void {
    this.arr[this.arrN++ % SAMPLES] = f.arrival - f.captureTime;
    if (Number.isNaN(this.delay)) this.delay = f.arrival - f.captureTime;
    this.q.push(f);
    while (this.q.length > this.capacity) {
      this.q.shift();
      this.dropped++;
    }
  }

  /** A tracking result for the frame captured at `captureTime` arrived at `now`. */
  result(captureTime: number, now: number): void {
    if (captureTime > this.lastResult) this.lastResult = captureTime;
    this.lat[this.latN++ % SAMPLES] = now - captureTime;
  }

  /** The frame to show at `now`, if one is due (older due frames are skipped). */
  next(now: number): PacedFrame<T> | null {
    const q = this.q;
    let pick = -1;
    for (let i = 0; i < q.length; i++) {
      if (q[i].arrival <= now && q[i].captureTime + this.delay <= now + 0.5) pick = i;
      else break;
    }
    if (pick < 0) return null;
    const f = q[pick];
    this.dropped += pick;
    q.splice(0, pick + 1);
    this.shown++;
    if (this.active && f.captureTime > this.lastResult + 4) this.late++;
    const target = this.target(), d = this.delay;
    const down = this.active ? this.o.slewDown : 2 * this.o.slewDown;
    this.delay = target > d ? Math.min(target, d + this.o.slewUp) : Math.max(target, d - down);
    return f;
  }

  clear(): void {
    this.q.length = 0;
    this.lastResult = -Infinity;
  }

  /** Delay the frames should settle at (ms after capture). */
  private target(): number {
    const a = this.arrivalDelay();
    if (!this.active || this.latN === 0) return a;
    const l = quantile(this.lat, Math.min(this.latN, SAMPLES), this.o.quantile, this.scratch) + 2;
    return Math.min(a + this.o.maxDelay, Math.max(a, l));
  }

  /** Typical capture-to-arrival time (ms). */
  private arrivalDelay(): number {
    return this.arrN === 0 ? 0 : quantile(this.arr, Math.min(this.arrN, SAMPLES), 0.5, this.scratch);
  }
}

function quantile(a: Float64Array, n: number, q: number, scratch: Float64Array): number {
  const s = scratch.subarray(0, n);
  s.set(a.subarray(0, n));
  s.sort();
  return s[Math.min(n - 1, Math.floor(q * n))];
}
