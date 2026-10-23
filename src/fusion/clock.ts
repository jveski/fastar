// Regular-clock reconstruction for periodic samples with jittery timestamps.
//
// iOS stamps devicemotion events with their dispatch time (and video frames
// with the display time), so samples taken every 16.7 ms arrive stamped 0 to
// 30 ms apart, sometimes two at once. Integrating the gyro on those stamps
// (or lining frames up with it) adds error proportional to the rotation
// rate. The samples themselves are periodic: sample k was taken at
// T0 + k P, and stamped a short, variable delay later. Over a sliding window
// the period is the slope of stamp against index, and the grid sits on the
// lower envelope of the stamps (the least-delayed ones).
export class RegularClock {
  private readonly ks: Float64Array;
  private readonly ss: Float64Array;
  private head = 0;
  private count = 0;
  private k = 0;
  private last = NaN;
  private lastRaw = NaN;
  private period: number;

  /**
   * period: nominal sample interval (same units as the stamps).
   * skipDelay: a stamp this many periods later than the grid predicts counts
   * as skipped samples; it must exceed the stamp delay jitter (video frames:
   * ~0.6, since display stamps jitter a few ms and frames do get dropped;
   * devicemotion: delays reach two periods, but events are not dropped).
   * window: samples used for the fit.
   */
  constructor(private readonly nominal: number, private readonly skipDelay = 0.6, private readonly window = 240) {
    this.period = nominal;
    this.ks = new Float64Array(window);
    this.ss = new Float64Array(window);
  }

  reset(): void {
    this.count = this.head = this.k = 0;
    this.last = this.lastRaw = NaN;
    this.period = this.nominal;
  }

  /** Regularized time for the next sample stamped `raw`. */
  map(raw: number): number {
    if (Number.isNaN(this.last) || raw - this.lastRaw > 20 * this.period || raw < this.lastRaw - 5 * this.period) {
      // Start, a long pause or a clock step: restart on this stamp.
      this.reset();
      this.push(0, raw);
      this.last = this.lastRaw = raw;
      return raw;
    }
    // Next index; missed samples show up as a stamp far beyond the grid.
    let k = this.k + 1;
    const pred = this.last + this.period;
    if (raw - pred > this.skipDelay * this.period) k += Math.max(1, Math.round((raw - pred) / this.period));
    this.k = k;
    this.push(k, raw);
    this.lastRaw = raw;
    // Period: least-squares slope of stamp against index (once the window spans enough).
    const n = this.count;
    if (n >= 30) {
      let mk = 0, ms = 0;
      for (let i = 0; i < n; i++) { mk += this.ks[i]; ms += this.ss[i]; }
      mk /= n; ms /= n;
      let skk = 0, sks = 0;
      for (let i = 0; i < n; i++) {
        const dk = this.ks[i] - mk;
        skk += dk * dk; sks += dk * (this.ss[i] - ms);
      }
      const P = sks / skk;
      if (P > 0.8 * this.nominal && P < 1.25 * this.nominal) this.period = P;
    }
    // Offset: lower envelope.
    let off = Infinity;
    for (let i = 0; i < n; i++) off = Math.min(off, this.ss[i] - this.period * this.ks[i]);
    let t = off + this.period * k;
    if (!(t > this.last)) t = this.last + 0.25 * this.period;
    this.last = t;
    return t;
  }

  private push(k: number, s: number): void {
    const i = (this.head + this.count) % this.window;
    if (this.count < this.window) this.count++;
    else this.head = (this.head + 1) % this.window;
    this.ks[i] = k;
    this.ss[i] = s;
  }
}
