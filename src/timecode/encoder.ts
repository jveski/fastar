// Timecode generator: renders the sweeps for a set time onwards, at any
// sample rate, optionally at a level that follows the music it is mixed into.
import { PositionCode, timecodeParams, type TimecodeOptions, type TimecodeParams } from './code';

export interface EncoderOptions extends TimecodeOptions {
  sampleRate: number;
  /** Set time (s) of the first sample (default 0). */
  start?: number;
  /** Peak level (dBFS) when render() is given no gain (default -12). */
  level?: number;
  /** Fade in over this long from the first sample (s, default 0.05). */
  fadeIn?: number;
}

export class TimecodeEncoder {
  readonly params: TimecodeParams;
  readonly sampleRate: number;
  readonly code: PositionCode;
  private readonly amp: number;
  private readonly start: number;
  private readonly fadeN: number;
  private n = 0;
  private phase = 0;

  constructor(o: EncoderOptions) {
    const p = (this.params = timecodeParams(o));
    this.sampleRate = o.sampleRate;
    if (!(p.f1 < 0.48 * o.sampleRate)) throw new Error(`fastar: a ${p.f1} Hz timecode needs a sample rate above ${Math.ceil(p.f1 / 0.48)} Hz`);
    this.code = new PositionCode(p.degree, p.poly);
    this.amp = 10 ** ((o.level ?? -12) / 20);
    this.start = o.start ?? 0;
    if (!(this.start >= 0)) throw new Error('fastar: the timecode start must be 0 or later');
    this.fadeN = Math.round((o.fadeIn ?? 0.05) * o.sampleRate);
  }

  /** Set time (s) of the next sample. */
  get time(): number {
    return this.start + this.n / this.sampleRate;
  }

  /**
   * Write the next out.length samples of timecode into out, scaled by gain
   * (peak amplitude: a number, or one per sample; default the level option).
   */
  render(out: Float32Array, gain?: number | Float32Array): void {
    const p = this.params, fs = this.sampleRate, T = p.symbol, tau = p.transition, lin = T - tau;
    const code = this.code, w = (2 * Math.PI) / fs;
    const g = gain ?? this.amp;
    let idx = -1, fPrev = 0, fa = 0, fb = 0;
    for (let i = 0; i < out.length; i++) {
      const t = this.start + this.n / fs;
      const k = Math.floor(t / T);
      if (k !== idx) {
        idx = k;
        const b = code.bit(k);
        fa = b ? p.f0 : p.f1;
        fb = b ? p.f1 : p.f0;
        fPrev = code.bit(k - 1) ? p.f1 : p.f0;
      }
      const u = t - k * T;
      // Each sweep is linear over the symbol's end; its start glides from where the last one ended.
      const f = u < tau ? fPrev + ((fa - fPrev) * (1 - Math.cos((Math.PI * u) / tau))) / 2 : fa + ((fb - fa) * (u - tau)) / lin;
      let a = typeof g === 'number' ? g : g[i];
      if (this.n < this.fadeN) a *= (1 - Math.cos((Math.PI * this.n) / this.fadeN)) / 2;
      out[i] = a * Math.sin(this.phase);
      this.phase += w * f;
      if (this.phase > 2 * Math.PI) this.phase -= 2 * Math.PI;
      this.n++;
    }
  }
}

export interface FollowOptions {
  /** Timecode level relative to the music's level above 4 kHz (dB, default -24). */
  relative?: number;
  /** Highest timecode level (dBFS peak, default -18). */
  max?: number;
  /** Music level above 4 kHz (dBFS) below which the timecode fades out (default -50). */
  gate?: number;
}

/**
 * Timecode level that follows the music: a fixed amount below its level
 * above 4 kHz (where most of what masks a sound near 20 kHz is), capped,
 * and faded out entirely in quiet passages where nothing would hide it.
 * The level moves smoothly (50 ms), so it adds no audible modulation.
 */
export class LevelFollower {
  private readonly rel: number;
  private readonly max: number;
  private readonly gate: number;
  private readonly b: Float64Array;
  private readonly a: Float64Array;
  /** High-pass state per channel: x1, x2, y1, y2. */
  private st: Float64Array[] = [];
  private env = 0;
  private gain = 0;
  private readonly kUp: number;
  private readonly kDown: number;
  private readonly kGain: number;

  constructor(readonly sampleRate: number, o: FollowOptions = {}) {
    this.rel = o.relative ?? -24;
    this.max = o.max ?? -18;
    this.gate = o.gate ?? -50;
    // Second-order Butterworth high-pass at 4 kHz.
    const w0 = (2 * Math.PI * 4000) / sampleRate, c = Math.cos(w0), al = Math.sin(w0) / Math.SQRT2;
    const a0 = 1 + al;
    this.b = Float64Array.of((1 + c) / 2 / a0, -(1 + c) / a0, (1 + c) / 2 / a0);
    this.a = Float64Array.of((-2 * c) / a0, (1 - al) / a0);
    this.kUp = 1 - Math.exp(-1 / (0.01 * sampleRate));
    // Falls about as fast as masking fades after a loud sound (~40 dB/s).
    this.kDown = 1 - Math.exp(-1 / (0.1 * sampleRate));
    this.kGain = 1 - Math.exp(-1 / (0.05 * sampleRate));
  }

  /** Music level above 4 kHz (dBFS: 0 for a full-scale sine). */
  get level(): number {
    return 10 * Math.log10(2 * this.env + 1e-20);
  }

  /**
   * Timecode gain (peak amplitude) for each sample of a block of music: one
   * channel, or several (their levels are averaged, so out-of-phase content
   * still counts).
   */
  process(music: Float32Array | Float32Array[], gains: Float32Array): void {
    const ch = Array.isArray(music) ? music : [music], { b, a } = this;
    while (this.st.length < ch.length) this.st.push(new Float64Array(4));
    for (let i = 0; i < gains.length; i++) {
      let pw = 0;
      for (let c = 0; c < ch.length; c++) {
        const s = this.st[c], x = ch[c][i];
        const y = b[0] * x + b[1] * s[0] + b[2] * s[1] - a[0] * s[2] - a[1] * s[3];
        s[1] = s[0]; s[0] = x; s[3] = s[2]; s[2] = y;
        pw += y * y;
      }
      pw /= ch.length;
      this.env += (pw > this.env ? this.kUp : this.kDown) * (pw - this.env);
      const db = 10 * Math.log10(2 * this.env + 1e-20);
      const open = Math.min(1, Math.max(0, (db - this.gate) / 6));
      const target = open * 10 ** (Math.min(this.max, db + this.rel) / 20);
      this.gain += this.kGain * (target - this.gain);
      gains[i] = this.gain;
    }
  }
}
