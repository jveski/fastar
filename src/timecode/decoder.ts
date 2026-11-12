// Timecode listener's signal processing: finds the sweeps in microphone
// audio, reads their bits, works out where in the code they are, and fits
// set time against the audio sample clock.
//
//   audio -> mix the band down to baseband, decimate to ~2.4 kHz
//         -> correlate with an up and a down sweep (each peaks where a sweep ends)
//         -> symbol timing: correlation peaks folded over one symbol period
//         -> per symbol: a soft bit (up vs down) and the time its sweep ended
//         -> position: any `degree` bits locate the code; later bits confirm it
//         -> set time = fit of (sweep end, symbol end in set time) pairs
//
// Times are seconds on the audio clock: sample frame / sample rate, from
// whatever origin the frames passed to push() count from.
import { PositionCode, timecodeParams, type TimecodeOptions, type TimecodeParams } from './code';
import { median } from '../math/select';

export interface DecoderOptions extends TimecodeOptions {
  sampleRate: number;
  /** Reported time is locked once its standard deviation is below this (s, default 0.01). */
  lockStd?: number;
  /** Stay locked this long without hearing the timecode (s, default 600). */
  maxCoast?: number;
}

/** Set time as a function of audio time: time + rate * (t - at). */
export interface TimecodeFit {
  /** The position in the code is known and recent symbols agree with it. */
  codeLocked: boolean;
  /** Points in the fit (0: no estimate yet). */
  points: number;
  /** Reference audio time (s). */
  at: number;
  /** Set time at `at` (s). */
  time: number;
  /** Set seconds per audio second. */
  rate: number;
  /** Variance of `time` (s^2) and of `rate`. */
  varTime: number;
  varRate: number;
  /** Audio time of the latest point (s). */
  last: number;
}

export interface DecoderStats {
  /** Symbols read so far. */
  symbols: number;
  /** Strength of recent clearly heard sweeps over the noise (dB, after correlation). */
  snr: number;
  /** Share of recent symbols heard clearly. */
  clear: number;
  /** Share of recent clear symbols that disagreed with the code (while locked). */
  errors: number;
  codeLocked: boolean;
  /** Audio time (s) of the first lock, NaN before. */
  lockedAt: number;
  /**
   * Times the sweeps' arrival jumped while following the same code: either
   * the microphone path's delay changed (recalibrate) or the phone moved far.
   */
  jumps: number;
}

export interface TimecodeEstimate {
  /** Set time (s). */
  time: number;
  /** Standard deviation of `time` (s). */
  std: number;
  locked: boolean;
}

export function fitTime(f: TimecodeFit, t: number): number {
  return f.points ? f.time + f.rate * (t - f.at) : NaN;
}

export function fitStd(f: TimecodeFit, t: number): number {
  return f.points ? Math.sqrt(f.varTime + (t - f.at) ** 2 * f.varRate) : Infinity;
}

/** Estimate at audio time t from a fit. */
export function fitEstimate(f: TimecodeFit, t: number, lockStd = 0.01, maxCoast = 600): TimecodeEstimate {
  const std = fitStd(f, t);
  return { time: fitTime(f, t), std, locked: f.codeLocked && f.points >= 5 && std < lockStd && t - f.last < maxCoast };
}

/** Complex baseband around fc at about `rate` Hz, by mixing and a windowed-sinc low-pass. */
export class Downconverter {
  readonly D: number;
  readonly rate: number;
  /** Input samples between an input sample and the output it is centred in. */
  readonly delay: number;
  private readonly h: Float32Array;
  private readonly re: Float32Array;
  private readonly im: Float32Array;
  private pos = 0;
  private count = 0;
  private cr = 1;
  private ci = 0;
  private readonly wr: number;
  private readonly wi: number;

  /** pass: half-width of the band to keep (Hz). */
  constructor(readonly sampleRate: number, readonly fc: number, pass: number) {
    const D = (this.D = Math.max(1, Math.floor(sampleRate / (2 * pass + 1400))));
    const fd = (this.rate = sampleRate / D);
    // Pass |f| < pass; anything that would alias into it (|f| > fd - pass) is stopped (Blackman: ~74 dB).
    const stop = fd - pass;
    const L = (Math.ceil((5.5 * sampleRate) / (stop - pass)) >> 1) * 2 + 1;
    const h = (this.h = new Float32Array(L));
    const fcut = (pass + stop) / 2 / sampleRate;
    let s = 0;
    for (let i = 0; i < L; i++) {
      const x = i - (L - 1) / 2;
      const sinc = x === 0 ? 2 * fcut : Math.sin(2 * Math.PI * fcut * x) / (Math.PI * x);
      const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (L - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (L - 1));
      s += h[i] = sinc * w;
    }
    for (let i = 0; i < L; i++) h[i] /= s;
    this.delay = (L - 1) / 2;
    this.re = new Float32Array(2 * L);
    this.im = new Float32Array(2 * L);
    this.wr = Math.cos((2 * Math.PI * fc) / sampleRate);
    this.wi = Math.sin((2 * Math.PI * fc) / sampleRate);
  }

  get taps(): number {
    return this.h.length;
  }

  reset(): void {
    this.re.fill(0);
    this.im.fill(0);
    this.pos = this.count = 0;
  }

  /** Mix and decimate x; returns the number of outputs written (out arrays need ceil(x.length / D) + 1). */
  process(x: Float32Array, outRe: Float32Array, outIm: Float32Array): number {
    const { h, re, im, D } = this, L = h.length;
    let n = 0, cr = this.cr, ci = this.ci, pos = this.pos, count = this.count;
    const wr = this.wr, wi = this.wi;
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      re[pos] = re[pos + L] = v * cr;
      im[pos] = im[pos + L] = -v * ci;
      pos = pos + 1 === L ? 0 : pos + 1;
      const t = cr * wr - ci * wi;
      ci = ci * wr + cr * wi;
      cr = t;
      if (++count === D) {
        count = 0;
        let sr = 0, si = 0;
        for (let k = 0; k < L; k++) {
          sr += h[k] * re[pos + k];
          si += h[k] * im[pos + k];
        }
        outRe[n] = sr;
        outIm[n++] = si;
      }
    }
    // Keep the oscillator on the unit circle.
    const g = 1 / Math.hypot(cr, ci);
    this.cr = cr * g;
    this.ci = ci * g;
    this.pos = pos;
    this.count = count;
    return n;
  }
}

// Decoding constants.
/** Half-width of the search for a sweep's end around the expected symbol boundary (s): Doppler (~12 ms) plus echoes. */
const PEAK_WINDOW = 0.03;
/** A symbol period whose strongest correlation is this far over the noise (power ratio) informs the timing. */
const TIMING_SNR = 12;
/** Memory of the symbol timing (s of clearly heard symbols). */
const TIMING_MEMORY = 3;
/** Soft bit: tanh(difference of the up and down peaks / (this * noise)). */
const SOFT_SCALE = 6;
/** Soft bits at least this large count as clearly heard. */
const CLEAR = 0.5;
/**
 * Symbols this far over the noise (power ratio) give clock points, and count
 * against the code when they disagree with it; noise alone rarely gets here,
 * so quiet passages do not skew the clock.
 */
const STRONG = 10;
/** A sweep's end is the earliest correlation peak at least this share of the strongest (later ones are echoes)... */
const FIRST_PEAK = 0.5;
/** ...and this far over the noise (power ratio), so that noise does not pass for an early arrival. */
const FIRST_PEAK_SNR = 9;
/** Symbols searched for the code position. */
const SEARCH = 64;
/** Clear symbols beyond the sequence degree needed for a candidate position (at most one may disagree). */
const CANDIDATE_MARGIN = 20;
/** Clear symbols, after a candidate is found, that must confirm it (at most one may disagree). */
const CONFIRM = 16;
/** While locked, look for the timecode elsewhere once this many of the last RECENT strong symbols disagree with the code. */
const RECENT = 24;
const DOUBT_ERRORS = 6;
/** Clock fit window (s) and largest number of points. */
const FIT_WINDOW = 120;
const FIT_POINTS = 1200;
/** Clock fit bins (s). */
const FIT_BIN = 2;
/** Largest difference in clock rate believed: two consumer audio clocks, each within ~100 ppm. */
const MAX_RATE = 3e-4;
/** Samples fed as silence across a gap in the input, at most (s); longer gaps restart the signal path. */
const MAX_FILL = 2;

const HIST = 256;

export class TimecodeDecoder {
  readonly params: TimecodeParams;
  readonly sampleRate: number;
  readonly code: PositionCode;
  readonly lockStd: number;
  readonly maxCoast: number;
  /** Called after each symbol is read. */
  onSymbol: (() => void) | null = null;

  private readonly dc: Downconverter;
  private readonly tick: number;
  // Correlators: one sweep's linear part, ending at the newest sample.
  private readonly ns: number;
  private readonly tRe: Float32Array;
  private readonly tIm: Float32Array;
  private readonly zRe: Float32Array;
  private readonly zIm: Float32Array;
  private zPos = 0;
  private zFill = 0;
  // Correlation power, by decimated sample.
  private readonly ring: number;
  private readonly pU: Float32Array;
  private readonly pD: Float32Array;
  private k = 0;
  /** Audio time of decimated sample 0. */
  private tStart = NaN;
  /** First decimated sample with a full correlation since the last restart. */
  private kValid = 0;
  private nextFrame = NaN;
  private bufRe = new Float32Array(0);
  private bufIm = new Float32Array(0);
  // Noise and symbol timing.
  private noise = NaN;
  private readonly nb: number;
  private readonly fold: Float64Array;
  private readonly cur: Float64Array;
  private readonly noiseBuf: Float64Array;
  private noiseN = 0;
  private period = NaN;
  private readonly foldDecay: number;
  /** Time within the symbol at which sweeps end (s), NaN until known. */
  private phase = NaN;
  private lastB = NaN;
  private j = 0;
  // Symbols.
  private readonly hj = new Float64Array(HIST);
  private readonly hs = new Float64Array(HIST);
  private readonly ht = new Float64Array(HIST);
  private readonly hp = new Float64Array(HIST);
  private hHead = 0;
  private hCount = 0;
  // Code position: sequence index of receiver symbol j is offset + j.
  private codeLocked = false;
  private offset = NaN;
  private pending: { offset: number; agree: number; errors: number } | null = null;
  /** The last symbol given to the clock, waiting for a neighbour sweeping the other way. */
  private pair: { j: number; t: number; up: boolean } | null = null;
  private readonly recent = new Uint8Array(RECENT);
  private recentN = 0;
  private recentHead = 0;
  private readonly clock = new SetClock();
  private readonly st: DecoderStats = { symbols: 0, snr: 0, clear: 0, errors: 0, codeLocked: false, lockedAt: NaN, jumps: 0 };

  constructor(o: DecoderOptions) {
    const p = (this.params = timecodeParams(o));
    this.sampleRate = o.sampleRate;
    if (!(p.f1 < 0.49 * o.sampleRate)) throw new Error(`fastar: a ${o.sampleRate} Hz input cannot carry a ${p.f1} Hz timecode`);
    this.lockStd = o.lockStd ?? 0.01;
    this.maxCoast = o.maxCoast ?? 600;
    this.code = new PositionCode(p.degree, p.poly);
    const B = p.f1 - p.f0, fc = (p.f0 + p.f1) / 2;
    // Room for the band and for Doppler from a phone waved about (2 m/s: ~120 Hz).
    this.dc = new Downconverter(o.sampleRate, fc, B / 2 + 150);
    const fd = this.dc.rate;
    this.tick = this.dc.D / o.sampleRate;
    const lin = p.symbol - p.transition;
    const ns = (this.ns = Math.floor(lin * fd));
    this.tRe = new Float32Array(ns);
    this.tIm = new Float32Array(ns);
    const g = 1 / Math.sqrt(ns);
    for (let i = 0; i < ns; i++) {
      const tau = (i - (ns - 1)) / fd;
      const ph = 2 * Math.PI * ((B / 2) * tau + 0.5 * (B / lin) * tau * tau);
      this.tRe[i] = g * Math.cos(ph);
      this.tIm[i] = g * Math.sin(ph);
    }
    this.zRe = new Float32Array(2 * ns);
    this.zIm = new Float32Array(2 * ns);
    this.nb = Math.max(8, Math.floor(p.symbol * fd));
    let r = 1;
    while (r < 4 * this.nb + 4 * PEAK_WINDOW * fd) r *= 2;
    this.ring = r;
    this.pU = new Float32Array(r);
    this.pD = new Float32Array(r);
    this.fold = new Float64Array(this.nb);
    this.cur = new Float64Array(this.nb);
    this.noiseBuf = new Float64Array(4 * this.nb + 8);
    this.foldDecay = Math.exp(-p.symbol / TIMING_MEMORY);
    // Build the position lookup now rather than on the first lock attempt.
    this.code.indexOf(1);
  }

  /** Audio time (s) of the newest sample pushed. */
  get time(): number {
    return Number.isNaN(this.nextFrame) ? NaN : (this.nextFrame - 1) / this.sampleRate;
  }

  get stats(): DecoderStats {
    return { ...this.st, codeLocked: this.codeLocked, jumps: this.clock.jumps };
  }

  get fit(): TimecodeFit {
    return { ...this.clock.fit, codeLocked: this.codeLocked };
  }

  /** Set time at audio time t (s). */
  estimate(t: number): TimecodeEstimate {
    return fitEstimate(this.fit, t, this.lockStd, this.maxCoast);
  }

  /** Forget everything (a new audio clock). */
  reset(): void {
    this.restart();
    this.nextFrame = NaN;
    this.noise = this.phase = this.lastB = NaN;
    this.fold.fill(0);
    this.j = 0;
    this.hCount = this.hHead = 0;
    this.unlock();
    this.offset = NaN;
    this.clock.reset();
  }

  /**
   * Feed audio. `frame`: the sample index of samples[0] on the audio clock
   * (default: straight after the last call). Short gaps are filled with
   * silence; time going backwards resets the decoder.
   */
  push(samples: Float32Array, frame = Number.isNaN(this.nextFrame) ? 0 : this.nextFrame): void {
    if (!Number.isNaN(this.nextFrame) && frame !== this.nextFrame) {
      const gap = frame - this.nextFrame;
      if (gap < 0) this.reset();
      else if (gap <= MAX_FILL * this.sampleRate) {
        const z = new Float32Array(Math.min(gap, 8192));
        for (let left = gap; left > 0; left -= z.length) this.process(left >= z.length ? z : z.subarray(0, left));
      } else {
        this.restart();
      }
    }
    if (Number.isNaN(this.tStart)) {
      // First output: once D samples are in, centred `delay` samples back.
      this.tStart = (frame + this.dc.D - 1 - this.dc.delay) / this.sampleRate;
      this.k = 0;
      this.kValid = this.ns - 1 + Math.ceil(this.dc.taps / this.dc.D);
    }
    this.process(samples);
    this.nextFrame = frame + samples.length;
  }

  /** Restart the signal path (after a long gap), keeping timing, position and clock. */
  private restart(): void {
    this.dc.reset();
    this.zRe.fill(0);
    this.zIm.fill(0);
    this.zPos = this.zFill = 0;
    this.tStart = NaN;
    this.period = NaN;
    this.cur.fill(0);
    this.noiseN = 0;
  }

  private process(x: Float32Array): void {
    const need = Math.ceil(x.length / this.dc.D) + 1;
    if (this.bufRe.length < need) {
      this.bufRe = new Float32Array(need);
      this.bufIm = new Float32Array(need);
    }
    const n = this.dc.process(x, this.bufRe, this.bufIm);
    for (let i = 0; i < n; i++) this.sample(this.bufRe[i], this.bufIm[i]);
  }

  /** One baseband sample. */
  private sample(re: number, im: number): void {
    const ns = this.ns, zr = this.zRe, zi = this.zIm;
    zr[this.zPos] = zr[this.zPos + ns] = re;
    zi[this.zPos] = zi[this.zPos + ns] = im;
    this.zPos = this.zPos + 1 === ns ? 0 : this.zPos + 1;
    if (this.zFill < ns) this.zFill++;
    let pu = 0, pd = 0;
    if (this.zFill === ns) {
      // z * conj(up) and z * conj(down) = z * up, from the same four sums.
      const tr = this.tRe, ti = this.tIm, o = this.zPos;
      let ac = 0, bd = 0, bc = 0, ad = 0;
      for (let i = 0; i < ns; i++) {
        const a = zr[o + i], b = zi[o + i], c = tr[i], d = ti[i];
        ac += a * c; bd += b * d; bc += b * c; ad += a * d;
      }
      pu = (ac + bd) ** 2 + (bc - ad) ** 2;
      pd = (ac - bd) ** 2 + (bc + ad) ** 2;
    }
    const k = this.k++, slot = k & (this.ring - 1);
    this.pU[slot] = pu;
    this.pD[slot] = pd;
    const t = this.tStart + k * this.tick;
    if (k < this.kValid) return;

    // Symbol periods: noise level, and the timing fold.
    const T = this.params.symbol, per = Math.floor(t / T);
    if (per !== this.period) {
      if (!Number.isNaN(this.period)) this.endPeriod();
      this.period = per;
    }
    const bin = Math.min(this.nb - 1, Math.floor((t / T - per) * this.nb));
    const m = pu > pd ? pu : pd;
    if (m > this.cur[bin]) this.cur[bin] = m;
    if (this.noiseN + 1 < this.noiseBuf.length) {
      this.noiseBuf[this.noiseN++] = pu;
      this.noiseBuf[this.noiseN++] = pd;
    }

    // Symbols whose search window has been fully heard.
    if (Number.isNaN(this.phase)) return;
    if (Number.isNaN(this.lastB)) this.lastB = t - T;
    for (;;) {
      const b = this.nextBoundary();
      if (b + PEAK_WINDOW > t) break;
      this.j += Math.max(1, Math.round((b - this.lastB) / T));
      this.lastB = b;
      this.symbolAt(b, k);
    }
  }

  /** The first boundary (a sweep end) more than half a symbol after the last one. */
  private nextBoundary(): number {
    const T = this.params.symbol, from = this.lastB + T / 2;
    let b = Math.floor((from - this.phase) / T) * T + this.phase;
    while (b <= from) b += T;
    return b;
  }

  private endPeriod(): void {
    const n = this.noiseN;
    this.noiseN = 0;
    if (n >= this.nb) {
      // Correlation power is exponential in noise: median = variance * ln 2.
      const s2 = median(this.noiseBuf, n) / Math.LN2;
      if (s2 > 0) this.noise = Number.isNaN(this.noise) ? s2 : this.noise + 0.2 * (s2 - this.noise);
    }
    const cur = this.cur, f = this.fold, nb = this.nb, noise = this.noise;
    let mx = 0;
    for (let i = 0; i < nb; i++) if (cur[i] > mx) mx = cur[i];
    if (noise > 0 && mx / noise > TIMING_SNR) {
      for (let i = 0; i < nb; i++) f[i] = this.foldDecay * f[i] + cur[i] / noise;
      // Peak of the fold (smoothed over neighbouring bins), to a fraction of a bin.
      const s = (i: number) => f[(i + nb - 1) % nb] + 2 * f[i] + f[(i + 1) % nb];
      let im = 0, sm = -1;
      for (let i = 0; i < nb; i++) {
        const v = s(i);
        if (v > sm) { sm = v; im = i; }
      }
      const y0 = s((im + nb - 1) % nb), y2 = s((im + 1) % nb), den = y0 - 2 * sm + y2;
      const d = den < 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (y0 - y2)) / den)) : 0;
      const T = this.params.symbol;
      this.phase = ((((im + d + 0.5) / nb) % 1) + 1) % 1 * T;
    }
    cur.fill(0);
  }

  /** Read the symbol whose sweep ends near audio time b; k is the newest decimated sample. */
  private symbolAt(b: number, k: number): void {
    const noise = this.noise;
    if (!(noise > 0)) return;
    const kc = (b - this.tStart) / this.tick, w = PEAK_WINDOW / this.tick;
    const k0 = Math.max(Math.ceil(kc - w), this.kValid, k - this.ring + 2), k1 = Math.min(Math.floor(kc + w), k - 1);
    if (k1 - k0 < 2) return;
    const R = this.ring - 1, pU = this.pU, pD = this.pD;
    let ku = k0, kd = k0;
    for (let i = k0 + 1; i <= k1; i++) {
      if (pU[i & R] > pU[ku & R]) ku = i;
      if (pD[i & R] > pD[kd & R]) kd = i;
    }
    const mu = pU[ku & R], md = pD[kd & R];
    const s = Math.tanh((mu - md) / (SOFT_SCALE * noise));
    // Where the stronger sweep ended, to a fraction of a sample: its first
    // strong peak (the direct sound; an echo or a further speaker comes later).
    const up = mu >= md, p = up ? pU : pD, top = up ? mu : md, floor = Math.max(FIRST_PEAK * top, FIRST_PEAK_SNR * noise);
    let kk = up ? ku : kd;
    for (let i = k0 + 1; i < kk; i++) {
      const v = p[i & R];
      if (v >= floor && v >= p[(i - 1) & R] && v >= p[(i + 1) & R]) { kk = i; break; }
    }
    let d = 0;
    if (kk > k0 && kk < k1) {
      const y0 = p[(kk - 1) & R], y1 = p[kk & R], y2 = p[(kk + 1) & R], den = y0 - 2 * y1 + y2;
      if (den < 0) d = Math.max(-0.5, Math.min(0.5, (0.5 * (y0 - y2)) / den));
    }
    this.symbol(this.j, s, this.tStart + (kk + d) * this.tick, Math.max(mu, md) / noise);
  }

  private symbol(j: number, s: number, t: number, snr: number): void {
    const st = this.st;
    st.symbols++;
    const clear = Math.abs(s) >= CLEAR;
    st.clear += 0.05 * ((clear ? 1 : 0) - st.clear);
    if (clear) st.snr += (st.symbols < 5 ? 0.5 : 0.05) * (10 * Math.log10(snr) - st.snr);
    const h = this.hHead;
    this.hj[h] = j; this.hs[h] = s; this.ht[h] = t; this.hp[h] = snr;
    this.hHead = (h + 1) % HIST;
    if (this.hCount < HIST) this.hCount++;

    if (this.codeLocked && clear && snr >= STRONG) {
      const ok = s > 0 === (this.code.bit(this.offset + j) === 1);
      this.recent[this.recentHead] = ok ? 0 : 1;
      this.recentHead = (this.recentHead + 1) % RECENT;
      if (this.recentN < RECENT) this.recentN++;
      if (ok) this.point(j, t, s > 0);
    }
    let errs = 0;
    for (let i = 0; i < this.recentN; i++) errs += this.recent[i];
    st.errors = this.recentN ? errs / this.recentN : 0;
    if (this.pending) {
      if (clear) {
        const ok = s > 0 === (this.code.bit(this.pending.offset + j) === 1);
        if (ok) this.pending.agree++;
        else this.pending.errors++;
        if (this.pending.errors > 1) this.pending = null;
        else if (this.pending.agree >= CONFIRM) this.lock(this.pending.offset, t);
      }
    } else if (clear && (!this.codeLocked || errs >= DOUBT_ERRORS)) {
      // Not locked, or the code no longer fits: a timecode that moved (restarted
      // at another time) shows up as another position that does. Until one is
      // found and confirmed, keep the lock: sounds in a breakdown also disagree.
      this.search();
    }
    this.onSymbol?.();
  }

  private unlock(): void {
    this.codeLocked = false;
    this.pending = null;
    this.pair = null;
    this.recentN = this.recentHead = 0;
    this.st.errors = 0;
  }

  /**
   * A clock point from a symbol that agrees with the code. A moving phone
   * hears a Doppler shift, which moves an up sweep's end one way and a down
   * sweep's the other (2 m/s: ~12 ms), so points come from neighbouring up
   * and down sweeps, averaged.
   */
  private point(j: number, t: number, up: boolean): void {
    const p = this.pair;
    if (p && p.j === j - 1 && p.up !== up) this.clock.add((p.t + t) / 2, (this.offset + j + 0.5) * this.params.symbol);
    this.pair = { j, t, up };
  }

  private lock(offset: number, t: number): void {
    this.pending = null;
    this.codeLocked = true;
    this.recentN = this.recentHead = 0;
    if (offset !== this.offset) this.clock.reset();
    this.offset = offset;
    this.pair = null;
    if (Number.isNaN(this.st.lockedAt)) this.st.lockedAt = t;
    // Every symbol heard so far that agrees with the code gives clock points.
    const n = this.hCount;
    for (let i = n; i >= 1; i--) {
      const h = (this.hHead - i + HIST) % HIST, s = this.hs[h], j = this.hj[h];
      if (Math.abs(s) >= CLEAR && this.hp[h] >= STRONG && s > 0 === (this.code.bit(offset + j) === 1)) this.point(j, this.ht[h], s > 0);
    }
  }

  /** Look for a position in the code that the latest symbols match (other than the one locked to). */
  private search(): void {
    const m = this.code.degree;
    // The fewest latest symbols with enough clearly heard among them, so a
    // timecode that just moved is found without waiting for older symbols to pass.
    let n = 0, clear = 0;
    while (n < Math.min(this.hCount, SEARCH) && clear < m + CANDIDATE_MARGIN) {
      if (Math.abs(this.hs[(this.hHead - 1 - n + 2 * HIST) % HIST]) >= CLEAR) clear++;
      n++;
    }
    if (clear < m + CANDIDATE_MARGIN) return;
    const hj = new Float64Array(n), hs = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const h = (this.hHead - n + i + HIST) % HIST;
      hj[i] = this.hj[h];
      hs[i] = this.hs[h];
    }
    const L = this.code.length, cands = new Set<number>();
    // Every run of m consecutive symbols gives a position, as do its variants with the two least certain bits flipped.
    for (let a = 0; a + m <= n; a++) {
      if (hj[a + m - 1] - hj[a] !== m - 1) continue;
      let w = 0, w1 = 0, w2 = 0, c1 = Infinity, c2 = Infinity;
      for (let t = 0; t < m; t++) {
        const s = hs[a + t], c = Math.abs(s);
        if (s > 0) w |= 1 << t;
        if (c < c1) { c2 = c1; w2 = w1; c1 = c; w1 = 1 << t; } else if (c < c2) { c2 = c; w2 = 1 << t; }
      }
      for (const v of [w, w ^ w1, w ^ w2, w ^ w1 ^ w2]) {
        const idx = this.code.indexOf(v);
        if (idx >= 0) cands.add((((idx - hj[a]) % L) + L) % L);
      }
    }
    let best = NaN, bestScore = -Infinity;
    for (const o of cands) {
      if (this.codeLocked && o === this.offset) continue;
      let score = 0, errs = 0;
      for (let i = 0; i < n; i++) {
        const v = this.code.bit(o + hj[i]) ? hs[i] : -hs[i];
        score += v;
        if (v <= -CLEAR && ++errs > 1) break;
      }
      if (errs <= 1 && score > bestScore) { bestScore = score; best = o; }
    }
    if (!Number.isNaN(best)) this.pending = { offset: best, agree: 0, errors: 0 };
  }
}

/** Robust straight-line fit of set time against audio time over a sliding window. */
class SetClock {
  private readonly x = new Float64Array(FIT_POINTS);
  private readonly y = new Float64Array(FIT_POINTS);
  private head = 0;
  private n = 0;
  private rejected = 0;
  private sigma = 0.001;
  /** Resets because points kept disagreeing with the fit. */
  jumps = 0;
  fit: TimecodeFit = SetClock.empty();
  // Scratch.
  private readonly d = new Float64Array(FIT_POINTS);
  private readonly r = new Float64Array(FIT_POINTS);

  private static empty(): TimecodeFit {
    return { codeLocked: false, points: 0, at: 0, time: 0, rate: 1, varTime: Infinity, varRate: 0, last: -Infinity };
  }

  reset(): void {
    this.head = this.n = this.rejected = 0;
    this.fit = SetClock.empty();
  }

  add(x: number, y: number): void {
    const f = this.fit;
    if (f.points >= 10) {
      // A peak from an echo or a stray sound: skip it, unless points keep disagreeing (the timecode moved).
      const r = y - fitTime(f, x);
      if (Math.abs(r) > Math.max(0.03, 6 * this.sigma)) {
        if (++this.rejected < 15) return;
        this.reset();
        this.jumps++;
      }
    }
    this.rejected = 0;
    this.x[this.head] = x;
    this.y[this.head] = y;
    this.head = (this.head + 1) % FIT_POINTS;
    if (this.n < FIT_POINTS) this.n++;
    // Drop points that have left the window.
    while (this.n > 1 && this.x[(this.head - this.n + FIT_POINTS) % FIT_POINTS] < x - FIT_WINDOW) this.n--;
    this.refit(x);
  }

  private refit(last: number): void {
    const n = this.n, d = this.d, xs = this.r;
    let x0 = Infinity, x1 = -Infinity;
    for (let i = 0; i < n; i++) {
      const s = (this.head - n + i + FIT_POINTS) % FIT_POINTS;
      xs[i] = this.x[s];
      d[i] = this.y[s] - this.x[s];
      if (xs[i] < x0) x0 = xs[i];
      if (xs[i] > x1) x1 = xs[i];
    }
    // xs: audio times, d: set time - audio time.
    const slope = n >= 20 && x1 - x0 >= 10;
    const keep = new Uint8Array(n), res = new Float64Array(n);
    // Start from the median offset, so a few stray points do not pull the first pass.
    for (let i = 0; i < n; i++) res[i] = d[i];
    const med = median(res, n);
    for (let i = 0; i < n; i++) res[i] = Math.abs(d[i] - med);
    let sigma = Math.max(3e-4, 1.4826 * median(res, n));
    for (let i = 0; i < n; i++) keep[i] = Math.abs(d[i] - med) <= Math.max(0.002, 4 * sigma) ? 1 : 0;
    let f = this.binFit(keep, x0, slope);
    // Then trim around the line.
    for (let i = 0; i < n; i++) res[i] = Math.abs(d[i] - f.dm - f.c * (xs[i] - f.at));
    const cut = res.slice(0, n);
    sigma = Math.max(3e-4, 1.4826 * median(cut, n));
    for (let i = 0; i < n; i++) keep[i] = res[i] <= Math.max(0.001, 3 * sigma) ? 1 : 0;
    f = this.binFit(keep, x0, slope);
    this.sigma = sigma;
    this.fit = {
      codeLocked: true, points: f.kept, at: f.at, time: f.at + f.dm, rate: 1 + f.c,
      varTime: f.bins >= 3 ? f.s2 / f.bins : (sigma * sigma) / f.kept,
      varRate: slope ? Math.max(f.s2 / f.sxx, 4e-12) : (MAX_RATE / 3) ** 2,
      last,
    };
  }

  /**
   * Line through the means of FIT_BIN-long bins: a phone in motion makes
   * neighbouring points err together, which bins average out, and the
   * scatter of the bin means is honest about what is left.
   */
  private binFit(keep: Uint8Array, x0: number, slope: boolean): { at: number; dm: number; c: number; s2: number; sxx: number; bins: number; kept: number } {
    const n = this.n, xs = this.r, d = this.d;
    const nb = Math.floor(FIT_WINDOW / FIT_BIN) + 2;
    const sx = new Float64Array(nb), sd = new Float64Array(nb), cnt = new Float64Array(nb);
    let kept = 0;
    for (let i = 0; i < n; i++) {
      if (!keep[i]) continue;
      const b = Math.min(nb - 1, Math.floor((xs[i] - x0) / FIT_BIN));
      sx[b] += xs[i]; sd[b] += d[i]; cnt[b]++;
      kept++;
    }
    let bins = 0, at = 0, dm = 0;
    for (let b = 0; b < nb; b++) {
      if (!cnt[b]) continue;
      sx[b] /= cnt[b]; sd[b] /= cnt[b];
      at += sx[b]; dm += sd[b]; bins++;
    }
    at /= bins; dm /= bins;
    let sxx = 0, sxd = 0;
    for (let b = 0; b < nb; b++) if (cnt[b]) { sxx += (sx[b] - at) ** 2; sxd += (sx[b] - at) * (sd[b] - dm); }
    const c = slope && bins >= 5 && sxx > 0 ? Math.max(-MAX_RATE, Math.min(MAX_RATE, sxd / sxx)) : 0;
    let ss = 0;
    for (let b = 0; b < nb; b++) if (cnt[b]) ss += (sd[b] - dm - c * (sx[b] - at)) ** 2;
    const s2 = Math.max(1e-8, ss / Math.max(1, bins - (c ? 2 : 1)));
    return { at, dm, c, s2, sxx, bins, kept };
  }
}
