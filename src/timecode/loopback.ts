// Loopback calibration: how long sound takes from the microphone to the
// page's audio clock, measured by playing a short burst through the phone's
// own speaker and finding it in what the microphone hears.
//
// Browsers do not report this delay reliably. WebKit's microphone path
// buffers by an amount fixed when the stream starts (often a few hundred
// milliseconds) that can change after an underrun, so it has to be measured.
// The burst is six short chirps at uneven spacing: lined up against the
// recording, only the true delay matches more than one of them.
import { Downconverter } from './decoder';
import { median } from '../math/select';

export const LOOPBACK = {
  /** Chirp band (Hz): audible, where phone speakers are loud, well away from the timecode. */
  f0: 2500,
  f1: 7500,
  /** Chirp length (s). */
  length: 0.04,
  /** Chirp starts within the burst (s): no two differences equal, so shifted copies never line up. */
  offsets: [0, 0.13, 0.29, 0.47, 0.68, 0.92],
  /** Peak level. */
  level: 0.2,
};

/** The burst to play, starting with the first chirp. */
export function loopbackSignal(sampleRate: number): Float32Array {
  const { f0, f1, length, offsets, level } = LOOPBACK;
  const n = Math.round(length * sampleRate), B = f1 - f0;
  const out = new Float32Array(Math.ceil((offsets[offsets.length - 1] + length) * sampleRate) + 1);
  for (const o of offsets) {
    const at = Math.round(o * sampleRate);
    for (let i = 0; i < n; i++) {
      const u = i / sampleRate;
      out[at + i] = level * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1))) * Math.sin(2 * Math.PI * (f0 * u + 0.5 * (B / length) * u * u));
    }
  }
  return out;
}

/** Strongest single chirp over the noise (power ratio) for the burst to count as heard. */
const MIN_SNR = 30;
/** Chirps that must be heard, at the same delay to within a millisecond. */
const MIN_CHIRPS = 4;

/**
 * Finds the burst played from audio time `start` (s; its first chirp's
 * start, as rendered) in the microphone audio. push() returns the delay (s)
 * once decided, null if the burst was not heard, undefined until then.
 */
export class LoopbackDetector {
  private readonly from: number;
  private readonly to: number;
  private readonly buf: Float32Array;
  private filled = 0;
  private done = false;

  constructor(readonly sampleRate: number, readonly start: number, readonly maxDelay = 3) {
    const { offsets, length } = LOOPBACK;
    this.from = Math.floor(start * sampleRate);
    this.to = Math.ceil((start + offsets[offsets.length - 1] + length + maxDelay) * sampleRate);
    this.buf = new Float32Array(this.to - this.from);
  }

  push(samples: Float32Array, frame: number): number | null | undefined {
    if (this.done) return undefined;
    const a = Math.max(frame, this.from), b = Math.min(frame + samples.length, this.to);
    if (b > a) {
      this.buf.set(samples.subarray(a - frame, b - frame), a - this.from);
      this.filled = Math.max(this.filled, b - this.from);
    }
    if (frame + samples.length < this.to) return undefined;
    this.done = true;
    return this.detect();
  }

  private detect(): number | null {
    const fs = this.sampleRate, { f0, f1, length, offsets } = LOOPBACK, B = f1 - f0, fc = (f0 + f1) / 2;
    const dc = new Downconverter(fs, fc, B / 2 + 200);
    const fd = dc.rate, tick = dc.D / fs;
    const re = new Float32Array(Math.ceil(this.buf.length / dc.D) + 1), im = new Float32Array(re.length);
    const n = dc.process(this.buf, re, im);
    // Audio time of baseband sample 0 (see TimecodeDecoder.push).
    const t0 = (this.from + dc.D - 1 - dc.delay) / fs;
    // Matched filter for one chirp, ending at the newest sample.
    const m = Math.floor(length * fd), tr = new Float32Array(m), ti = new Float32Array(m);
    for (let i = 0; i < m; i++) {
      const u = length + (i - (m - 1)) / fd;
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * u) / length);
      const ph = 2 * Math.PI * ((f0 - fc) * u + 0.5 * (B / length) * u * u);
      tr[i] = w * Math.cos(ph);
      ti[i] = w * Math.sin(ph);
    }
    const p = new Float64Array(n);
    for (let k = m - 1; k < n; k++) {
      let sr = 0, si = 0;
      for (let i = 0; i < m; i++) {
        const a = re[k - m + 1 + i], b = im[k - m + 1 + i];
        sr += a * tr[i] + b * ti[i];
        si += b * tr[i] - a * ti[i];
      }
      p[k] = sr * sr + si * si;
    }
    const noise = median(p.slice(m - 1), n - m + 1) / Math.LN2;
    if (!(noise > 0)) return null;
    // Sample index where chirp c ends, for delay d (s).
    const at = (c: number, d: number) => (this.start + offsets[c] + length + d - t0) / tick;
    // Coarse: the delay at which the chirps' correlation, summed over the burst, peaks.
    let best = -1, bestSum = -1;
    const steps = Math.floor(this.maxDelay / tick);
    for (let s = 0; s <= steps; s++) {
      let sum = 0;
      for (let c = 0; c < offsets.length; c++) {
        const k = Math.round(at(c, s * tick));
        if (k >= m - 1 && k < n) sum += Math.min(p[k] / noise, 1e4);
      }
      if (sum > bestSum) { bestSum = sum; best = s * tick; }
    }
    // Fine: each chirp's own peak near there, to a fraction of a sample.
    const delays: number[] = [];
    let strongest = 0;
    for (let c = 0; c < offsets.length; c++) {
      const kc = Math.round(at(c, best));
      let kk = -1;
      for (let k = Math.max(m, kc - 3); k <= Math.min(n - 2, kc + 3); k++) if (kk < 0 || p[k] > p[kk]) kk = k;
      if (kk < 0 || p[kk] / noise < MIN_SNR / 3) continue;
      strongest = Math.max(strongest, p[kk] / noise);
      const y0 = p[kk - 1], y1 = p[kk], y2 = p[kk + 1], den = y0 - 2 * y1 + y2;
      const d = den < 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (y0 - y2)) / den)) : 0;
      delays.push(t0 + (kk + d) * tick - (this.start + offsets[c] + length));
    }
    if (strongest < MIN_SNR || delays.length < MIN_CHIRPS) return null;
    const med = median(Float64Array.from(delays), delays.length);
    const agree = delays.filter((d) => Math.abs(d - med) < 0.001).length;
    return agree >= MIN_CHIRPS ? med : null;
  }
}
