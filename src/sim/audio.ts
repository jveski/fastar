// Simulated acoustic channel for the audio timecode: a PA playing music with
// timecode, heard by a phone some distance away that moves about, with
// echoes, noise, a clipping microphone and its own sample clock.
import { Rng } from '../math/rng';

const C = 343;

export interface ChannelOptions {
  /** Receiver sample rate (Hz). */
  rate: number;
  /** Receiver clock error (ppm; positive runs fast). */
  ppm?: number;
  /** Distance from the speaker (m). */
  distance?: number;
  /** Back-and-forth motion along the line to the speaker: amplitude (m) and frequency (Hz). */
  sway?: { amplitude: number; frequency: number };
  /** Reflections and other speakers: extra path delay (s) and gain. */
  echoes?: Array<[number, number]>;
  /** Microphone noise (RMS). */
  noise?: number;
  /** The microphone clips at this level. */
  clip?: number;
  seed?: number;
}

export interface Received {
  samples: Float32Array;
  /** Transmit time (s from the first transmitted sample) of the direct sound heard at receiver time t (sample / rate). */
  emittedAt: (t: number) => number;
}

/** Band-limited interpolation of x (rate fs) at fractional sample positions, from a table of windowed-sinc phases. */
class Interpolator {
  private static readonly HALF = 16;
  private static readonly PHASES = 512;
  private readonly table: Float32Array;

  constructor(private readonly x: Float32Array, fs: number, outRate: number) {
    const h = Interpolator.HALF, P = Interpolator.PHASES, fc = (0.95 * Math.min(fs, outRate)) / 2 / fs;
    const t = (this.table = new Float32Array((P + 1) * 2 * h));
    for (let p = 0; p <= P; p++) {
      for (let k = 0; k < 2 * h; k++) {
        const d = p / P + h - 1 - k;
        const sinc = d === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * d) / (Math.PI * d);
        const w = Math.abs(d) >= h ? 0 : 0.42 + 0.5 * Math.cos((Math.PI * d) / h) + 0.08 * Math.cos((2 * Math.PI * d) / h);
        t[p * 2 * h + k] = sinc * w;
      }
    }
  }

  at(pos: number): number {
    const x = this.x, h = Interpolator.HALF, i0 = Math.floor(pos);
    const row = Math.round((pos - i0) * Interpolator.PHASES) * 2 * h, t = this.table;
    let s = 0;
    const a = i0 - h + 1;
    if (a >= 0 && a + 2 * h <= x.length) {
      for (let k = 0; k < 2 * h; k++) s += x[a + k] * t[row + k];
    } else {
      for (let k = 0; k < 2 * h; k++) if (a + k >= 0 && a + k < x.length) s += x[a + k] * t[row + k];
    }
    return s;
  }
}

/** What a phone records of tx (played at txRate from time 0) over `seconds` of its own clock. */
export function receive(tx: Float32Array, txRate: number, seconds: number, o: ChannelOptions): Received {
  const n = Math.floor(seconds * o.rate), out = new Float32Array(n), rng = new Rng(o.seed ?? 1);
  const ip = new Interpolator(tx, txRate, o.rate);
  const scale = 1 + (o.ppm ?? 0) * 1e-6, dist = o.distance ?? 10, sway = o.sway;
  const emittedAt = (t: number) => {
    const tt = t / scale;
    const d = dist + (sway ? sway.amplitude * Math.sin(2 * Math.PI * sway.frequency * tt) : 0);
    return tt - d / C;
  };
  const att = 10 / Math.max(1, dist);
  for (let i = 0; i < n; i++) {
    const te = emittedAt(i / o.rate);
    let v = ip.at(te * txRate);
    for (const [dl, g] of o.echoes ?? []) v += g * ip.at((te - dl) * txRate);
    v = v * att + (o.noise ?? 0) * rng.gauss();
    if (o.clip !== undefined) v = Math.max(-o.clip, Math.min(o.clip, v));
    out[i] = v;
  }
  return { samples: out, emittedAt };
}

/** Something like dance music: a kick on every beat, open hats between, a noisy pad; peak about 1. */
export function makeMusic(rate: number, seconds: number, seed = 7, bpm = 126): Float32Array {
  const n = Math.floor(rate * seconds), out = new Float32Array(n), rng = new Rng(seed);
  const beat = 60 / bpm;
  // Pink-ish noise (Paul Kellet's filter).
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  for (let i = 0; i < n; i++) {
    const t = i / rate, w = rng.gauss() * 0.05;
    b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759; b2 = 0.969 * b2 + w * 0.153852;
    b3 = 0.8665 * b3 + w * 0.3104856; b4 = 0.55 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.016898;
    const pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362;
    b6 = w * 0.115926;
    const tb = t % beat, th = (t + beat / 2) % beat;
    const kick = Math.exp(-tb / 0.12) * Math.sin(2 * Math.PI * (50 * tb + 60 * (1 - Math.exp(-tb / 0.03)) * 0.03));
    const hat = Math.exp(-th / 0.04) * rng.gauss() * 0.15;
    out[i] = 0.6 * kick + hat + 0.8 * pink;
  }
  return out;
}
