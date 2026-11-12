// Audio timecode: the parameters and position code shared by the generator
// and the listeners.
//
// The timecode is a stream of 100 ms frequency sweeps just below the top of
// the audible band (19 to 20 kHz by default): a sweep up for a 1 bit, down
// for a 0. The bits follow a maximal-length sequence, as on DJ timecode
// vinyl: any `degree` consecutive bits occur once in the whole sequence, so
// a listener that has heard a few seconds knows exactly which symbol it is
// hearing, and the end of each sweep marks the time to well under a
// millisecond. Set time t is in symbol floor(t / symbol); the sequence
// repeats after (2^degree - 1) symbols (7.3 hours at the defaults).

export interface TimecodeOptions {
  /**
   * Shared by the generator and its listeners: each key gives a different
   * sequence, so listeners never lock onto another event's timecode (or an
   * old recording of it). Default 'fastar'.
   */
  key?: string;
  /** Band of the sweeps (Hz, default [19000, 20000]). */
  band?: [number, number];
  /** Symbol duration (s, default 0.1). Longer symbols can be played quieter but take longer to lock. */
  symbol?: number;
  /** Sequence degree (10 to 22, default 18): the timecode repeats after (2^degree - 1) symbols. */
  degree?: number;
}

export interface TimecodeParams {
  key: string;
  /** Band edges (Hz). */
  f0: number;
  f1: number;
  /** Symbol duration (s). */
  symbol: number;
  /** Start of each symbol spent moving from the previous sweep's end frequency to this one's start (s). */
  transition: number;
  degree: number;
  /** Symbols before the sequence repeats. */
  length: number;
  /** Seconds before the timecode repeats. */
  period: number;
  /** Feedback polynomial of the sequence (bit k: coefficient of x^k). */
  poly: number;
}

export const DEFAULT_TIMECODE = { key: 'fastar', band: [19000, 20000] as [number, number], symbol: 0.1, degree: 18 };

export function timecodeParams(o: TimecodeOptions = {}): TimecodeParams {
  const key = o.key ?? DEFAULT_TIMECODE.key;
  const [f0, f1] = o.band ?? DEFAULT_TIMECODE.band;
  const symbol = o.symbol ?? DEFAULT_TIMECODE.symbol;
  const degree = o.degree ?? DEFAULT_TIMECODE.degree;
  if (!(f0 > 0 && f1 > f0)) throw new Error(`fastar: bad timecode band ${f0}-${f1} Hz`);
  if (f1 - f0 < 200) throw new Error('fastar: the timecode band must be at least 200 Hz wide');
  if (!(symbol >= 0.02 && symbol <= 1)) throw new Error('fastar: the timecode symbol must be 0.02 to 1 s');
  if (!(Number.isInteger(degree) && degree >= 10 && degree <= 22)) throw new Error('fastar: the timecode degree must be an integer from 10 to 22');
  const length = 2 ** degree - 1;
  return {
    key, f0, f1, symbol, transition: Math.min(0.004, 0.1 * symbol), degree, length, period: length * symbol,
    poly: primitivePoly(degree, hash(key)),
  };
}

function hash(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193);
  return h >>> 0;
}

/** a * b mod g over GF(2), for polynomials below degree m. */
function mulmod(a: number, b: number, g: number, m: number): number {
  let r = 0;
  for (let k = m - 1; k >= 0; k--) {
    r <<= 1;
    if ((r >>> m) & 1) r ^= g;
    if ((b >>> k) & 1) r ^= a;
  }
  return r;
}

/** x^e mod g. */
function xpow(e: number, g: number, m: number): number {
  let r = 1, b = 2;
  while (e > 0) {
    if (e % 2 === 1) r = mulmod(r, b, g, m);
    b = mulmod(b, b, g, m);
    e = Math.floor(e / 2);
  }
  return r;
}

function primeFactors(n: number): number[] {
  const out: number[] = [];
  for (let p = 2; p * p <= n; p++) {
    if (n % p) continue;
    out.push(p);
    while (n % p === 0) n /= p;
  }
  if (n > 1) out.push(n);
  return out;
}

/** g (degree m, with its x^m term) is primitive: x has order 2^m - 1 modulo g. */
export function isPrimitive(g: number, m: number): boolean {
  if (!(g & 1) || g >>> m !== 1) return false;
  const L = 2 ** m - 1;
  if (xpow(L, g, m) !== 1) return false;
  for (const p of primeFactors(L)) if (xpow(L / p, g, m) === 1) return false;
  return true;
}

/** The first primitive polynomial of degree m at or after a seed. */
export function primitivePoly(m: number, seed: number): number {
  const half = 2 ** (m - 1);
  for (let k = 0; k < half; k++) {
    const g = 2 ** m + 2 * ((seed + k) % half) + 1;
    if (isPrimitive(g, m)) return g;
  }
  throw new Error('unreachable: every degree has primitive polynomials');
}

/**
 * The maximal-length sequence a_0, a_1, ...: a_{i+m} is the parity of
 * a_i..a_{i+m-1} under the polynomial's lower coefficients, from
 * a_0 = 1, a_1..a_{m-1} = 0. The window at i (a_i in bit 0 up to a_{i+m-1}
 * in bit m-1) occurs at i only.
 */
export class PositionCode {
  readonly bits: Uint8Array;
  readonly length: number;
  private table: Int32Array | null = null;

  constructor(readonly degree: number, readonly poly: number) {
    const L = (this.length = 2 ** degree - 1);
    const bits = (this.bits = new Uint8Array(L));
    const taps = poly & L, top = degree - 1;
    let w = 1;
    for (let i = 0; i < L; i++) {
      bits[i] = w & 1;
      w = (w >>> 1) | (parity(w & taps) << top);
    }
  }

  /** Bit at index i (any integer: the sequence repeats). */
  bit(i: number): number {
    const L = this.length;
    return this.bits[((i % L) + L) % L];
  }

  /** Index of a window of `degree` bits (see the class comment), or -1 for the all-zero window. */
  indexOf(window: number): number {
    if (!this.table) {
      const L = this.length, m = this.degree, t = (this.table = new Int32Array(L + 1).fill(-1));
      let w = 0;
      for (let k = 0; k < m; k++) w |= this.bits[k] << k;
      for (let i = 0; i < L; i++) {
        t[w] = i;
        w = (w >>> 1) | (this.bits[(i + m) % L] << (m - 1));
      }
    }
    return window > 0 && window <= this.length ? this.table[window] : -1;
  }
}

function parity(v: number): number {
  v ^= v >>> 16;
  v ^= v >>> 8;
  v ^= v >>> 4;
  v ^= v >>> 2;
  v ^= v >>> 1;
  return v & 1;
}
