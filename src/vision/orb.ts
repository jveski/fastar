import type { GrayImage } from '../image/gray';
import { Rng } from '../math/rng';

/** Patch radius used for orientation; descriptor samples stay within PATTERN_RADIUS. */
export const ORIENT_RADIUS = 15;
export const PATTERN_RADIUS = 13;
/** Keypoints must be at least this far from the border. */
export const ORB_BORDER = 16;

const ANGLE_BINS = 30;
const NPAIRS = 256;

/** Rotated BRIEF patterns: for each angle bin, 256 pairs of (dx1, dy1, dx2, dy2). */
const PATTERNS: Int8Array[] = (() => {
  const rng = new Rng(0x0b1e5);
  const base = new Float64Array(NPAIRS * 4);
  const sigma = 31 / 5;
  for (let i = 0; i < NPAIRS * 2; i++) {
    let x: number, y: number;
    do {
      x = rng.gauss() * sigma;
      y = rng.gauss() * sigma;
    } while (x * x + y * y > PATTERN_RADIUS * PATTERN_RADIUS);
    base[2 * i] = x;
    base[2 * i + 1] = y;
  }
  const out: Int8Array[] = [];
  for (let b = 0; b < ANGLE_BINS; b++) {
    const a = (b * 2 * Math.PI) / ANGLE_BINS;
    const c = Math.cos(a), s = Math.sin(a);
    const p = new Int8Array(NPAIRS * 4);
    for (let i = 0; i < NPAIRS * 2; i++) {
      const x = base[2 * i], y = base[2 * i + 1];
      p[2 * i] = Math.round(c * x - s * y);
      p[2 * i + 1] = Math.round(s * x + c * y);
    }
    out.push(p);
  }
  return out;
})();

const UMAX: Int32Array = (() => {
  const u = new Int32Array(ORIENT_RADIUS + 1);
  for (let v = 0; v <= ORIENT_RADIUS; v++) u[v] = Math.round(Math.sqrt(ORIENT_RADIUS * ORIENT_RADIUS - v * v));
  return u;
})();

/** Intensity-centroid orientation in radians, in [-pi, pi]. */
export function orientation(img: GrayImage, x: number, y: number): number {
  const w = img.width, d = img.data;
  const c = y * w + x;
  let m10 = 0, m01 = 0;
  for (let u = -ORIENT_RADIUS; u <= ORIENT_RADIUS; u++) m10 += u * d[c + u];
  for (let v = 1; v <= ORIENT_RADIUS; v++) {
    const um = UMAX[v];
    let vsum = 0;
    for (let u = -um; u <= um; u++) {
      const top = d[c - v * w + u], bot = d[c + v * w + u];
      vsum += bot - top;
      m10 += u * (bot + top);
    }
    m01 += v * vsum;
  }
  return Math.atan2(m01, m10);
}

/** Linear pixel offsets of the rotated patterns, cached per image width. */
const offsetCache = new Map<number, Int32Array[]>();
function patternOffsets(w: number): Int32Array[] {
  let o = offsetCache.get(w);
  if (!o) {
    o = PATTERNS.map((p) => {
      const off = new Int32Array(NPAIRS * 2);
      for (let i = 0; i < NPAIRS * 2; i++) off[i] = p[2 * i + 1] * w + p[2 * i];
      return off;
    });
    offsetCache.set(w, o);
  }
  return o;
}

/** 256-bit rotated BRIEF on a smoothed image; writes 8 words at desc[o..o+8). */
export function describe(img: GrayImage, x: number, y: number, angle: number, desc: Uint32Array, o: number): void {
  let bin = Math.round((angle / (2 * Math.PI)) * ANGLE_BINS) % ANGLE_BINS;
  if (bin < 0) bin += ANGLE_BINS;
  const off = patternOffsets(img.width)[bin];
  const d = img.data;
  const c = y * img.width + x;
  for (let word = 0, i = 0; word < 8; word++) {
    let bits = 0;
    for (let k = 0; k < 32; k++, i += 2) {
      bits |= ((d[c + off[i]] - d[c + off[i + 1]]) >>> 31) << k;
    }
    desc[o + word] = bits >>> 0;
  }
}

/** 16-bit popcount table: two lookups per word beat the bit-twiddling version in JS. */
const POP16: Uint8Array = (() => {
  const t = new Uint8Array(65536);
  for (let i = 1; i < 65536; i++) t[i] = (i & 1) + t[i >>> 1];
  return t;
})();

/** Hamming distance between 256-bit descriptors a[ao..ao+8) and b[bo..bo+8). */
export function hamming(a: Uint32Array, ao: number, b: Uint32Array, bo: number): number {
  const T = POP16;
  let x = a[ao] ^ b[bo], s = T[x & 0xffff] + T[x >>> 16];
  x = a[ao + 1] ^ b[bo + 1]; s += T[x & 0xffff] + T[x >>> 16];
  x = a[ao + 2] ^ b[bo + 2]; s += T[x & 0xffff] + T[x >>> 16];
  x = a[ao + 3] ^ b[bo + 3]; s += T[x & 0xffff] + T[x >>> 16];
  x = a[ao + 4] ^ b[bo + 4]; s += T[x & 0xffff] + T[x >>> 16];
  x = a[ao + 5] ^ b[bo + 5]; s += T[x & 0xffff] + T[x >>> 16];
  x = a[ao + 6] ^ b[bo + 6]; s += T[x & 0xffff] + T[x >>> 16];
  x = a[ao + 7] ^ b[bo + 7]; s += T[x & 0xffff] + T[x >>> 16];
  return s;
}

/** Growable struct-of-arrays keypoint set: kp = [x, y, scale, angle] in level-0 pixels. */
export class FeatureSet {
  kp: Float32Array;
  desc: Uint32Array;
  count = 0;

  constructor(capacity = 1024) {
    this.kp = new Float32Array(capacity * 4);
    this.desc = new Uint32Array(capacity * 8);
  }

  get capacity(): number {
    return this.kp.length / 4;
  }

  reserve(n: number): void {
    if (n <= this.capacity) return;
    const cap = Math.max(n, this.capacity * 2);
    const kp = new Float32Array(cap * 4);
    kp.set(this.kp);
    const desc = new Uint32Array(cap * 8);
    desc.set(this.desc);
    this.kp = kp;
    this.desc = desc;
  }

  clear(): void {
    this.count = 0;
  }
}

/**
 * Orientation + descriptors for detected keypoints of one pyramid level.
 * `orientImg` should be the unsmoothed level, `descImg` a smoothed copy.
 * map(x) converts level pixels to level-0 pixels.
 */
export function describeLevel(
  orientImg: GrayImage,
  descImg: GrayImage,
  xs: Float32Array,
  ys: Float32Array,
  n: number,
  scale: number,
  offset: number,
  out: FeatureSet,
): void {
  out.reserve(out.count + n);
  const w = orientImg.width, h = orientImg.height;
  for (let i = 0; i < n; i++) {
    const x = xs[i], y = ys[i];
    if (x < ORB_BORDER || y < ORB_BORDER || x >= w - ORB_BORDER || y >= h - ORB_BORDER) continue;
    const a = orientation(orientImg, x, y);
    const k = out.count++;
    out.kp[4 * k] = (x + offset) * scale - offset;
    out.kp[4 * k + 1] = (y + offset) * scale - offset;
    out.kp[4 * k + 2] = scale;
    out.kp[4 * k + 3] = a;
    describe(descImg, x, y, a, out.desc, 8 * k);
  }
}
