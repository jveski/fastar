import { createGray, type GrayImage } from './gray';

/** Scratch for pyrDown: a 16-bit buffer and a 32-bit view of the same memory. */
export class PyrScratch {
  readonly t16: Uint16Array;
  readonly t32: Uint32Array;
  readonly t32i: Int32Array;
  constructor(maxSrcWidth: number, maxSrcHeight: number) {
    const stride = ((maxSrcWidth >> 1) + 1) & ~1;
    this.t32 = new Uint32Array((stride * maxSrcHeight) / 2 + 1);
    this.t32i = new Int32Array(this.t32.buffer);
    this.t16 = new Uint16Array(this.t32.buffer);
  }
}

/**
 * Gaussian [1 4 6 4 1] blur fused with 2x decimation. Output pixel j is
 * centred on input pixel 2j, so coordinates map exactly as x_{l+1} = x_l / 2.
 *
 * Horizontal pass: consecutive outputs share three taps, carried in registers
 * (2 loads/output). Vertical pass: SWAR, two 16-bit columns per 32-bit word;
 * each lane's weighted sum is <= 65280, so lanes never carry into each other.
 */
export function pyrDown(src: GrayImage, dst: GrayImage, scratch: PyrScratch): void {
  const sw = src.width, sh = src.height, s = src.data;
  const dw = dst.width, dh = dst.height, d = dst.data;
  const stride = (dw + 1) & ~1; // even, so rows are 32-bit aligned
  const t16 = scratch.t16;
  const xmax = sw - 1;
  if ((sw & 3) === 0 && (s.byteOffset & 3) === 0) {
    horizontalSwar(s, sw, sh, scratch.t32, stride >> 1);
  } else {
    horizontalScalar(s, sw, sh, dw, t16, stride, xmax);
  }
  const ymax = sh - 1;
  const half = stride >> 1;
  const pairs = dw >> 1;
  const ti = scratch.t32i;
  for (let i = 0; i < dh; i++) {
    const c = 2 * i;
    const r0 = (c - 2 < 0 ? 0 : c - 2) * half;
    const r1 = (c - 1 < 0 ? 0 : c - 1) * half;
    const r2 = (c > ymax ? ymax : c) * half;
    const r3 = (c + 1 > ymax ? ymax : c + 1) * half;
    const r4 = (c + 2 > ymax ? ymax : c + 2) * half;
    let o = i * dw;
    for (let k = 0; k < pairs; k++, o += 2) {
      // int32 wrap-around is fine: each 16-bit lane stays <= 65408.
      const v = (ti[r0 + k] + ((ti[r1 + k] + ti[r3 + k]) << 2) + 6 * ti[r2 + k] + ti[r4 + k] + 0x00800080) | 0;
      d[o] = v >>> 8;
      d[o + 1] = v >>> 24;
    }
    if (dw & 1) {
      const j = dw - 1, sr = stride;
      const v = t16[(r0 / half) * sr + j] + 4 * (t16[(r1 / half) * sr + j] + t16[(r3 / half) * sr + j]) + 6 * t16[(r2 / half) * sr + j] + t16[(r4 / half) * sr + j];
      d[o] = (v + 128) >> 8;
    }
  }
}

const u32Views = new WeakMap<ArrayBufferLike, Uint32Array>();
function u32View(s: Uint8Array): Uint32Array {
  let v = u32Views.get(s.buffer);
  if (!v || v.byteOffset !== 0) {
    v = new Uint32Array(s.buffer, 0, s.buffer.byteLength >> 2);
    u32Views.set(s.buffer, v);
  }
  return v;
}

/** Scalar horizontal pass; consecutive outputs share three taps carried in registers. */
function horizontalScalar(s: Uint8Array, sw: number, sh: number, dw: number, t16: Uint16Array, stride: number, xmax: number): void {
  const jEnd = Math.min(dw, ((xmax - 2) >> 1) + 1); // interior: 2j + 2 <= xmax
  for (let y = 0; y < sh; y++) {
    const r = y * sw, o = y * stride;
    const a = s[r], b = s[r + 1], c = s[r + 2];
    t16[o] = 11 * a + 4 * b + c; // taps -2..2 clamped to 0
    let p0 = a, p1 = b, p2 = c;
    let idx = r + 3;
    for (let j = 1; j < jEnd; j++, idx += 2) {
      const q1 = s[idx], q2 = s[idx + 1];
      t16[o + j] = p0 + 4 * (p1 + q1) + 6 * p2 + q2;
      p0 = p2; p1 = q1; p2 = q2;
    }
    for (let j = Math.max(1, jEnd); j < dw; j++) {
      const cc = 2 * j;
      const e = cc + 1 > xmax ? xmax : cc + 1, f = cc + 2 > xmax ? xmax : cc + 2;
      const c0 = cc > xmax ? xmax : cc;
      t16[o + j] = s[r + cc - 2] + 4 * (s[r + cc - 1] + s[r + e]) + 6 * s[r + c0] + s[r + f];
    }
    if (stride !== dw) t16[o + dw] = 0;
  }
}

/**
 * SWAR horizontal pass for widths divisible by 4: one 32-bit load yields two
 * outputs. Taps for outputs (2m, 2m+1) are packed into 16-bit lanes:
 * A=(s[4m-2], s[4m]) B=(s[4m-1], s[4m+1]) C=(s[4m], s[4m+2])
 * D=(s[4m+1], s[4m+3]) E=(s[4m+2], s[4m+4]); out = A + 4(B + D) + 6C + E.
 */
function horizontalSwar(s: Uint8Array, sw: number, sh: number, t32: Uint32Array, half: number): void {
  const src = u32View(s);
  const base = s.byteOffset >> 2;
  const words = sw >> 2;
  for (let y = 0; y < sh; y++) {
    let k = base + y * words;
    const o = y * half;
    let w = src[k];
    const b0 = w & 0xff;
    let p = (b0 << 16) | (b0 << 24); // clamped left border: s[-2] = s[-1] = s[0]
    for (let m = 0; m < words; m++, k++) {
      const n = m + 1 < words ? src[k + 1] : (w >>> 24) * 0x01010101;
      const C = w & 0x00ff00ff;
      const D = (w >>> 8) & 0x00ff00ff;
      const A = ((p >>> 16) & 0xff) | ((w & 0xff) << 16);
      const B = (p >>> 24) | (((w >>> 8) & 0xff) << 16);
      const E = ((w >>> 16) & 0xff) | ((n & 0xff) << 16);
      t32[o + m] = A + 4 * (B + D) + 6 * C + E;
      p = w;
      w = n;
    }
  }
}

/** Preallocated image pyramid. Level 0 is a copy of the source frame. */
export class Pyramid {
  readonly levels: GrayImage[] = [];
  private readonly scratch: PyrScratch;

  constructor(readonly width: number, readonly height: number, readonly numLevels: number) {
    let w = width, h = height;
    for (let l = 0; l < numLevels; l++) {
      this.levels.push(createGray(w, h));
      w = w >> 1;
      h = h >> 1;
    }
    this.scratch = new PyrScratch(width, height);
  }

  /** Build from luma data of size width*height. */
  build(luma: Uint8Array): this {
    this.levels[0].data.set(luma.subarray(0, this.width * this.height));
    for (let l = 1; l < this.numLevels; l++) pyrDown(this.levels[l - 1], this.levels[l], this.scratch);
    return this;
  }
}
