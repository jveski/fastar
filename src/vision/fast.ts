import { i32View, type GrayImage } from '../image/gray';

// Bresenham circle of radius 3, clockwise from the top.
const CIRCLE: ReadonlyArray<[number, number]> = [
  [0, -3], [1, -3], [2, -2], [3, -1], [3, 0], [3, 1], [2, 2], [1, 3],
  [0, 3], [-1, 3], [-2, 2], [-3, 1], [-3, 0], [-3, -1], [-2, -2], [-1, -3],
];

export interface DetectOptions {
  /** FAST intensity threshold. */
  threshold: number;
  /** Max points kept after grid-bucketed NMS. */
  maxPoints: number;
  /** Min distance to the image border. */
  border: number;
  /** Grid cell size (px) for spatial bucketing. */
  cell: number;
}

/**
 * FAST-9 detection, Harris scoring, 3x3 non-maximum suppression and
 * grid-bucketed selection. Scratch buffers are sized for the largest image
 * and reused.
 */
export class FastDetector {
  /** Selected keypoints (level pixels) and their Harris scores. */
  readonly x: Float32Array;
  readonly y: Float32Array;
  readonly score: Float32Array;
  count = 0;
  /** Number of FAST corners before NMS in the last call. */
  candidates = 0;
  /** Use the 4-pixel SWAR pre-test when the image layout allows it. */
  swarEnabled = true;

  private readonly cx: Int32Array;
  private readonly cy: Int32Array;
  private readonly cs: Float32Array;
  private readonly order: Uint32Array;
  private readonly map: Float32Array;
  private readonly offsets = new Int32Array(16);
  private offsetsWidth = -1;
  private cellCounts = new Int32Array(0);

  constructor(maxWidth: number, maxHeight: number, readonly maxCandidates = 30000, maxPoints = 1000) {
    this.cx = new Int32Array(maxCandidates);
    this.cy = new Int32Array(maxCandidates);
    this.cs = new Float32Array(maxCandidates);
    this.order = new Uint32Array(maxCandidates);
    this.map = new Float32Array(maxWidth * maxHeight);
    this.x = new Float32Array(maxPoints);
    this.y = new Float32Array(maxPoints);
    this.score = new Float32Array(maxPoints);
  }

  detect(img: GrayImage, o: DetectOptions): number {
    const w = img.width, h = img.height, d = img.data;
    if (this.offsetsWidth !== w) {
      for (let k = 0; k < 16; k++) this.offsets[k] = CIRCLE[k][1] * w + CIRCLE[k][0];
      this.offsetsWidth = w;
    }
    const off = this.offsets;
    const t = o.threshold;
    const b = Math.max(4, o.border);
    const cx = this.cx, cy = this.cy, cs = this.cs, map = this.map;
    let nc = 0;
    const maxC = this.maxCandidates;
    const swar = this.swarEnabled && (w & 3) === 0 && (d.byteOffset & 3) === 0 && t < 255;
    const d32 = swar ? i32View(d.buffer) : null;
    const T1 = (t + 1) * 0x10001;
    const BIAS = 0x02000200;
    for (let y = b; y < h - b && nc < maxC; y++) {
      const row = y * w;
      // Words fully inside [b, w - b) use the SWAR pre-test; the rest are scalar.
      const kb = swar ? (b + 3) >> 2 : 0, ke = swar ? (w - b) >> 2 : 0;
      const xs = swar ? 4 * kb : w - b;
      for (let x = b; x < Math.min(xs, w - b); x++) {
        const s = this.testPixel(d, row + x, t, off);
        if (s > 0) { cx[nc] = x; cy[nc] = y; cs[nc] = s; map[row + x] = s; if (++nc >= maxC) break; }
      }
      if (swar && d32) {
        const base = (d.byteOffset + row) >> 2, wq = w >> 2;
        let prev = d32[base + kb - 1], cur = d32[base + kb];
        for (let k = kb; k < ke && nc < maxC; k++) {
          const next = d32[base + k + 1];
          const U = d32[base + k - 3 * wq], D = d32[base + k + 3 * wq];
          const L = (prev >>> 8) | (cur << 24), R = (cur >>> 24) | (next << 8);
          const Ce = cur & 0xff00ff, Co = (cur >>> 8) & 0xff00ff;
          let e = U & 0xff00ff;
          let ve = (e + BIAS - Ce - T1) | (Ce + BIAS - e - T1);
          e = D & 0xff00ff;
          ve |= (e + BIAS - Ce - T1) | (Ce + BIAS - e - T1);
          e = L & 0xff00ff;
          let he = (e + BIAS - Ce - T1) | (Ce + BIAS - e - T1);
          e = R & 0xff00ff;
          he |= (e + BIAS - Ce - T1) | (Ce + BIAS - e - T1);
          e = (U >>> 8) & 0xff00ff;
          let vo = (e + BIAS - Co - T1) | (Co + BIAS - e - T1);
          e = (D >>> 8) & 0xff00ff;
          vo |= (e + BIAS - Co - T1) | (Co + BIAS - e - T1);
          e = (L >>> 8) & 0xff00ff;
          let ho = (e + BIAS - Co - T1) | (Co + BIAS - e - T1);
          e = (R >>> 8) & 0xff00ff;
          ho |= (e + BIAS - Co - T1) | (Co + BIAS - e - T1);
          const me = (ve & he) >>> 9 & 0x10001, mo = (vo & ho) >>> 9 & 0x10001;
          prev = cur;
          cur = next;
          if ((me | mo) === 0) continue;
          const x0 = 4 * k;
          // Lanes: even word -> pixels 0 (bit 0) and 2 (bit 16); odd -> 1 and 3.
          if (me & 1) { const s = this.testPixel(d, row + x0, t, off); if (s > 0) { cx[nc] = x0; cy[nc] = y; cs[nc] = s; map[row + x0] = s; nc++; } }
          if (mo & 1) { const s = this.testPixel(d, row + x0 + 1, t, off); if (s > 0) { cx[nc] = x0 + 1; cy[nc] = y; cs[nc] = s; map[row + x0 + 1] = s; nc++; } }
          if (me & 0x10000) { const s = this.testPixel(d, row + x0 + 2, t, off); if (s > 0) { cx[nc] = x0 + 2; cy[nc] = y; cs[nc] = s; map[row + x0 + 2] = s; nc++; } }
          if (mo & 0x10000) { const s = this.testPixel(d, row + x0 + 3, t, off); if (s > 0) { cx[nc] = x0 + 3; cy[nc] = y; cs[nc] = s; map[row + x0 + 3] = s; nc++; } }
          if (nc >= maxC - 4) break;
        }
        for (let x = 4 * ke; x < w - b && nc < maxC; x++) {
          const s = this.testPixel(d, row + x, t, off);
          if (s > 0) { cx[nc] = x; cy[nc] = y; cs[nc] = s; map[row + x] = s; nc++; }
        }
      }
    }
    this.candidates = nc;
    // 3x3 NMS on the FAST score, then Harris on the survivors only.
    let ns = 0;
    const order = this.order;
    for (let i = 0; i < nc; i++) {
      const p = cy[i] * w + cx[i];
      const s = cs[i];
      if (
        map[p - w - 1] > s || map[p - w] > s || map[p - w + 1] > s || map[p - 1] > s ||
        map[p + 1] >= s || map[p + w - 1] >= s || map[p + w] >= s || map[p + w + 1] >= s
      ) continue;
      order[ns++] = i;
    }
    for (let i = 0; i < nc; i++) map[cy[i] * w + cx[i]] = 0;
    let nh = 0;
    for (let k = 0; k < ns; k++) {
      const i = order[k];
      const hs = harris(d, w, cx[i], cy[i]);
      if (hs <= 0) continue;
      cs[i] = hs;
      order[nh++] = i;
    }
    ns = nh;
    // Strongest first, then grid-bucketed selection for spatial spread.
    const sorted = order.subarray(0, ns).sort((a, c) => cs[c] - cs[a]);
    const gw = Math.ceil(w / o.cell), gh = Math.ceil(h / o.cell);
    const cells = gw * gh;
    if (this.cellCounts.length < cells) this.cellCounts = new Int32Array(cells);
    const cc = this.cellCounts;
    cc.fill(0, 0, cells);
    const maxP = Math.min(o.maxPoints, this.x.length);
    const perCell = Math.max(1, Math.ceil((maxP / cells) * 1.5));
    let n = 0;
    for (let pass = 0; pass < 2 && n < maxP; pass++) {
      for (let k = 0; k < ns && n < maxP; k++) {
        const i = sorted[k];
        if (i === 0xffffffff) continue;
        const cell = ((cy[i] / o.cell) | 0) * gw + ((cx[i] / o.cell) | 0);
        if (pass === 0 && cc[cell] >= perCell) continue;
        cc[cell]++;
        this.x[n] = cx[i]; this.y[n] = cy[i]; this.score[n] = cs[i];
        n++;
        sorted[k] = 0xffffffff;
      }
    }
    this.count = n;
    return n;
  }

  /**
   * Exact FAST-9 test at linear index p. Returns a positive score (sum of
   * absolute differences beyond the threshold) for corners, else 0.
   */
  private testPixel(d: Uint8Array, p: number, t: number, off: Int32Array): number {
    const c = d[p];
    const hi = c + t, lo = c - t;
    const v0 = d[p + off[0]], v4 = d[p + off[4]], v8 = d[p + off[8]], v12 = d[p + off[12]];
    // A 9-arc of one polarity covers two adjacent compass points (0/4/8/12) of that polarity.
    const nb = (v0 > hi ? 1 : 0) | (v4 > hi ? 2 : 0) | (v8 > hi ? 4 : 0) | (v12 > hi ? 8 : 0);
    const nd = (v0 < lo ? 1 : 0) | (v4 < lo ? 2 : 0) | (v8 < lo ? 4 : 0) | (v12 < lo ? 8 : 0);
    if (!ADJ2[nb] && !ADJ2[nd]) return 0;
    let bright = 0, dark = 0, score = 0;
    for (let k = 0; k < 16; k++) {
      const v = d[p + off[k]];
      // Branch-free: sign bits select bright/dark (data-dependent branches mispredict).
      const bk = (hi - v) >>> 31, dk = (v - lo) >>> 31;
      bright |= bk << k;
      dark |= dk << k;
      score += bk * (v - hi) + dk * (lo - v);
    }
    if (!hasArc9(bright) && !hasArc9(dark)) return 0;
    return score + 1;
  }
}

/** ADJ2[m]: the 4-bit compass mask m (bits 0,4,8,12 -> 1,2,4,8) has two cyclically adjacent bits. */
const ADJ2: Uint8Array = (() => {
  const t = new Uint8Array(16);
  for (let m = 0; m < 16; m++) t[m] = (m & (((m << 1) | (m >> 3)) & 15)) !== 0 ? 1 : 0;
  return t;
})();

function hasArc9(m: number): boolean {
  if (m === 0) return false;
  const m2 = m | (m << 16);
  let r = m2;
  for (let k = 1; k < 9; k++) r &= m2 >>> k;
  return (r & 0xffff) !== 0;
}

/** Harris response over a 5x5 window of central-difference gradients. */
function harris(d: Uint8Array, w: number, x: number, y: number): number {
  let sxx = 0, syy = 0, sxy = 0;
  for (let j = -2; j <= 2; j++) {
    let p = (y + j) * w + x - 2;
    for (let i = -2; i <= 2; i++, p++) {
      const gx = d[p + 1] - d[p - 1];
      const gy = d[p + w] - d[p - w];
      sxx += gx * gx; syy += gy * gy; sxy += gx * gy;
    }
  }
  const tr = sxx + syy;
  return sxx * syy - sxy * sxy - 0.04 * tr * tr;
}
