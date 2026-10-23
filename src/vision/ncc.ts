import type { GrayImage } from '../image/gray';

/**
 * Template warping plus NCC search with a subpixel quadratic peak fit, used
 * for drift correction against the target image. Scratch memory is
 * preallocated; window sums come from a small local integral image.
 */
export class NccMatcher {
  readonly half: number;
  readonly search: number;
  private readonly tpl: Float32Array; // zero-mean template
  private tplNorm = 0;
  private readonly region: number;
  private readonly sum: Float64Array;
  private readonly sum2: Float64Array;
  private readonly scores: Float32Array;
  private readonly reg: Float32Array;
  private cx = 0;
  private cy = 0;
  private ix = 0;
  private iy = 0;
  /** Result of the last match (level-0 frame pixels) and its NCC score. */
  x = 0;
  y = 0;
  score = 0;

  constructor(half = 4, search = 4) {
    this.half = half;
    this.search = search;
    const n = 2 * half + 1;
    this.tpl = new Float32Array(n * n);
    this.region = n + 2 * search;
    this.sum = new Float64Array((this.region + 1) * (this.region + 1));
    this.sum2 = new Float64Array((this.region + 1) * (this.region + 1));
    this.scores = new Float32Array((2 * search + 1) * (2 * search + 1));
    this.reg = new Float32Array(this.region * this.region);
  }

  /**
   * Warp a template from `target` for a point predicted at frame position
   * (cx, cy). `Hinv` (row-major 3x3) maps frame pixels to `target` pixels.
   * Returns false if the template leaves the target or is textureless.
   */
  buildTemplate(target: GrayImage, Hinv: ArrayLike<number>, cx: number, cy: number): boolean {
    const h = this.half, n = 2 * h + 1;
    const ix = Math.round(cx), iy = Math.round(cy);
    this.cx = cx; this.cy = cy; this.ix = ix; this.iy = iy;
    const tw = target.width, th = target.height, td = target.data;
    const h0 = Hinv[0], h1 = Hinv[1], h2 = Hinv[2], h3 = Hinv[3], h4 = Hinv[4], h5 = Hinv[5];
    const h6 = Hinv[6], h7 = Hinv[7], h8 = Hinv[8];
    const tpl = this.tpl;
    let s = 0, s2 = 0;
    for (let j = 0; j < n; j++) {
      const fy = iy - h + j;
      for (let i = 0; i < n; i++) {
        const fx = ix - h + i;
        const w = h6 * fx + h7 * fy + h8;
        const u = (h0 * fx + h1 * fy + h2) / w;
        const v = (h3 * fx + h4 * fy + h5) / w;
        if (!(u >= 0 && v >= 0 && u < tw - 1 && v < th - 1)) return false;
        const u0 = u | 0, v0 = v | 0, au = u - u0, av = v - v0;
        const k = v0 * tw + u0;
        const top = td[k] + (td[k + 1] - td[k]) * au;
        const bot = td[k + tw] + (td[k + tw + 1] - td[k + tw]) * au;
        const val = top + (bot - top) * av;
        tpl[j * n + i] = val;
        s += val;
        s2 += val * val;
      }
    }
    const cnt = n * n;
    const mean = s / cnt;
    const varSum = s2 - s * mean;
    if (varSum < cnt * 4) return false; // std < 2 grey levels: textureless
    for (let k = 0; k < cnt; k++) tpl[k] -= mean;
    this.tplNorm = Math.sqrt(varSum);
    return true;
  }

  /**
   * Search `frame` within +/-search px of the template centre. On success,
   * (x, y) is the corrected position and `score` the peak NCC.
   */
  match(frame: GrayImage, minScore = 0.8): boolean {
    const h = this.half, s = this.search, R = this.region, R1 = R + 1;
    const fw = frame.width, fh = frame.height, fd = frame.data;
    const ox = this.ix - h - s, oy = this.iy - h - s;
    if (ox < 0 || oy < 0 || ox + R > fw || oy + R > fh) return false;
    const S = this.sum, S2 = this.sum2, reg = this.reg;
    for (let j = 0; j < R; j++) {
      let rs = 0, rs2 = 0;
      const row = (oy + j) * fw + ox;
      const o = (j + 1) * R1, op = j * R1, ro = j * R;
      for (let i = 0; i < R; i++) {
        const v = fd[row + i];
        reg[ro + i] = v;
        rs += v;
        rs2 += v * v;
        S[o + i + 1] = S[op + i + 1] + rs;
        S2[o + i + 1] = S2[op + i + 1] + rs2;
      }
    }
    const sc = this.scores, sw = 2 * s + 1;
    sc.fill(NaN);
    // Coarse pass on even offsets, then hill-climb over 3x3 neighbourhoods
    // until the centre is a local maximum (its neighbours are then known).
    let bi = 0, bj = 0, best = -2;
    for (let v = 0; v < sw; v += 2) {
      for (let u = 0; u < sw; u += 2) {
        this.scoreAt(u, v);
        const sv = sc[v * sw + u];
        if (sv > best) { best = sv; bi = u; bj = v; }
      }
    }
    for (let step = 0; step < 2 * s; step++) {
      let ni = bi, nj = bj;
      for (let v = Math.max(0, bj - 1); v <= Math.min(sw - 1, bj + 1); v++) {
        for (let u = Math.max(0, bi - 1); u <= Math.min(sw - 1, bi + 1); u++) {
          if (sc[v * sw + u] !== sc[v * sw + u]) this.scoreAt(u, v);
          const sv = sc[v * sw + u];
          if (sv > best) { best = sv; ni = u; nj = v; }
        }
      }
      if (ni === bi && nj === bj) break;
      bi = ni; bj = nj;
    }
    this.score = best;
    if (best < minScore) return false;
    if (bi === 0 || bj === 0 || bi === sw - 1 || bj === sw - 1) return false; // peak not bracketed
    const c00 = sc[bj * sw + bi];
    const l = sc[bj * sw + bi - 1], r = sc[bj * sw + bi + 1];
    const t = sc[(bj - 1) * sw + bi], btm = sc[(bj + 1) * sw + bi];
    const gx = 0.5 * (r - l), gy = 0.5 * (btm - t);
    const hxx = r - 2 * c00 + l, hyy = btm - 2 * c00 + t;
    const hxy = 0.25 * (sc[(bj + 1) * sw + bi + 1] - sc[(bj - 1) * sw + bi + 1] - sc[(bj + 1) * sw + bi - 1] + sc[(bj - 1) * sw + bi - 1]);
    const det = hxx * hyy - hxy * hxy;
    let dx = 0, dy = 0;
    if (det > 1e-12 && hxx < 0) {
      dx = -(hyy * gx - hxy * gy) / det;
      dy = -(hxx * gy - hxy * gx) / det;
      if (Math.abs(dx) > 1 || Math.abs(dy) > 1) { dx = 0; dy = 0; }
    }
    // Template was sampled on the integer grid around (ix, iy), so the peak
    // offset is the correction to the predicted position.
    this.x = this.cx + (bi - s + dx);
    this.y = this.cy + (bj - s + dy);
    return true;
  }

  /** NCC at search offset (u, v), written into `scores`. */
  private scoreAt(u: number, v: number): void {
    const n = 2 * this.half + 1, R = this.region, R1 = R + 1, cnt = n * n;
    const S = this.sum, S2 = this.sum2, reg = this.reg, tpl = this.tpl;
    const a = v * R1 + u, b = a + n, c = (v + n) * R1 + u, d = c + n;
    const ws = S[d] - S[b] - S[c] + S[a];
    const ws2 = S2[d] - S2[b] - S2[c] + S2[a];
    const fv = ws2 - (ws * ws) / cnt;
    let dot = 0;
    let ri = v * R + u, ti = 0;
    for (let j = 0; j < n; j++, ri += R) {
      for (let i = 0; i < n; i++) dot += tpl[ti++] * reg[ri + i];
    }
    this.scores[v * (2 * this.search + 1) + u] = fv > 1e-6 ? dot / (this.tplNorm * Math.sqrt(fv)) : 0;
  }
}
