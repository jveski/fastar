import type { Pyramid } from '../image/pyramid';
import type { GrayImage } from '../image/gray';

export interface KltParams {
  /** Half window size; 4 gives the 9x9 window from the design. */
  halfWin: number;
  /** Number of pyramid levels used (<= pyramid depth). */
  levels: number;
  maxIter: number;
  /** Convergence threshold on the update, in pixels at the current level. */
  eps: number;
  /** Minimum eigenvalue of the normalized structure tensor at level 0. */
  minEig: number;
}

export const DEFAULT_KLT: KltParams = { halfWin: 4, levels: 3, maxIter: 10, eps: 0.02, minEig: 1e-4 };

/**
 * Pyramidal Lucas-Kanade (Bouguet). The template gradient and structure
 * tensor are computed once per level (inverse-compositional for translation),
 * and because window offsets are integers every bilinear sample in a window
 * shares the same four weights. All scratch memory is preallocated.
 */
export class Klt {
  private readonly tI: Float32Array;
  private readonly gx: Float32Array;
  private readonly gy: Float32Array;
  private readonly patch: Float32Array;
  private readonly win: number;
  private readonly pw: number;
  /** Result of the last trackPoint call (level-0 pixels). */
  x = 0;
  y = 0;

  constructor(readonly p: KltParams = DEFAULT_KLT) {
    this.win = 2 * p.halfWin + 1;
    this.pw = this.win + 2;
    this.tI = new Float32Array(this.win * this.win);
    this.gx = new Float32Array(this.win * this.win);
    this.gy = new Float32Array(this.win * this.win);
    this.patch = new Float32Array(this.pw * this.pw);
  }

  /**
   * Track one point from `prev` to `next`. (gx0, gy0) is the initial guess in
   * `next` (e.g. a gyro-predicted position). Returns false if lost.
   */
  trackPoint(prev: Pyramid, next: Pyramid, px: number, py: number, guessX: number, guessY: number): boolean {
    const { halfWin: h, maxIter, eps, minEig } = this.p;
    const L = Math.min(this.p.levels, prev.numLevels, next.numLevels);
    const win = this.win, pw = this.pw, n = win * win;
    const tI = this.tI, gxs = this.gx, gys = this.gy, patch = this.patch;
    const top = 1 / (1 << (L - 1));
    let dx = (guessX - px) * top;
    let dy = (guessY - py) * top;
    const eps2 = eps * eps;

    for (let l = L - 1; l >= 0; l--) {
      const scale = 1 / (1 << l);
      const ux = px * scale, uy = py * scale;
      const A: GrayImage = prev.levels[l];
      const B: GrayImage = next.levels[l];
      const w = A.width, hgt = A.height;
      const ad = A.data, bd = B.data;

      // Template patch (win+2)^2 around (ux, uy), constant bilinear weights.
      const x0 = Math.floor(ux), y0 = Math.floor(uy);
      if (x0 - h - 1 < 0 || y0 - h - 1 < 0 || x0 + h + 2 >= w || y0 + h + 2 >= hgt) {
        if (l === 0) return false;
        dx *= 2; dy *= 2;
        continue;
      }
      const ax = ux - x0, ay = uy - y0;
      const w00 = (1 - ax) * (1 - ay), w01 = ax * (1 - ay), w10 = (1 - ax) * ay, w11 = ax * ay;
      for (let j = 0; j < pw; j++) {
        let idx = (y0 - h - 1 + j) * w + (x0 - h - 1);
        const o = j * pw;
        for (let i = 0; i < pw; i++, idx++) {
          patch[o + i] = w00 * ad[idx] + w01 * ad[idx + 1] + w10 * ad[idx + w] + w11 * ad[idx + w + 1];
        }
      }
      // Scharr gradients on the inner window and structure tensor.
      let gxx = 0, gxy = 0, gyy = 0;
      for (let j = 0; j < win; j++) {
        const r0 = j * pw, r1 = r0 + pw, r2 = r1 + pw;
        for (let i = 0; i < win; i++) {
          const k = j * win + i;
          const ix =
            (3 * (patch[r0 + i + 2] - patch[r0 + i]) + 10 * (patch[r1 + i + 2] - patch[r1 + i]) +
              3 * (patch[r2 + i + 2] - patch[r2 + i])) * (1 / 32);
          const iy =
            (3 * (patch[r2 + i] - patch[r0 + i]) + 10 * (patch[r2 + i + 1] - patch[r0 + i + 1]) +
              3 * (patch[r2 + i + 2] - patch[r0 + i + 2])) * (1 / 32);
          tI[k] = patch[r1 + i + 1];
          gxs[k] = ix;
          gys[k] = iy;
          gxx += ix * ix; gxy += ix * iy; gyy += iy * iy;
        }
      }
      const det = gxx * gyy - gxy * gxy;
      if (l === 0) {
        const tr = (gxx + gyy) * 0.5;
        const minE = (tr - Math.sqrt(Math.max(0, tr * tr - det))) / (n * 255 * 255);
        if (minE < minEig) return false;
      }
      if (det < 1e-6) {
        if (l === 0) return false;
        dx *= 2; dy *= 2;
        continue;
      }
      const idet = 1 / det;

      for (let it = 0; it < maxIter; it++) {
        const cx = ux + dx, cy = uy + dy;
        const bx0 = Math.floor(cx), by0 = Math.floor(cy);
        if (bx0 - h < 0 || by0 - h < 0 || bx0 + h + 1 >= w || by0 + h + 1 >= hgt) {
          if (l === 0) return false;
          break;
        }
        const bx = cx - bx0, by = cy - by0;
        const v00 = (1 - bx) * (1 - by), v01 = bx * (1 - by), v10 = (1 - bx) * by, v11 = bx * by;
        let bxs = 0, bys = 0;
        for (let j = 0; j < win; j++) {
          let idx = (by0 - h + j) * w + (bx0 - h);
          const o = j * win;
          for (let i = 0; i < win; i++, idx++) {
            const J = v00 * bd[idx] + v01 * bd[idx + 1] + v10 * bd[idx + w] + v11 * bd[idx + w + 1];
            const diff = tI[o + i] - J;
            bxs += diff * gxs[o + i];
            bys += diff * gys[o + i];
          }
        }
        const ex = (gyy * bxs - gxy * bys) * idet;
        const ey = (gxx * bys - gxy * bxs) * idet;
        dx += ex;
        dy += ey;
        if (ex * ex + ey * ey < eps2) break;
      }
      if (l > 0) { dx *= 2; dy *= 2; }
    }
    this.x = px + dx;
    this.y = py + dy;
    return Number.isFinite(this.x) && Number.isFinite(this.y);
  }

  /**
   * Track a batch with a forward-backward check. `pts` holds [x, y] in prev;
   * `out` holds the initial guesses on input and the results on output.
   * status[i] is set to 1 on success.
   */
  trackBatch(
    prev: Pyramid,
    next: Pyramid,
    pts: Float32Array,
    out: Float32Array,
    count: number,
    status: Uint8Array,
    fbThreshold = 0.5,
  ): number {
    let ok = 0;
    const fb2 = fbThreshold * fbThreshold;
    for (let i = 0; i < count; i++) {
      const px = pts[2 * i], py = pts[2 * i + 1];
      if (!this.trackPoint(prev, next, px, py, out[2 * i], out[2 * i + 1])) {
        status[i] = 0;
        continue;
      }
      const nx = this.x, ny = this.y;
      if (!this.trackPoint(next, prev, nx, ny, px, py)) {
        status[i] = 0;
        continue;
      }
      const ex = this.x - px, ey = this.y - py;
      if (ex * ex + ey * ey > fb2) {
        status[i] = 0;
        continue;
      }
      out[2 * i] = nx;
      out[2 * i + 1] = ny;
      status[i] = 1;
      ok++;
    }
    return ok;
  }
}
