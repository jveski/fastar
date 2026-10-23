// 8-bit greyscale images, row-major, stride === width.

export interface GrayImage {
  width: number;
  height: number;
  data: Uint8Array;
}

export function createGray(width: number, height: number, data?: Uint8Array): GrayImage {
  return { width, height, data: data ?? new Uint8Array(width * height) };
}

/** RGBA (or RGBX/BGRA with `bgr`) to luma with BT.601 integer weights. */
export function rgbaToGray(
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  out?: GrayImage,
  bgr = false,
): GrayImage {
  const img = out ?? createGray(width, height);
  const d = img.data;
  const n = width * height;
  const ri = bgr ? 2 : 0, bi = bgr ? 0 : 2;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    d[i] = (rgba[j + ri] * 77 + rgba[j + 1] * 150 + rgba[j + bi] * 29 + 128) >> 8;
  }
  return img;
}

/**
 * Area-average resample to (dw, dh). Works for any scale factor; exact box
 * filtering for downscales and bilinear-like for upscales.
 */
export function resizeArea(src: GrayImage, dw: number, dh: number, out?: GrayImage): GrayImage {
  const dst = out ?? createGray(dw, dh);
  const sw = src.width, sh = src.height, s = src.data, d = dst.data;
  const fx = sw / dw, fy = sh / dh;
  if (fx <= 1 || fy <= 1) {
    // Upscale (or mixed): bilinear.
    for (let y = 0; y < dh; y++) {
      const syf = Math.min(sh - 1, Math.max(0, (y + 0.5) * fy - 0.5));
      const y0 = Math.min(sh - 2, Math.floor(syf)), ay = syf - y0;
      for (let x = 0; x < dw; x++) {
        const sxf = Math.min(sw - 1, Math.max(0, (x + 0.5) * fx - 0.5));
        const x0 = Math.min(sw - 2, Math.floor(sxf)), ax = sxf - x0;
        const i = y0 * sw + x0;
        const top = s[i] + (s[i + 1] - s[i]) * ax;
        const bot = s[i + sw] + (s[i + sw + 1] - s[i + sw]) * ax;
        d[y * dw + x] = (top + (bot - top) * ay + 0.5) | 0;
      }
    }
    return dst;
  }
  // Fast path: exact 2x.
  if (fx === 2 && fy === 2) {
    for (let y = 0; y < dh; y++) {
      const r0 = 2 * y * sw, r1 = r0 + sw;
      for (let x = 0, sx = 0; x < dw; x++, sx += 2) {
        d[y * dw + x] = (s[r0 + sx] + s[r0 + sx + 1] + s[r1 + sx] + s[r1 + sx + 1] + 2) >> 2;
      }
    }
    return dst;
  }
  const row = new Float32Array(dw);
  for (let y = 0; y < dh; y++) {
    row.fill(0);
    const y0 = y * fy, y1 = y0 + fy;
    for (let sy = Math.floor(y0); sy < Math.min(sh, Math.ceil(y1)); sy++) {
      const wy = Math.min(y1, sy + 1) - Math.max(y0, sy);
      const base = sy * sw;
      for (let x = 0; x < dw; x++) {
        const x0 = x * fx, x1 = x0 + fx;
        let acc = 0;
        for (let sx = Math.floor(x0); sx < Math.min(sw, Math.ceil(x1)); sx++) {
          acc += s[base + sx] * (Math.min(x1, sx + 1) - Math.max(x0, sx));
        }
        row[x] += acc * wy;
      }
    }
    const inv = 1 / (fx * fy);
    for (let x = 0; x < dw; x++) d[y * dw + x] = Math.min(255, (row[x] * inv + 0.5) | 0);
  }
  return dst;
}

/**
 * Separable [1 4 6 4 1]/16 blur with clamped borders. `tmp` must hold
 * width*height Uint16 values; pass it to avoid allocation.
 */
export function blur5(src: GrayImage, out: GrayImage, tmp?: Uint16Array): GrayImage {
  const w = src.width, h = src.height, s = src.data, d = out.data;
  const t = tmp ?? new Uint16Array(w * h);
  if ((w & 3) === 0 && (s.byteOffset & 3) === 0 && (d.byteOffset & 3) === 0 && (t.byteOffset & 3) === 0 && h >= 1) {
    blur5Swar(s, d, t, w, h);
    return out;
  }
  for (let y = 0; y < h; y++) {
    const r = y * w;
    for (let x = 0; x < w; x++) {
      const xm2 = x < 2 ? 0 : x - 2, xm1 = x < 1 ? 0 : x - 1;
      const xp1 = x + 1 >= w ? w - 1 : x + 1, xp2 = x + 2 >= w ? w - 1 : x + 2;
      t[r + x] = s[r + xm2] + 4 * s[r + xm1] + 6 * s[r + x] + 4 * s[r + xp1] + s[r + xp2];
    }
  }
  for (let y = 0; y < h; y++) {
    const ym2 = (y < 2 ? 0 : y - 2) * w, ym1 = (y < 1 ? 0 : y - 1) * w, y0 = y * w;
    const yp1 = (y + 1 >= h ? h - 1 : y + 1) * w, yp2 = (y + 2 >= h ? h - 1 : y + 2) * w;
    for (let x = 0; x < w; x++) {
      d[y0 + x] = (t[ym2 + x] + 4 * t[ym1 + x] + 6 * t[y0 + x] + 4 * t[yp1 + x] + t[yp2 + x] + 128) >> 8;
    }
  }
  return out;
}

const views = new WeakMap<ArrayBufferLike, Int32Array>();
/** Cached Int32Array view over a whole buffer (avoids per-call allocation). */
export function i32View(buf: ArrayBufferLike): Int32Array {
  let v = views.get(buf);
  if (!v) {
    v = new Int32Array(buf, 0, buf.byteLength >> 2);
    views.set(buf, v);
  }
  return v;
}

/**
 * SWAR [1 4 6 4 1] blur for widths divisible by 4. Each 32-bit load yields an
 * even-position pair and an odd-position pair of 16-bit lane sums; the
 * vertical pass works on those pairs and packs four bytes per store.
 */
function blur5Swar(s: Uint8Array, d: Uint8Array, t: Uint16Array, w: number, h: number): void {
  const src = i32View(s.buffer), dst = i32View(d.buffer), tt = i32View(t.buffer);
  const sb = s.byteOffset >> 2, db = d.byteOffset >> 2, tb = t.byteOffset >> 2;
  const words = w >> 2, half = w >> 1;
  for (let y = 0; y < h; y++) {
    let k = sb + y * words;
    let o = tb + y * half;
    let cur = src[k];
    const b0 = cur & 0xff;
    let prev = (b0 << 16) | (b0 << 24);
    for (let m = 0; m < words; m++, k++, o += 2) {
      const nxt = m + 1 < words ? src[k + 1] : Math.imul(cur >>> 24, 0x01010101);
      const A = ((prev >>> 16) & 0xff) | ((cur & 0xff) << 16);
      const B = (prev >>> 24) | (((cur >>> 8) & 0xff) << 16);
      const C = cur & 0x00ff00ff;
      const D = (cur >>> 8) & 0x00ff00ff;
      const E = ((cur >>> 16) & 0xff) | ((nxt & 0xff) << 16);
      const F = (cur >>> 24) | (((nxt >>> 8) & 0xff) << 16);
      tt[o] = A + 4 * (B + D) + 6 * C + E;
      tt[o + 1] = B + 4 * (C + E) + 6 * D + F;
      prev = cur;
      cur = nxt;
    }
  }
  const ymax = h - 1;
  for (let y = 0; y < h; y++) {
    const r0 = tb + (y < 2 ? 0 : y - 2) * half, r1 = tb + (y < 1 ? 0 : y - 1) * half, r2 = tb + y * half;
    const r3 = tb + (y + 1 > ymax ? ymax : y + 1) * half, r4 = tb + (y + 2 > ymax ? ymax : y + 2) * half;
    let o = db + y * words;
    for (let k = 0; k < half; k += 2, o++) {
      const ve = (tt[r0 + k] + ((tt[r1 + k] + tt[r3 + k]) << 2) + 6 * tt[r2 + k] + tt[r4 + k] + 0x00800080) | 0;
      const vo = (tt[r0 + k + 1] + ((tt[r1 + k + 1] + tt[r3 + k + 1]) << 2) + 6 * tt[r2 + k + 1] + tt[r4 + k + 1] + 0x00800080) | 0;
      dst[o] = ((ve >>> 8) & 0xff) | (((vo >>> 8) & 0xff) << 8) | ((ve >>> 24) << 16) | ((vo >>> 24) << 24);
    }
  }
}
