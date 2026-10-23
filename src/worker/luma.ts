// Luma extraction for the worker frame path.

/**
 * Downscale an 8-bit plane (stride in bytes) to dw x dh. Exact 2x uses a
 * 2x2 box; other ratios use area averaging over the covered source pixels.
 */
export function downscaleLuma(
  src: Uint8Array, sw: number, sh: number, stride: number, dst: Uint8Array, dw: number, dh: number,
): Uint8Array {
  if (sw === dw && sh === dh) {
    for (let y = 0; y < dh; y++) dst.set(src.subarray(y * stride, y * stride + dw), y * dw);
    return dst;
  }
  if (sw === 2 * dw && sh === 2 * dh) {
    for (let y = 0; y < dh; y++) {
      const r0 = 2 * y * stride, r1 = r0 + stride, o = y * dw;
      for (let x = 0, sx = 0; x < dw; x++, sx += 2) {
        dst[o + x] = (src[r0 + sx] + src[r0 + sx + 1] + src[r1 + sx] + src[r1 + sx + 1] + 2) >> 2;
      }
    }
    return dst;
  }
  const fx = sw / dw, fy = sh / dh;
  for (let y = 0; y < dh; y++) {
    const y0 = Math.floor(y * fy), y1 = Math.max(y0 + 1, Math.min(sh, Math.floor((y + 1) * fy)));
    for (let x = 0; x < dw; x++) {
      const x0 = Math.floor(x * fx), x1 = Math.max(x0 + 1, Math.min(sw, Math.floor((x + 1) * fx)));
      let acc = 0;
      for (let yy = y0; yy < y1; yy++) {
        const r = yy * stride;
        for (let xx = x0; xx < x1; xx++) acc += src[r + xx];
      }
      dst[y * dw + x] = (acc / ((y1 - y0) * (x1 - x0)) + 0.5) | 0;
    }
  }
  return dst;
}

let scratch: Uint8Array | null = null;
let rgbaTmp: Uint8Array | null = null;

/**
 * Copy a VideoFrame's luma into `out` (w x h). YUV formats copy only what is
 * needed from the Y plane; RGB formats are converted with BT.601 weights.
 */
export async function videoFrameToLuma(frame: VideoFrame, w: number, h: number, out: Uint8Array): Promise<Uint8Array | null> {
  const fmt = frame.format;
  const vw = frame.visibleRect?.width ?? frame.displayWidth, vh = frame.visibleRect?.height ?? frame.displayHeight;
  if (!fmt) return null;
  const size = frame.allocationSize();
  if (!scratch || scratch.length < size) scratch = new Uint8Array(size);
  const layout = await frame.copyTo(scratch);
  if (fmt.startsWith('I4') || fmt.startsWith('NV12')) {
    const y = layout[0];
    return downscaleLuma(scratch.subarray(y.offset), vw, vh, y.stride, out, w, h);
  }
  // RGBA / RGBX / BGRA / BGRX
  const stride = layout[0].stride, off = layout[0].offset;
  const bgr = fmt.startsWith('BGR');
  if (!rgbaTmp || rgbaTmp.length < vw * vh) rgbaTmp = new Uint8Array(vw * vh);
  const g = rgbaTmp;
  const ri = bgr ? 2 : 0, bi = bgr ? 0 : 2;
  for (let yy = 0; yy < vh; yy++) {
    let j = off + yy * stride;
    const o = yy * vw;
    for (let xx = 0; xx < vw; xx++, j += 4) g[o + xx] = (scratch[j + ri] * 77 + scratch[j + 1] * 150 + scratch[j + bi] * 29 + 128) >> 8;
  }
  return downscaleLuma(g, vw, vh, vw, out, w, h);
}
