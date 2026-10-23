// Minimal PNG decoder (non-interlaced, 8/16-bit, all colour types) to greyscale.
import { inflateSync } from 'node:zlib';
import type { GrayImage } from '../src/image/gray';

export function decodePng(buf: Uint8Array): GrayImage {
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  for (let i = 0; i < 8; i++) if (buf[i] !== sig[i]) throw new Error('not a PNG file');
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let width = 0, height = 0, depth = 0, ctype = 0, interlace = 0;
  let palette: Uint8Array | null = null;
  const idat: Uint8Array[] = [];
  for (let o = 8; o < buf.length;) {
    const len = dv.getUint32(o);
    const type = String.fromCharCode(buf[o + 4], buf[o + 5], buf[o + 6], buf[o + 7]);
    const data = buf.subarray(o + 8, o + 8 + len);
    if (type === 'IHDR') {
      width = dv.getUint32(o + 8); height = dv.getUint32(o + 12);
      depth = data[8]; ctype = data[9]; interlace = data[12];
    } else if (type === 'PLTE') palette = data;
    else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    o += 12 + len;
  }
  if (interlace) throw new Error('interlaced PNGs are not supported; re-save without interlacing');
  const channels = ({ 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 } as Record<number, number>)[ctype];
  if (!channels) throw new Error(`unsupported PNG colour type ${ctype}`);
  if (depth !== 8 && depth !== 16 && !(depth < 8 && (ctype === 0 || ctype === 3))) throw new Error(`unsupported bit depth ${depth}`);
  const total = idat.reduce((s, d) => s + d.length, 0);
  const z = new Uint8Array(total);
  let p = 0;
  for (const d of idat) { z.set(d, p); p += d.length; }
  const raw = inflateSync(z);
  const bitsPP = channels * depth;
  const bpp = Math.max(1, bitsPP >> 3);
  const stride = Math.ceil((width * bitsPP) / 8);
  const cur = new Uint8Array(stride), prev = new Uint8Array(stride);
  const out = new Uint8Array(width * height);
  let r = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[r++];
    for (let i = 0; i < stride; i++) {
      const x = raw[r++];
      const a = i >= bpp ? cur[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
      let v: number;
      switch (filter) {
        case 0: v = x; break;
        case 1: v = x + a; break;
        case 2: v = x + b; break;
        case 3: v = x + ((a + b) >> 1); break;
        case 4: {
          const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c);
          v = x + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
          break;
        }
        default: throw new Error(`bad PNG filter ${filter}`);
      }
      cur[i] = v & 255;
    }
    for (let x = 0; x < width; x++) {
      let R: number, G: number, B: number;
      const sample = (ch: number): number => {
        if (depth === 16) return cur[(x * channels + ch) * 2];
        if (depth === 8) return cur[x * channels + ch];
        const bit = x * depth, byte = cur[bit >> 3], shift = 8 - depth - (bit & 7);
        return (byte >> shift) & ((1 << depth) - 1);
      };
      if (ctype === 3) {
        const idx = sample(0) * 3;
        R = palette![idx]; G = palette![idx + 1]; B = palette![idx + 2];
      } else if (ctype === 0 || ctype === 4) {
        const v = sample(0);
        R = G = B = depth < 8 ? Math.round((v * 255) / ((1 << depth) - 1)) : v;
      } else {
        R = sample(0); G = sample(1); B = sample(2);
      }
      let lum = (R * 77 + G * 150 + B * 29 + 128) >> 8;
      if (ctype === 4 || ctype === 6) {
        // Composite alpha over white.
        const A = sample(ctype === 4 ? 1 : 3);
        lum = Math.round((lum * A + 255 * (255 - A)) / 255);
      }
      out[y * width + x] = lum;
    }
    prev.set(cur);
  }
  return { width, height, data: out };
}
