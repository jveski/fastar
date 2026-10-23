import { describe, expect, it } from 'vitest';
import { deflateSync } from 'node:zlib';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { decodePng } from '../cli/png';
import { getPoster } from './fixtures';
import { decodeTargets } from '../src/target/format';

function crc32(b: Uint8Array): number {
  let c = ~0;
  for (const x of b) {
    c ^= x;
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c >>> 0;
}

/** Encode RGB(A)/grey 8-bit PNG cycling through all five filter types. */
function encodePng(w: number, h: number, channels: number, px: Uint8Array): Uint8Array {
  const stride = w * channels;
  const raw = new Uint8Array((stride + 1) * h);
  for (let y = 0; y < h; y++) {
    const f = y % 5;
    raw[y * (stride + 1)] = f;
    for (let i = 0; i < stride; i++) {
      const x = px[y * stride + i];
      const a = i >= channels ? px[y * stride + i - channels] : 0;
      const b = y > 0 ? px[(y - 1) * stride + i] : 0;
      const c = i >= channels && y > 0 ? px[(y - 1) * stride + i - channels] : 0;
      const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
      const pred = [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][f];
      raw[y * (stride + 1) + 1 + i] = (x - pred) & 255;
    }
  }
  const chunk = (type: string, data: Uint8Array) => {
    const out = new Uint8Array(12 + data.length);
    const dv = new DataView(out.buffer);
    dv.setUint32(0, data.length);
    for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
    out.set(data, 8);
    dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
    return out;
  };
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, w); dv.setUint32(4, h);
  ihdr[8] = 8; ihdr[9] = channels === 1 ? 0 : channels === 3 ? 2 : 6;
  const parts = [Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', new Uint8Array(0))];
  const total = parts.reduce((s, p) => s + p.length, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

describe('PNG decoder', () => {
  it('decodes greyscale with every filter type exactly', () => {
    const img = getPoster();
    const dec = decodePng(encodePng(img.width, img.height, 1, img.data));
    expect(dec.width).toBe(img.width);
    expect(dec.data).toEqual(img.data);
  });

  it('decodes RGB to luma', () => {
    const w = 7, h = 5, px = new Uint8Array(w * h * 3);
    for (let i = 0; i < px.length; i++) px[i] = (i * 37) & 255;
    const dec = decodePng(encodePng(w, h, 3, px));
    for (let i = 0; i < w * h; i++) {
      expect(dec.data[i]).toBe((px[3 * i] * 77 + px[3 * i + 1] * 150 + px[3 * i + 2] * 29 + 128) >> 8);
    }
  });
});

describe('fastar-compile CLI', () => {
  it('compiles a PNG into a loadable targets file', () => {
    const dir = mkdtempSync(join(tmpdir(), 'fastar-'));
    const img = getPoster();
    writeFileSync(join(dir, 'poster.png'), encodePng(img.width, img.height, 1, img.data));
    const out = execFileSync('npx', ['tsx', 'cli/compile.ts', '-o', join(dir, 't.bin'), '--report', join(dir, 'r.json'), join(dir, 'poster.png:0.2')], {
      encoding: 'utf8',
    });
    expect(out).toMatch(/poster: 640x480px, 0.2 m wide/);
    const buf = readFileSync(join(dir, 't.bin'));
    const [t] = decodeTargets(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    expect(t.meta.name).toBe('poster');
    expect(t.meta.physicalWidth).toBe(0.2);
    expect(JSON.parse(readFileSync(join(dir, 'r.json'), 'utf8'))[0].features).toBeGreaterThan(1000);
  });
});
