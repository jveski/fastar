// Generates the demo target for examples/basic: a printable PNG and its
// compiled targets.bin.  npx tsx scripts/example-target.ts
import { writeFileSync, mkdirSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import { makePoster } from '../src/sim/poster';
import { compileTarget } from '../src/target/compiler';
import { encodeTargets } from '../src/target/format';
import type { GrayImage } from '../src/image/gray';

const OUT = 'examples/basic';
const WIDTH_M = 0.2; // print it 20 cm wide (or measure it on your screen and change this)

function crc32(b: Uint8Array): number {
  let c = ~0;
  for (const x of b) {
    c ^= x;
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c >>> 0;
}

function encodeGrayPng(img: GrayImage): Uint8Array {
  const raw = new Uint8Array((img.width + 1) * img.height);
  for (let y = 0; y < img.height; y++) raw.set(img.data.subarray(y * img.width, (y + 1) * img.width), y * (img.width + 1) + 1);
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
  new DataView(ihdr.buffer).setUint32(0, img.width);
  new DataView(ihdr.buffer).setUint32(4, img.height);
  ihdr[8] = 8; // 8-bit greyscale
  const parts = [Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

mkdirSync(OUT, { recursive: true });
const poster = makePoster(1024, 768, 2024);
writeFileSync(`${OUT}/target.png`, encodeGrayPng(poster));
const target = compileTarget(poster, { physicalWidth: WIDTH_M, name: 'demo' });
writeFileSync(`${OUT}/targets.bin`, new Uint8Array(encodeTargets([target])));
const q = target.meta.quality;
console.log(`wrote ${OUT}/target.png and targets.bin: ${q.features} features, ${(q.coverage * 100).toFixed(0)}% coverage${q.warnings.length ? `, warnings: ${q.warnings.join('; ')}` : ''}`);
