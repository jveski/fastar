// Generates the bending-paper demo target for examples/paper: a printable PNG
// with detail over the whole page (every part of a bending target must be
// tracked to be shaped) and its compiled targets.bin.
//   npx tsx scripts/example-paper.ts
import { writeFileSync, mkdirSync } from 'node:fs';
import { makePoster } from '../src/sim/poster';
import { compileTarget } from '../src/target/compiler';
import { encodeTargets } from '../src/target/format';
import { encodeGrayPng } from '../cli/png';
import { createGray } from '../src/image/gray';

const OUT = 'examples/paper';
const WIDTH_M = 0.25; // print it 25 cm wide: fits landscape A4 or US Letter with margins

mkdirSync(OUT, { recursive: true });
// A4's aspect ratio (sqrt 2), tiled from small posters so no flat area
// spans a mesh cell (~3 cm).
const W = 1024, H = 724, TX = 4, TY = 3;
const poster = createGray(W, H);
for (let ty = 0; ty < TY; ty++) {
  for (let tx = 0; tx < TX; tx++) {
    const x0 = Math.round((tx * W) / TX), x1 = Math.round(((tx + 1) * W) / TX);
    const y0 = Math.round((ty * H) / TY), y1 = Math.round(((ty + 1) * H) / TY);
    const tile = makePoster(x1 - x0, y1 - y0, 4242 + 17 * (ty * TX + tx), 6, 260);
    for (let y = y0; y < y1; y++) poster.data.set(tile.data.subarray((y - y0) * tile.width, (y - y0 + 1) * tile.width), y * W + x0);
  }
}
writeFileSync(`${OUT}/target.png`, encodeGrayPng(poster));
const target = compileTarget(poster, { physicalWidth: WIDTH_M, name: 'paper' });
writeFileSync(`${OUT}/targets.bin`, new Uint8Array(encodeTargets([target])));
const q = target.meta.quality;
console.log(`wrote ${OUT}/target.png and targets.bin: ${q.features} features, ${(q.coverage * 100).toFixed(0)}% coverage${q.warnings.length ? `, warnings: ${q.warnings.join('; ')}` : ''}`);
