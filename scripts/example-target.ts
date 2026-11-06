// Generates the demo target for examples/basic: a printable PNG and its
// compiled targets.bin.  npx tsx scripts/example-target.ts
import { writeFileSync, mkdirSync } from 'node:fs';
import { makePoster } from '../src/sim/poster';
import { compileTarget } from '../src/target/compiler';
import { encodeTargets } from '../src/target/format';
import { encodeGrayPng } from '../cli/png';

const OUT = 'examples/basic';
const WIDTH_M = 0.2; // print it 20 cm wide (or measure it on your screen and change this)

mkdirSync(OUT, { recursive: true });
const poster = makePoster(1024, 768, 2024);
writeFileSync(`${OUT}/target.png`, encodeGrayPng(poster));
const target = compileTarget(poster, { physicalWidth: WIDTH_M, name: 'demo' });
writeFileSync(`${OUT}/targets.bin`, new Uint8Array(encodeTargets([target])));
const q = target.meta.quality;
console.log(`wrote ${OUT}/target.png and targets.bin: ${q.features} features, ${(q.coverage * 100).toFixed(0)}% coverage${q.warnings.length ? `, warnings: ${q.warnings.join('; ')}` : ''}`);
