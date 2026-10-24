import { runBench, printResults, type BenchCase } from './harness';
import { mathCases } from './cases/math';
import { visionCases } from './cases/vision';
import { acquireCases } from './cases/acquire';
import { trackingCases } from './cases/tracking';
import { videoCases } from './cases/video';

const filter = process.argv[2];
const all: BenchCase[] = [...mathCases(), ...visionCases(), ...acquireCases(), ...trackingCases(), ...videoCases()];
const cases = all.filter((c) => !filter || c.name.includes(filter));

const results = [];
for (const c of cases) results.push(await runBench(c));
const ok = printResults(results);
if (!ok) {
  console.error('One or more stages exceeded their budget.');
  process.exit(1);
}
