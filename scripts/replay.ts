// Replay a recorded session through the pipeline and print a summary:
//   npx tsx scripts/replay.ts <session.json> <session.luma[.gz]> <targets.bin> [options]
// Options: sm (pass poses through the render-side smoother, as the page draws
// them), devtd (use the recording's calibrated t_d instead of re-estimating
// it), v (per-frame log).
// Jumps are the second difference of the displayed target position (mm; depth
// weighted 0.3, since it only scales the overlay).
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { replaySession } from '../src/sim/replay';
import { decodeTargets } from '../src/target/format';
import type { SessionLog } from '../src/shell/recorder';

const [jsonPath, lumaPath, targetsPath] = process.argv.slice(2);
if (!jsonPath || !lumaPath || !targetsPath) {
  console.error('usage: tsx scripts/replay.ts <session.json> <session.luma[.gz]> <targets.bin> [sm] [devtd] [v]');
  process.exit(1);
}
const flag = (k: string) => process.argv.slice(5).includes(k);
const log = JSON.parse(readFileSync(jsonPath, 'utf8')) as SessionLog;
let luma: Uint8Array = readFileSync(lumaPath);
if (lumaPath.endsWith('.gz')) luma = gunzipSync(luma);
const tb = readFileSync(targetsPath);
const targets = decodeTargets(tb.buffer.slice(tb.byteOffset, tb.byteOffset + tb.byteLength));
const s = replaySession(log, luma, targets, flag('devtd') ? { calibrateTd: false } : {}, flag('sm') ? {} : null);

const f = s.frames;
const pct = (xs: number[], p: number) => {
  const a = xs.filter(Number.isFinite).sort((x, y) => x - y);
  return a.length ? a[Math.min(a.length - 1, Math.floor(p * a.length))] : NaN;
};
const jumps: number[] = [];
const rows: string[] = [];
for (let i = 2; i < f.length; i++) {
  const a = f[i].tvec, b = f[i - 1].tvec, c = f[i - 2].tvec;
  const d = a && b && c ? Math.hypot(a[0] - 2 * b[0] + c[0], a[1] - 2 * b[1] + c[1], 0.3 * (a[2] - 2 * b[2] + c[2])) * 1000 : NaN;
  jumps.push(d);
  rows.push(
    `${(f[i].t - f[0].t).toFixed(0).padStart(6)} ${f[i].status.padEnd(8)} inl=${String(f[i].inliers).padStart(3)} ` +
      `nis=${f[i].nis.toFixed(1).padStart(6)} jump=${d.toFixed(2).padStart(6)}mm z=${a ? (a[2] * 1000).toFixed(0) : '-'}${f[i].acquired ? ' ACQ' : ''}`,
  );
}
const vis = f.filter((x) => x.visible).length;
console.log(`frames ${f.length}, visible ${vis} (${((vis / f.length) * 100).toFixed(1)}%)`);
console.log(`track losses ${s.trackLosses}, acquisitions ${s.acquisitions}`);
console.log(`still jitter ${s.stillJitterMm.toFixed(3)} mm`);
console.log(`jumps mm: p50 ${pct(jumps, 0.5).toFixed(2)} p90 ${pct(jumps, 0.9).toFixed(2)} p99 ${pct(jumps, 0.99).toFixed(2)} max ${pct(jumps, 1).toFixed(2)}`);
console.log(`frame ms median ${s.medianFrameMs.toFixed(2)}, p95 ${s.p95FrameMs.toFixed(2)}`);
console.log(`t_d ${(s.td * 1000).toFixed(1)} ms${s.tdCalibrated ? ' (calibrated)' : ''}`);
if (flag('v')) console.log(rows.join('\n'));
