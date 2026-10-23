// Ad-hoc scenario runner: npx tsx scripts/sim.ts <scenario> [imu]
import { intrinsicsFromFov } from '../src/vision/camera';
import { makePoster } from '../src/sim/poster';
import { scenarios } from '../src/sim/scene';
import { runScenario, jitter, percentile } from '../src/sim/harness';

const K = intrinsicsFromFov(640, 480, 63);
const poster = makePoster(640, 480, 11);
const name = (process.argv[2] ?? 'static') as keyof typeof scenarios;
const useImu = process.argv.includes('imu');
const scn = (scenarios[name] as () => any)();
const r = runScenario(scn, { K, poster, posterWidthM: 0.2, useImu, td: useImu ? Number((process.argv.find((a) => a.startsWith('td=')) ?? 'td=0.03').slice(3)) : 0, imu: { gyroBias: process.argv.includes('nobias') ? [0, 0, 0] : [0.01, -0.02, 0.015] },
  pipeline: { eskf: { ...(process.argv.includes('nograv') ? { gravityNoise: 1e6 } : {}), ...(process.argv.includes('notd') ? { estimateTd: false } : {}) } } });
const vis = r.frames.filter((f) => f.visible);
console.log(`${scn.name} imu=${useImu}: frames=${r.frames.length} visible=${vis.length} acq=${r.acquisitions}/${r.acquireAttempts} firstLock=${r.firstLockTime.toFixed(3)}s`);
const ov = vis.map((f) => f.overlayErrPx);
console.log(`overlay px: median=${percentile(ov, 0.5).toFixed(3)} p95=${percentile(ov, 0.95).toFixed(3)} max=${percentile(ov, 1).toFixed(3)}`);
console.log(`trans err mm: median=${(percentile(vis.map((f) => f.transErr), 0.5) * 1000).toFixed(2)}  rot err deg median=${percentile(vis.map((f) => f.rotErrDeg), 0.5).toFixed(3)}`);
const j = jitter(r.frames.filter((f) => f.t > 0.5));
console.log(`jitter: trans=${(j.transStd * 1000).toFixed(3)} mm rot=${j.rotStdDeg.toFixed(4)} deg`);
console.log(`track ms: median=${percentile(r.frames.map((f) => f.trackMs), 0.5).toFixed(2)} p95=${percentile(r.frames.map((f) => f.trackMs), 0.95).toFixed(2)}`);
if (process.argv.includes('v')) for (const f of r.frames) console.log(f.t.toFixed(3), f.status, f.visible ? 'V' : '-', f.inliers, f.overlayErrPx.toFixed(2), (f.transErr*1000).toFixed(2), f.rotErrDeg.toFixed(3), f.acquired ? 'ACQ' : '');
