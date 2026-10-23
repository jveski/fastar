import type { BenchCase } from '../harness';
import { intrinsicsFromFov } from '../../src/vision/camera';
import { makePoster } from '../../src/sim/poster';
import { createSceneRenderer, scenarios } from '../../src/sim/scene';
import { targetFor } from '../../src/sim/harness';
import { Acquirer } from '../../src/track/acquire';
import { Pipeline } from '../../src/track/pipeline';
import { Eskf } from '../../src/fusion/eskf';
import { ImuBuffer } from '../../src/fusion/imu';
import { simulateImu, relativePose } from '../../src/sim/scene';

export function trackingCases(): BenchCase[] {
  const K = intrinsicsFromFov(640, 480, 63);
  const poster = makePoster(640, 480, 11);
  const target = targetFor(poster, 0.2);
  const scn = scenarios.slowSweep();
  const r = createSceneRenderer(scn, { K, poster, posterWidthM: 0.2 });
  // Ping-pong sequence so consecutive frames are always adjacent in time.
  const fwd = Array.from({ length: 31 }, (_, k) => r.render(0.5 + k / 30, scn.camera, { noise: 2 }).data);
  const seq = [...fwd, ...fwd.slice(1, -1).reverse()];
  return [
    {
      name: 'tracking path per frame (pyramid+KLT+NCC+pose+ESKF)',
      budgetMs: 10 / 3,
      iterations: 300,
      warmup: 60,
      setup: () => {
        const p = new Pipeline([target], K, { width: 640, height: 480 });
        const acq = new Acquirer([target], 640, 480, K);
        let k = 0, t = 0;
        const res = p.processFrame(seq[0], 0);
        const a = acq.acquire(seq[0]);
        if (!a || p.applyAcquisition(a, res.frameId) === null) throw new Error('no lock');
        return () => {
          k = (k + 1) % seq.length;
          t += 1 / 30;
          const fr = p.processFrame(seq[k], t);
          if (fr.targets[0]?.status !== 'tracking') throw new Error('lost track in benchmark');
        };
      },
    },
    {
      name: 'ESKF update + re-propagation (100 ms delay)',
      budgetMs: 0.3 / 3,
      iterations: 300,
      setup: () => {
        const buf = new ImuBuffer(512);
        const imu = simulateImu(scn, {});
        for (const s of imu) buf.push(s.t, s.gyro[0], s.gyro[1], s.gyro[2], s.accel[0], s.accel[1], s.accel[2]);
        const f = new Eskf({ useImu: true, estimateTd: true });
        const C = new Float64Array(36);
        for (let i = 0; i < 6; i++) C[i * 7] = i < 3 ? 4e-6 : 2.5e-7;
        let tc = 0.5;
        const g = relativePose(scn.camera(tc), scn.target(tc));
        f.initialize(tc, g.R, g.t, C, null);
        return () => {
          tc += 1 / 30;
          if (tc > 3.5) {
            tc = 0.5;
            f.initialize(tc, g.R, g.t, C, null);
          }
          f.propagateTo(tc + 0.1, buf);
          const m = relativePose(scn.camera(tc), scn.target(tc));
          f.updatePose(tc, m.R, m.t, C, buf);
        };
      },
    },
  ];
}
