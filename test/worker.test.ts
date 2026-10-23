import { describe, expect, it } from 'vitest';
import { TrackerCore } from '../src/worker/tracker-core';
import { AcquireCore } from '../src/worker/acquire-core';
import type { FromTracker, ResultMessage } from '../src/worker/protocol';
import { encodeTargets } from '../src/target/format';
import { targetFor } from '../src/sim/harness';
import { createSceneRenderer, scenarios, simulateImu, relativePose } from '../src/sim/scene';
import { PosePredictor } from '../src/fusion/predict';
import { ImuBuffer } from '../src/fusion/imu';
import { K, POSTER_W_M, getPoster } from './fixtures';
import { mat3, vec3, rotationAngle } from '../src/math/rot';

const tick = () => new Promise((r) => setTimeout(r, 0));

describe('worker cores (tracking + acquisition over a MessageChannel)', () => {
  it('finds the target, streams snapshots, and the main-side predictor matches ground truth', async () => {
    const poster = getPoster();
    const buf = encodeTargets([targetFor(poster, POSTER_W_M)]);
    const msgs: FromTracker[] = [];
    const core = new TrackerCore((m) => msgs.push(m));
    const acq = new AcquireCore();
    const ch = new MessageChannel();
    acq.init({ type: 'init-acquire', targets: buf.slice(0), width: 640, height: 480, K, port: ch.port2 });
    core.handle({
      type: 'init', targets: buf.slice(0), width: 640, height: 480, K, maxTrack: 1, useImu: true,
      mainTimeOrigin: performance.timeOrigin, acquirePort: ch.port1,
    });
    expect(msgs[0]).toEqual({ type: 'ready' });

    const scn = scenarios.slowSweep();
    const imu = simulateImu(scn, {});
    const mainImu = new ImuBuffer(512);
    const pred = new PosePredictor();
    const r = createSceneRenderer(scn, { K, poster, posterWidthM: POSTER_W_M });
    let ii = 0;
    const R = mat3(), t = vec3();
    const errs: number[] = [];
    let found = false;
    for (let k = 0; k < 45; k++) {
      const tc = k / 30;
      const batch: number[] = [];
      while (ii < imu.length && imu[ii].t <= tc + 0.02) {
        const s = imu[ii++];
        batch.push(s.t * 1000, s.gyro[0], s.gyro[1], s.gyro[2], s.accel[0], s.accel[1], s.accel[2]);
        mainImu.push(s.t, s.gyro[0], s.gyro[1], s.gyro[2], s.accel[0], s.accel[1], s.accel[2]);
      }
      core.handle({ type: 'imu', samples: Float64Array.from(batch) });
      const luma = r.render(tc, scn.camera, { noise: 2 }).data;
      const lb = luma.slice().buffer;
      core.handle({ type: 'frame', id: k, t: tc * 1000, luma: lb });
      // Let the acquisition worker answer.
      for (let i = 0; i < 5; i++) await tick();
      for (const m of msgs.splice(0)) {
        if (m.type !== 'result') continue;
        const res = m as ResultMessage;
        if (res.found.length || res.targets.length) found = true;
        for (const s of res.snapshots) pred.ingest(s.snap);
      }
      if (pred.ready && pred.poseAt(tc, mainImu, R, t)) {
        const g = relativePose(scn.camera(tc), scn.target(tc));
        errs.push(rotationAngle(R, g.R) * 180 / Math.PI);
        expect(Math.hypot(t[0] - g.t[0], t[1] - g.t[1], t[2] - g.t[2])).toBeLessThan(0.002);
      }
    }
    expect(found).toBe(true);
    expect(errs.length).toBeGreaterThan(35);
    expect(Math.max(...errs)).toBeLessThan(0.5);
  });

  it('hands the frame buffer back for reuse', () => {
    const poster = getPoster();
    const buf = encodeTargets([targetFor(poster, POSTER_W_M)]);
    const msgs: FromTracker[] = [];
    const core = new TrackerCore((m) => msgs.push(m));
    const ch = new MessageChannel();
    core.handle({
      type: 'init', targets: buf, width: 640, height: 480, K, maxTrack: 1, useImu: false,
      mainTimeOrigin: performance.timeOrigin, acquirePort: ch.port1,
    });
    const lb = new ArrayBuffer(640 * 480);
    core.handle({ type: 'frame', id: 1, t: 0, luma: lb });
    const res = msgs.find((m) => m.type === 'result') as ResultMessage;
    expect(res.buffer).toBe(lb);
    ch.port1.close();
  });
});
