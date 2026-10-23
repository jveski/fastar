import { describe, expect, it } from 'vitest';
import { Eskf } from '../src/fusion/eskf';
import { ImuBuffer } from '../src/fusion/imu';
import { PosePredictor } from '../src/fusion/predict';
import { scenarios, simulateImu, relativePose } from '../src/sim/scene';
import { mat3, vec3, rotationAngle } from '../src/math/rot';

describe('render-side PosePredictor', () => {
  for (const useImu of [true, false]) {
    it(`matches the filter's poseAt from snapshots (imu=${useImu})`, () => {
      const scn = scenarios.slowSweep();
      const imu = simulateImu(scn, { gyroBias: [0.003, 0, -0.002] });
      const workerBuf = new ImuBuffer(512), mainBuf = new ImuBuffer(512);
      const f = new Eskf({ useImu, estimateTd: false });
      const C = new Float64Array(36);
      for (let i = 0; i < 6; i++) C[i * 7] = i < 3 ? 4e-6 : 2.5e-7;
      const pred = new PosePredictor();
      let ii = 0;
      const Ra = mat3(), ta = vec3(), Rb = mat3(), tb = vec3();
      let checked = 0;
      for (let k = 0; k < 60; k++) {
        const tc = k / 30;
        while (ii < imu.length && imu[ii].t <= tc + 0.02) {
          const s = imu[ii++];
          workerBuf.push(s.t, s.gyro[0], s.gyro[1], s.gyro[2]);
          mainBuf.push(s.t, s.gyro[0], s.gyro[1], s.gyro[2]);
        }
        const g = relativePose(scn.camera(tc), scn.target(tc));
        const src = useImu ? workerBuf : null;
        if (k === 0) f.initialize(tc, g.R, g.t, C, null);
        else f.updatePose(tc, g.R, g.t, C, src);
        f.propagateTo(useImu ? workerBuf.latestTime : tc, src);
        pred.ingest(f.exportSnapshot(f.t - 0.3));
        // Main thread keeps receiving gyro after the snapshot; query presented frames around now.
        for (const dtq of [-0.05, -0.01, 0, 0.01]) {
          const tq = tc + dtq;
          if (!f.poseAt(tq, Ra, ta, src)) continue;
          expect(pred.poseAt(tq, useImu ? mainBuf : null, Rb, tb)).toBe(true);
          expect(rotationAngle(Ra, Rb)).toBeLessThan(1e-9);
          expect(Math.hypot(ta[0] - tb[0], ta[1] - tb[1], ta[2] - tb[2])).toBeLessThan(1e-9);
          checked++;
        }
      }
      expect(checked).toBeGreaterThan(200);
    });
  }
});
