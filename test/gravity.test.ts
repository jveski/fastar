import { describe, expect, it } from 'vitest';
import { AccelSigns } from '../src/fusion/gravity';
import { ImuBuffer } from '../src/fusion/imu';
import { lookAt, pose, scenarios, simulateImu, WALL_R, type Scenario } from '../src/sim/scene';
import { mat3, mat3Mul, so3Exp } from '../src/math/rot';

/** Looking down at a page on a table, wobbling. */
const table: Scenario = {
  name: 'table', duration: 4,
  camera: (t) => {
    const eye = [0.04 * Math.sin(0.9 * t), -0.25, 0.3];
    const R = mat3Mul(lookAt(eye, [0, 0, 0]), so3Exp(0.15 * Math.sin(1.7 * t), 0.2 * Math.sin(1.3 * t), 0.1 * Math.sin(2.1 * t), mat3()), mat3());
    return pose(R, eye);
  },
  target: () => pose(WALL_R, [0, 0, 0]),
};

function signsFor(scn: Scenario, accelSigns: number[]): number[] {
  const imu = new ImuBuffer(512), est = new AccelSigns();
  for (const s of simulateImu(scn, { rate: 100 })) {
    imu.push(s.t, s.gyro[0], s.gyro[1], s.gyro[2], accelSigns[0] * s.accel[0], accelSigns[1] * s.accel[1], accelSigns[2] * s.accel[2]);
    if (imu.count % 3 === 0) est.observe(imu);
  }
  est.observe(imu);
  return Array.from(est.signs);
}

describe('accelerometer signs', () => {
  for (const [name, scn] of [['facing a wall', scenarios.calibrateThenShake()], ['looking down at a table', table]] as const) {
    it(`${name}: keeps correct readings, and undoes any sign convention`, () => {
      for (const s of [[1, 1, 1], [-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, -1, 1]]) {
        expect(signsFor(scn, s)).toEqual(s);
      }
    });
  }
});
