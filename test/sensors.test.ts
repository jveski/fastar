import { describe, expect, it } from 'vitest';
import { deviceToCamera, SensorTimebase } from '../src/shell/sensors';

const v = (x: number, y: number, z: number, ang: number, facing: 'environment' | 'user') =>
  Array.from(deviceToCamera(x, y, z, ang, facing)).map((n) => Math.round(n * 1e9) / 1e9 + 0);

describe('sensor axis normalization', () => {
  it('rear camera, portrait: (x, -y, -z)', () => {
    expect(v(1, 2, 3, 0, 'environment')).toEqual([1, -2, -3]);
  });

  it('front camera, portrait: mirrored (-x, -y, z)', () => {
    expect(v(1, 2, 3, 0, 'user')).toEqual([-1, -2, 3]);
  });

  it('landscape (screen angle 90): device top points left', () => {
    // Device +x points up on screen, device +y points left.
    // A rotation about device y (pointing screen-left) is about camera -x.
    expect(v(0, 1, 0, 90, 'environment')).toEqual([-1, 0, 0]);
    // Rotation about device x (screen up) is about camera -y (camera y is down).
    expect(v(1, 0, 0, 90, 'environment')).toEqual([0, -1, 0]);
  });

  it('gravity reads as camera "up" for an upright phone', () => {
    // Upright portrait phone: gravity reaction along device +y -> camera -y (up in image).
    expect(v(0, 9.81, 0, 0, 'environment')).toEqual([0, -9.81, 0]);
    // Rotated to landscape (angle 90, device top to the left): reaction along device +x.
    expect(v(9.81, 0, 0, 90, 'environment')).toEqual([0, -9.81, 0]);
    expect(v(-9.81, 0, 0, 270, 'environment')).toEqual([0, -9.81, 0]);
  });

  it('is a proper rotation (preserves handedness) for every case', () => {
    for (const ang of [0, 90, 180, 270]) {
      for (const f of ['environment', 'user'] as const) {
        const ex = v(1, 0, 0, ang, f), ey = v(0, 1, 0, ang, f), ez = v(0, 0, 1, ang, f);
        const cross = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]];
        expect(cross.map((n) => Math.round(n) + 0)).toEqual(ez.map((n) => Math.round(n) + 0));
      }
    }
  });
});

describe('sensor timebase', () => {
  // 100 Hz readings delivered 2-12 ms after they were taken.
  const run = (clockOffset: number) => {
    const tb = new SensorTimebase();
    const out: number[] = [];
    for (let k = 0; k < 50; k++) {
      const taken = 1000 + 10 * k;
      out.push(tb.map(taken + clockOffset, taken + 2 + (k * 7) % 11));
    }
    return out;
  };

  it('keeps stamps already on performance.now()', () => {
    const out = run(0);
    out.forEach((t, k) => expect(t).toBe(1000 + 10 * k));
  });

  it('moves Android stamps (boot clock, counting sleep) onto performance.now()', () => {
    // Device slept 3 hours since boot: readings are stamped far in the future.
    const out = run(3 * 3600e3);
    for (let k = 1; k < out.length; k++) expect(out[k] - out[k - 1]).toBeCloseTo(10, 6);
    // Settles on the least delivery delay (2 ms) once a least-delayed reading is seen.
    expect(out[out.length - 1] - (1000 + 10 * (out.length - 1))).toBeCloseTo(2, 6);
  });

  it('follows a sleep while running', () => {
    const tb = new SensorTimebase();
    expect(tb.map(5000, 1002)).toBe(1002);
    // The device sleeps 60 s: the sensor clock jumps ahead, performance.now() does not.
    expect(tb.map(5010 + 60e3, 1012)).toBe(1012);
    expect(tb.map(5020 + 60e3, 1025)).toBe(1022);
  });

  it('falls back to the delivery time without a stamp', () => {
    expect(new SensorTimebase().map(null, 1234)).toBe(1234);
  });
});
