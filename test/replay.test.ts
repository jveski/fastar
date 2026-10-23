// Recorded-session regression tests.
//
// Real sessions recorded with the benchmark page ("Record + luma") go in
// test/sessions/<name>.json + <name>.luma, with the targets file as
// test/sessions/<name>.bin (or test/sessions/targets.bin). Each session may
// carry expectations in <name>.expect.json:
//   { "maxTrackLosses": 0, "maxStillJitterMm": 0.5, "maxP95FrameMs": 4 }
// A synthetic session exercises the same path when none are present.
import { describe, expect, it } from 'vitest';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { replaySession } from '../src/sim/replay';
import { decodeTargets } from '../src/target/format';
import type { SessionLog } from '../src/shell/recorder';
import { createSceneRenderer, scenarios, simulateImu } from '../src/sim/scene';
import { targetFor } from '../src/sim/harness';
import { K, POSTER_W_M, getPoster } from './fixtures';

function syntheticSession(): { log: SessionLog; luma: Uint8Array } {
  const scn = scenarios.calibrateThenShake(0.35, 2.5, 1.0);
  const poster = getPoster();
  const r = createSceneRenderer(scn, { K, poster, posterWidthM: POSTER_W_M });
  const td = 0.025;
  const imu = simulateImu(scn, { gyroBias: [0.004, -0.006, 0.002] });
  const n = Math.floor(scn.duration * 30);
  const luma = new Uint8Array(n * 640 * 480);
  const frames = [];
  for (let k = 0; k < n; k++) {
    const t = k / 30;
    luma.set(r.render(t, scn.camera, { noise: 2 }).data, k * 640 * 480);
    frames.push({ t: (t - td) * 1000, mediaTime: t, presentedFrames: k, expectedDisplayTime: t * 1000 + 50 });
  }
  const log: SessionLog = {
    version: 1,
    device: { userAgent: 'synthetic', videoWidth: 1280, videoHeight: 960, width: 640, height: 480, intrinsics: K, sensors: 'sim', path: 'copy', td: 0, tdCalibrated: false },
    startTime: frames[0].t,
    frames,
    imu: imu.map((s) => [s.t * 1000, ...s.gyro, ...s.accel]),
    lumaFrames: frames.map((_, i) => i),
  };
  return { log, luma };
}

describe('recorded replay', () => {
  it('replays a synthetic recording end to end', () => {
    const { log, luma } = syntheticSession();
    // Round-trip through JSON like a real file.
    const s = replaySession(JSON.parse(JSON.stringify(log)), luma, [targetFor(getPoster(), POSTER_W_M)]);
    expect(s.frames.filter((f) => f.visible).length).toBeGreaterThan(s.frames.length - 2);
    expect(s.trackLosses).toBe(0);
    expect(s.acquisitions).toBe(1);
    expect(s.tdCalibrated).toBe(true);
    expect(Math.abs(s.td - 0.025)).toBeLessThan(0.002);
    expect(s.p95FrameMs).toBeLessThan(10);
  });

  const dir = join(__dirname, 'sessions');
  const sessions = existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith('.json') && !f.endsWith('.expect.json')) : [];
  for (const file of sessions) {
    const name = file.replace(/\.json$/, '');
    it(`session ${name}`, () => {
      const log = JSON.parse(readFileSync(join(dir, file), 'utf8')) as SessionLog;
      const lumaPath = [`${name}.luma`, `${name}.luma.gz`].map((f) => join(dir, f)).find(existsSync);
      if (!lumaPath) throw new Error(`missing ${name}.luma`);
      let luma: Uint8Array = readFileSync(lumaPath);
      if (lumaPath.endsWith('.gz')) luma = gunzipSync(luma);
      const tPath = [`${name}.bin`, 'targets.bin'].map((f) => join(dir, f)).find(existsSync);
      if (!tPath) throw new Error(`missing targets for ${name}`);
      const tb = readFileSync(tPath);
      const targets = decodeTargets(tb.buffer.slice(tb.byteOffset, tb.byteOffset + tb.byteLength));
      const exp = existsSync(join(dir, `${name}.expect.json`))
        ? JSON.parse(readFileSync(join(dir, `${name}.expect.json`), 'utf8'))
        : {};
      const s = replaySession(log, luma, targets);
      expect(s.trackLosses).toBeLessThanOrEqual(exp.maxTrackLosses ?? Infinity);
      if (exp.maxStillJitterMm !== undefined) expect(s.stillJitterMm).toBeLessThanOrEqual(exp.maxStillJitterMm);
      if (exp.maxP95FrameMs !== undefined) expect(s.p95FrameMs).toBeLessThanOrEqual(exp.maxP95FrameMs);
    });
  }
});
