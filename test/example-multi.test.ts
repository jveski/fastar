// The multi-target example as shipped: its four clips playing side by side on
// one screen, tracked from its targets.bin, each clip's time recovered.
import { describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { K } from './fixtures';
import { decodeTargets } from '../src/target/format';
import { multiScenarios } from '../src/sim/scene';
import { percentile, runMultiScenario } from '../src/sim/harness';
import type { GrayImage } from '../src/image/gray';

const DIR = 'examples/multi';
const NAMES = ['amber', 'cobalt', 'jade', 'coral'];
const W = 640, H = 360, FPS = 30, LOOP = 12;
const DURATION = 3;
/** Each screen's clip starts at a different frame (the videos are not in step). */
const STARTS = [0, 93, 192, 276];
const hasFfmpeg = spawnSync('ffmpeg', ['-version']).status === 0;
const ready = hasFfmpeg && existsSync(`${DIR}/targets.bin`) && NAMES.every((n) => existsSync(`${DIR}/clips/${n}.mp4`));

/** Frames [first, first + count) of a clip, as decoded from its mp4 (what the screen shows). */
function decode(name: string, first: number, count: number): GrayImage[] {
  // Looped once: a window may run past the end of the clip, as the screen's does.
  const r = spawnSync('ffmpeg', [
    '-v', 'error', '-stream_loop', '1', '-i', `${DIR}/clips/${name}.mp4`, '-vf', `select=gte(n\\,${first}),format=gray`, '-frames:v', String(count),
    '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'gray', '-',
  ], { maxBuffer: 1 << 28 });
  if (r.status !== 0) throw new Error(`ffmpeg: ${r.stderr}`);
  const out: GrayImage[] = [];
  for (let k = 0; k < count; k++) out.push({ width: W, height: H, data: new Uint8Array(r.stdout.buffer, r.stdout.byteOffset + k * W * H, W * H) });
  return out;
}

describe('multi-target example', () => {
  it.skipIf(!ready)('tracks the four clips on one screen at once, and recovers each one\'s time', () => {
    const buf = readFileSync(`${DIR}/targets.bin`);
    const targets = decodeTargets(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    expect(targets.map((t) => t.meta.name)).toEqual(NAMES);
    const widthM = targets[0].meta.physicalWidth;
    const frames = NAMES.map((n, i) => decode(n, STARTS[i], DURATION * FPS + 2));
    const posters = NAMES.map((_, i) => ({
      texture: frames[i][0], widthM,
      textureAt: (t: number) => frames[i][Math.min(frames[i].length - 1, Math.floor(t * FPS + 1e-6))],
    }));
    // A 2 x 2 wall of 14 cm screens, seen from 32 cm (the whole wall in view).
    const scn = multiScenarios.grid(2, 2, widthM + 0.03, (widthM * H) / W + 0.03, 0.32, DURATION);
    const r = runMultiScenario(scn, { K, posters, targets });
    const settled = r.frames.slice(45);
    for (let i = 0; i < 4; i++) {
      expect(settled.every((f) => f.targets[i].visible)).toBe(true);
      const errs = settled.map((f) => f.targets[i].overlayErrPx);
      expect(percentile(errs, 0.95)).toBeLessThan(3);
      const locked = settled.filter((f) => f.targets[i].videoLocked);
      expect(locked.length / settled.length).toBeGreaterThan(0.8);
      const timeErr = locked.map((f) => {
        const truth = (STARTS[i] / FPS + f.t) % LOOP;
        let d = Math.abs(f.targets[i].videoTime - truth) % LOOP;
        if (d > LOOP / 2) d = LOOP - d;
        return d;
      });
      expect(percentile(timeErr, 0.95)).toBeLessThan(0.06);
    }
  });
});
