// The example animation (two textures churned by swirling flows, flooding
// over each other and changing colour all the time) compiles into a healthy
// target although nothing on screen looks the same a second later, acquires
// at any point of the loop, and tracks with the clock locked, through the
// loop seam too. The scene description the overlay uses matches the picture.
import { describe, expect, it } from 'vitest';
import { makeShowcaseClip, waveSum } from '../src/sim/showcase';
import { runScenario, videoTargetFor, percentile, type RunResult } from '../src/sim/harness';
import { createSceneRenderer, scenarios } from '../src/sim/scene';
import { Acquirer } from '../src/track/acquire';
import { VPT_STATIONARY, VPT_STRIDE } from '../src/target/format';
import { K } from './fixtures';

const clip = makeShowcaseClip({ width: 640 });
const target = videoTargetFor(clip, 0.5);
const D = clip.duration;

/** NCC of two greyscale frames over cell (cx, cy) of a gx x gy grid. */
function cellNcc(a: Uint8Array, b: Uint8Array, cx: number, cy: number, gx: number, gy: number): number {
  const W = clip.width, H = clip.height;
  const x0 = Math.floor((cx * W) / gx), x1 = Math.floor(((cx + 1) * W) / gx);
  const y0 = Math.floor((cy * H) / gy), y1 = Math.floor(((cy + 1) * H) / gy);
  let sa = 0, sb = 0, n = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { sa += a[y * W + x]; sb += b[y * W + x]; n++; }
  const ma = sa / n, mb = sb / n;
  let ab = 0, aa = 0, bb = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const u = a[y * W + x] - ma, v = b[y * W + x] - mb;
      ab += u * v; aa += u * u; bb += v * v;
    }
  }
  return ab / Math.sqrt(aa * bb);
}

const timeErr = (r: RunResult, from: number) => r.frames.filter((f) => f.t > from).map((f) => {
  const d = (((f.videoTime - f.videoTruth) % D) + D) % D;
  return Math.abs(d > D / 2 ? d - D : d);
});

/** Tracked from the first frames on, overlay within half a pixel, clock locked and right. */
function expectSteady(r: RunResult, from: number): void {
  expect(r.acquisitions).toBe(1);
  const late = r.frames.filter((f) => f.t > from);
  expect(late.every((f) => f.visible)).toBe(true);
  expect(percentile(late.map((f) => f.overlayErrPx), 0.95)).toBeLessThan(0.5);
  expect(late.filter((f) => f.videoLocked).length).toBeGreaterThan(0.95 * late.length);
  expect(percentile(timeErr(r, from), 0.95)).toBeLessThan(0.02);
}

describe('showcase clip', () => {
  it('loops seamlessly', () => {
    const a = clip.frame(clip.frameCount - 1).data, b = clip.frame(0).data, c = clip.frame(1).data;
    let seam = 0, step = 0;
    for (let i = 0; i < a.length; i++) { seam += Math.abs(a[i] - b[i]); step += Math.abs(b[i] - c[i]); }
    // The seam changes the picture no more than an ordinary frame step does.
    expect(seam).toBeLessThan(1.1 * step);
  });

  it('never holds still: every part of the picture is unrecognisable a second later', () => {
    let worst = -1, sum = 0, n = 0;
    for (let f = 0; f < clip.frameCount; f += 37) {
      const a = clip.frame(f).data, b = clip.frame(f + clip.fps).data;
      for (let cy = 0; cy < 4; cy++) {
        for (let cx = 0; cx < 8; cx++) {
          const c = cellNcc(a, b, cx, cy, 8, 4);
          worst = Math.max(worst, c);
          sum += c; n++;
        }
      }
    }
    expect(worst).toBeLessThan(0.5);
    expect(Math.abs(sum / n)).toBeLessThan(0.1);
  });

  it('compiles into a healthy target with nothing that stays put', () => {
    const v = target.meta.video!, pts = target.video!.points, range = target.video!.featureRange;
    let still = 0, through = 0;
    for (let j = 0; j < pts.length; j += VPT_STRIDE) {
      if (pts[j + 6] & VPT_STATIONARY) still++;
      if (pts[j + 1] >= v.frameCount) through++;
    }
    expect(still).toBe(0);
    // No feature outlasts a few keyframes: every look is gone within seconds.
    let longest = 0;
    for (let i = 0; i < range.length; i += 2) longest = Math.max(longest, range[i + 1] - range[i] + 1);
    expect(longest).toBeLessThan(4);
    // Continuous change, not cuts; points everywhere, timing every second,
    // and points carried through the seam.
    expect(v.changes).toEqual([]);
    expect(v.report.minPoints).toBeGreaterThan(100);
    expect(v.report.timingPerSecond.every(Boolean)).toBe(true);
    expect(target.meta.quality.warnings).toEqual([]);
    expect(through).toBeGreaterThan(50);
  });

  it('acquires at any time of the loop, with the right time', () => {
    const acq = new Acquirer([target], K.width, K.height, K);
    const scn = scenarios.static(0.6, 1);
    let found = 0, n = 0;
    for (let i = 5; i < clip.frameCount; i += 29) {
      const img = createSceneRenderer(scn, { K, poster: clip.frame(0), posterWidthM: 0.5, textureAt: () => clip.frame(i) })
        .render(0, scn.camera, { noise: 2, blurPasses: 1 });
      const a = acq.acquire(img.data);
      n++;
      if (!a?.time) continue;
      found++;
      const tv = (i + 0.5) / clip.fps, r = a.time.ranges;
      let miss = Infinity;
      for (let k = 0; k < r.length; k += 2) {
        for (const t of [tv - D, tv, tv + D]) miss = Math.min(miss, Math.max(0, r[k] - t, t - r[k + 1]));
      }
      // The matched keyframe's time window holds (or, by the loop seam or a
      // neighbouring keyframe whose content slid into place, just misses)
      // the shown frame; tracking then finds the time itself.
      expect(miss).toBeLessThan(0.25);
    }
    expect(found).toBeGreaterThanOrEqual(n - 1);
  });

  it('tracks a still camera with the clock locked', () => {
    const r = runScenario(scenarios.static(0.6, 3), { K, poster: clip.frame(0), posterWidthM: 0.5, video: { clip, start: -0.37 } });
    expectSteady(r, 0.4);
  });

  it('tracks a moving camera through the loop seam with the clock locked', () => {
    // Video time 10.4 s at the start, so the seam comes 1.6 s in.
    const r = runScenario(scenarios.slowSweep(0.6, 4), { K, poster: clip.frame(0), posterWidthM: 0.5, video: { clip, start: -10.41 } });
    expectSteady(r, 0.4);
  });

  it('tracks with the IMU', () => {
    const r = runScenario(scenarios.slowSweep(0.6, 3), { K, poster: clip.frame(0), posterWidthM: 0.5, useImu: true, video: { clip, start: -6.52 } });
    expectSteady(r, 0.5);
  });

  it('describes the swirl at the centre as the picture shows it', () => {
    // Around the screen centre, each layer is sampled from c + z R(a) (p - c)
    // plus a shift: the picture turns by a (counter-clockwise as seen, with
    // y down in design pixels) and scales by 1 / z.
    const { spin, zoom, designWidth: W, designHeight: H } = clip.scene;
    const h = 0.5;
    for (let t = 0; t < D; t += 0.7) {
      for (const k of [0, 1]) {
        const q = clip.warp(k, W / 2, H / 2, t), qx = clip.warp(k, W / 2 + h, H / 2, t), qy = clip.warp(k, W / 2, H / 2 + h, t);
        const a = (qx[0] - q[0]) / h, b = (qy[0] - q[0]) / h, c = (qx[1] - q[1]) / h, d = (qy[1] - q[1]) / h;
        expect(Math.abs(Math.atan2(c - b, a + d) - waveSum(spin, t))).toBeLessThan(0.02);
        expect(Math.abs(0.5 * Math.log(a * d - b * c) - waveSum(zoom, t))).toBeLessThan(0.01);
      }
    }
  });
});
