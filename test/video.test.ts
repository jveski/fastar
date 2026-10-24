// Video targets: compiler, format, acquisition, and the clock + tracking end to end.
import { describe, expect, it } from 'vitest';
import { makeMotionClip } from '../src/sim/motion';
import { runScenario, videoTargetFor, percentile } from '../src/sim/harness';
import { createSceneRenderer, lookAt, pose, scenarios, WALL_R, type Scenario } from '../src/sim/scene';
import { compileTarget } from '../src/target/compiler';
import { decodeTargets, encodeTargets } from '../src/target/format';
import { VideoModel } from '../src/target/video';
import { Acquirer } from '../src/track/acquire';
import { Pipeline } from '../src/track/pipeline';
import { TargetTracker } from '../src/track/tracker';
import { Pyramid } from '../src/image/pyramid';
import { K, getPoster } from './fixtures';

const WIDTH_M = 0.5;
const clip = makeMotionClip({ duration: 6 });
const target = videoTargetFor(clip, WIDTH_M);
const D = clip.duration;

/** Signed video-time error, wrapped to (-D/2, D/2]. */
const tvErr = (est: number, truth: number) => {
  const d = (((est - truth) % D) + D) % D;
  return d > D / 2 ? d - D : d;
};

describe('video target compiler', () => {
  it('finds the cut and keeps enough points everywhere', () => {
    const v = target.meta.video!;
    expect(v.frameCount).toBe(clip.frameCount);
    expect(v.changes).toEqual(clip.cuts);
    expect(v.keyframes[0]).toBe(0);
    expect(v.keyframes).toContain(clip.cuts[0]);
    expect(v.report.minPoints).toBeGreaterThanOrEqual(60);
    expect(v.report.movingPoints).toBeGreaterThan(30);
    expect(v.report.timingPerSecond.every(Boolean)).toBe(true);
    expect(target.meta.quality.warnings).toEqual([]);
    expect(target.images).toEqual([]);
  });

  it('round-trips through format v2; image-only files stay v1', () => {
    const buf = encodeTargets([target]);
    expect(new DataView(buf).getUint32(4, true)).toBe(2);
    expect(buf.byteLength).toBeLessThan(1.5e6);
    const [t] = decodeTargets(buf);
    expect(t.meta).toEqual(target.meta);
    expect(t.video!.points).toEqual(target.video!.points);
    expect(t.video!.path).toEqual(target.video!.path);
    expect(t.video!.patches).toEqual(target.video!.patches);
    expect(t.video!.featureRange).toEqual(target.video!.featureRange);
    expect(t.video!.points.buffer).toBe(buf);
    const img = encodeTargets([compileTarget(getPoster(), { physicalWidth: 0.2 })]);
    expect(new DataView(img).getUint32(4, true)).toBe(1);
    // Mixed files carry both.
    const [a, b] = decodeTargets(encodeTargets([compileTarget(getPoster(), { physicalWidth: 0.2 }), target]));
    expect(a.video).toBeUndefined();
    expect(b.video).toBeDefined();
  });

  it('bakes moving points along the true motion', () => {
    const V = new VideoModel(target);
    const cw = Math.round(clip.width * 0.26), ch = Math.round(clip.height * 0.32);
    const p = new Float64Array(2), q = new Float64Array(2);
    let checked = 0;
    for (let j = 0; j < V.numPoints; j++) {
      const s = V.start[j], e = V.end[j];
      if (e >= clip.cuts[0] || e - s < 10 || V.flags[j]) continue;
      const c0 = clip.cardAt(s)!, c1 = clip.cardAt(e)!;
      V.positionAt(j, s + 0.5, p);
      // Well inside the card at its first frame.
      if (p[0] < c0[0] + 10 || p[1] < c0[1] + 10 || p[0] > c0[0] + cw - 10 || p[1] > c0[1] + ch - 10) continue;
      V.positionAt(j, e + 0.5, q);
      expect(Math.abs(q[0] - p[0] - (c1[0] - c0[0]))).toBeLessThan(0.5);
      expect(Math.abs(q[1] - p[1] - (c1[1] - c0[1]))).toBeLessThan(0.5);
      checked++;
    }
    expect(checked).toBeGreaterThan(5);
  });

  it('keeps the channel logo as points spanning the whole loop', () => {
    const V = new VideoModel(target);
    const lx = clip.width - clip.width * 0.16 - clip.width * 0.03;
    const p = new Float64Array(2);
    let logo = 0;
    for (let j = 0; j < V.numPoints; j++) {
      V.positionAt(j, 0.5, p);
      if (p[0] > lx + 2 && p[1] < clip.height * 0.04 + clip.width * 0.16 - 2 && V.always(j)) logo++;
    }
    expect(logo).toBeGreaterThanOrEqual(3);
  });
});

describe('video acquisition', () => {
  it('finds the screen and when the matched content is on screen', () => {
    const eye = [0.1, -0.6, 0.04];
    const cam = pose(lookAt(eye, [0, 0, 0]), eye);
    const scn: Scenario = { name: 'v', duration: 1, camera: () => cam, target: () => pose(WALL_R, [0, 0, 0]) };
    const acq = new Acquirer([target], K.width, K.height, K);
    let found = 0;
    for (const tv of [0.4, 1.7, 2.9, 3.2, 4.5, 5.8]) {
      const f = Math.floor(tv * clip.fps);
      const r = createSceneRenderer(scn, { K, poster: clip.frame(0), posterWidthM: WIDTH_M, textureAt: () => clip.frame(f) });
      const img = r.render(0, scn.camera, { noise: 2, blurPasses: 1 });
      const res = acq.acquire(img.data);
      if (!res) continue;
      found++;
      expect(res.time).toBeDefined();
      const rg = res.time!.ranges;
      let inside = false;
      for (let k = 0; k < rg.length; k += 2) {
        const a = rg[k] - 0.05, len = rg[k + 1] - rg[k] + 0.1;
        inside ||= ((tv - a) % D + D) % D <= len;
      }
      expect(inside).toBe(true);
    }
    expect(found).toBeGreaterThanOrEqual(5);
  });
});

describe('video tracker', () => {
  it('seeds from patches at the level matching the viewing distance', () => {
    const acq = new Acquirer([target], K.width, K.height, K);
    for (const dist of [0.35, 0.7, 1.2]) {
      const eye = [0.02, -dist, 0.01];
      const cam = pose(lookAt(eye, [0, 0, 0]), eye);
      const scn: Scenario = { name: 'v', duration: 1, camera: () => cam, target: () => pose(WALL_R, [0, 0, 0]) };
      const f = 40;
      const img = createSceneRenderer(scn, { K, poster: clip.frame(0), posterWidthM: WIDTH_M, textureAt: () => clip.frame(f) })
        .render(0, scn.camera, { noise: 2 }).data;
      const a = acq.acquire(img);
      expect(a).not.toBeNull();
      const tr = new TargetTracker(target, K);
      tr.setTime({ x: f + 0.5, lo: f + 0.5, hi: f + 0.5, locked: true });
      expect(tr.init(new Pyramid(K.width, K.height, 3).build(img), a!.R, a!.t)).toBe(true);
      expect(tr.inliers).toBeGreaterThan(40);
    }
  });
});

describe('video end-to-end', () => {
  it('static camera: locks the clock within a second and tracks through the cut and the loop seam', () => {
    const c25 = makeMotionClip({ duration: 6, fps: 25 });
    // Lock at 1.7 s into the clip; the cut comes at 3 s and the seam at 6 s.
    const r = runScenario(scenarios.static(0.6, 6), { K, poster: c25.frame(0), posterWidthM: WIDTH_M, video: { clip: c25, start: -1.712 } });
    expect(r.acquisitions).toBe(1);
    expect(r.firstLockTime).toBeLessThan(0.2);
    const after = r.frames.filter((f) => f.t > 0.2);
    expect(after.every((f) => f.visible && f.status === 'tracking')).toBe(true);
    expect(percentile(after.map((f) => f.overlayErrPx), 0.95)).toBeLessThan(0.5);
    const late = r.frames.filter((f) => f.t > 1);
    expect(late.filter((f) => f.videoLocked).length).toBeGreaterThan(0.95 * late.length);
    const err = late.map((f) => Math.abs(tvErr(f.videoTime, f.videoTruth)));
    expect(percentile(err, 0.5)).toBeLessThan(0.008);
    expect(percentile(err, 0.95)).toBeLessThan(0.02);
  });

  it('moving camera, player pausing at the loop seam: keeps tracking and learns the pause', () => {
    const c25 = makeMotionClip({ duration: 6, fps: 25 });
    const video = { clip: c25, start: -4.21, loopGap: 0.06 };
    // Seams at about 1.8 s and 7.85 s.
    const r = runScenario(scenarios.slowSweep(0.6, 9), { K, poster: c25.frame(0), posterWidthM: WIDTH_M, video });
    expect(r.acquisitions).toBe(1);
    const after = r.frames.filter((f) => f.t > 0.2);
    expect(after.every((f) => f.visible)).toBe(true);
    expect(percentile(after.map((f) => f.overlayErrPx), 0.95)).toBeLessThan(0.6);
    const clock = r.pipeline.clocks.get(0)!;
    expect(clock.loopGap).toBeGreaterThan(0.03);
    expect(clock.loopGap).toBeLessThan(0.09);
    // Re-locked well before the end, and correct.
    const tail = r.frames.filter((f) => f.t > 8.3);
    expect(tail.every((f) => f.videoLocked)).toBe(true);
    expect(Math.max(...tail.map((f) => Math.abs(tvErr(f.videoTime, f.videoTruth))))).toBeLessThan(0.02);
  });

  it('follows the player seeking (jumping 2.4 s, across the cut) within 0.3 s', () => {
    const c25 = makeMotionClip({ duration: 6, fps: 25 });
    const video = { clip: c25, start: -0.512, jumps: [[2, 2.37]] as Array<[number, number]> };
    const r = runScenario(scenarios.static(0.6, 3.2), { K, poster: c25.frame(0), posterWidthM: WIDTH_M, video });
    expect(r.acquisitions).toBe(1);
    expect(r.frames.filter((f) => f.t > 0.2).every((f) => f.visible)).toBe(true);
    const after = r.frames.filter((f) => f.t > 2.3);
    expect(after.every((f) => f.videoLocked && Math.abs(tvErr(f.videoTime, f.videoTruth)) < 0.02)).toBe(true);
  });

  it('with the IMU: locks and tracks through the cut', () => {
    const c25 = makeMotionClip({ duration: 6, fps: 25 });
    const r = runScenario(scenarios.slowSweep(0.6, 3), {
      K, poster: c25.frame(0), posterWidthM: WIDTH_M, useImu: true, video: { clip: c25, start: -1.512 },
    });
    expect(r.frames.filter((f) => f.t > 0.3).every((f) => f.visible)).toBe(true);
    expect(percentile(r.frames.map((f) => f.overlayErrPx), 0.95)).toBeLessThan(1);
    const late = r.frames.filter((f) => f.t > 0.5);
    expect(late.filter((f) => f.videoLocked).length).toBeGreaterThan(0.9 * late.length);
  });

  it('pulls a wrong lock back to the true time (whole-clip probe sweeps)', () => {
    const c25 = makeMotionClip({ duration: 6, fps: 25 });
    const video = { clip: c25, start: -0.512 };
    let corrupted = false;
    const r = runScenario(scenarios.static(0.6, 3.5), {
      K, poster: c25.frame(0), posterWidthM: WIDTH_M, video,
      onFrame: (p, tCam, tTrue) => {
        if (corrupted || tTrue < 1) return;
        // Convince the clock it is 1.3 s later (same shot, mostly the same content).
        const c = p.clocks.get(0)!;
        const tv = (tTrue + 0.512 + 1.3) % c.duration;
        c.reset();
        for (let k = 0; k < 4; k++) c.addRanges(tCam, [tv - 0.01, tv + 0.01], 0);
        corrupted = true;
      },
    });
    const at = r.frames.find((f) => f.t > 1.05)!;
    expect(Math.abs(tvErr(at.videoTime, at.videoTruth))).toBeGreaterThan(1);
    const tail = r.frames.filter((f) => f.t > 3);
    expect(tail.every((f) => f.visible && f.videoLocked && Math.abs(tvErr(f.videoTime, f.videoTruth)) < 0.02)).toBe(true);
  });

  it('reports the clock to the render side', () => {
    const p = new Pipeline([target], K, { width: K.width, height: K.height });
    expect(p.clockStates()).toEqual([]);
  });
});
