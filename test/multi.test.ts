import { describe, expect, it } from 'vitest';
import { K, POSTER_W_M } from './fixtures';
import { makePoster } from '../src/sim/poster';
import { compileTarget } from '../src/target/compiler';
import { encodeTargets } from '../src/target/format';
import { Acquirer, targetQuad } from '../src/track/acquire';
import { createMultiSceneRenderer, createSceneRenderer, lookAt, multiScenarios, pose, relativePose, WALL_R, type Scenario } from '../src/sim/scene';
import { percentile, runMultiScenario, type MultiRunResult } from '../src/sim/harness';
import { rotationAngle } from '../src/math/rot';
import { TrackerCore } from '../src/worker/tracker-core';
import { AcquireCore } from '../src/worker/acquire-core';
import type { FromTracker, ResultMessage } from '../src/worker/protocol';

const posters = [11, 23, 37, 41].map((s) => makePoster(640, 480, s));
const targets = posters.map((p, i) => compileTarget(p, { physicalWidth: POSTER_W_M, name: `poster-${i}` }));
const sim = posters.map((texture) => ({ texture, widthM: POSTER_W_M }));

/** Posters 0 and 1 side by side, both in view. */
const pair = multiScenarios.grid(2, 1, 0.24, 0, 0.55, 1);
const pairFrame = () => createMultiSceneRenderer(pair, K, sim.slice(0, 2)).render(0, pair.camera, { noise: 2, blurPasses: 1 }).data;

function single(eye: number[], posterIndex = 0, at = [0, 0, 0]): Uint8Array {
  const cam = pose(lookAt(eye, at), eye);
  const scn: Scenario = { name: 's', duration: 1, camera: () => cam, target: () => pose(WALL_R, [0, 0, 0]) };
  return createSceneRenderer(scn, { K, poster: posters[posterIndex], posterWidthM: POSTER_W_M }).render(0, scn.camera, { noise: 2, blurPasses: 1 }).data;
}

describe('acquiring several targets', () => {
  it('finds every target in the frame in one pass, each at its own pose', () => {
    const acq = new Acquirer(targets.slice(0, 3), 640, 480, K);
    const b = acq.acquireAll(pairFrame(), { maxResults: 3 });
    expect(b.results.map((r) => r.target).sort()).toEqual([0, 1]);
    expect(b.tried.sort()).toEqual([0, 1, 2]);
    for (const r of b.results) {
      const gt = relativePose(pair.camera(0), pair.targets[r.target](0));
      expect(rotationAngle(r.R, gt.R) * 180 / Math.PI).toBeLessThan(2.5);
      expect(Math.hypot(r.t[0] - gt.t[0], r.t[1] - gt.t[1], r.t[2] - gt.t[2])).toBeLessThan(0.02);
    }
    // Best first.
    expect(b.results[0].inliers).toBeGreaterThanOrEqual(b.results[1].inliers);
  });

  it('returns at most maxResults, and the single best through acquire()', () => {
    const acq = new Acquirer(targets.slice(0, 2), 640, 480, K);
    const all = acq.acquireAll(pairFrame(), { maxResults: 2 });
    const one = acq.acquireAll(pairFrame(), { maxResults: 1 });
    expect(one.results.length).toBe(1);
    expect(one.results[0].target).toBe(all.results[0].target);
    expect(acq.acquire(pairFrame())!.target).toBe(all.results[0].target);
  });

  it('does not report two targets on the same part of the frame', () => {
    // The same picture compiled twice: both match the one poster in view.
    const acq = new Acquirer([targets[0], targets[0]], 640, 480, K);
    const b = acq.acquireAll(single([0.02, -0.35, 0.01]), { maxResults: 2 });
    expect(b.results.length).toBe(1);
  });

  it('skips frame regions already taken by tracked targets, and excluded targets', () => {
    const acq = new Acquirer(targets.slice(0, 2), 640, 480, K);
    const img = pairFrame();
    const both = acq.acquireAll(img, { maxResults: 2 });
    const r0 = both.results.find((r) => r.target === 0)!;
    // Target 1's matches land inside target 0's quad: as if 0 were tracked and 1 looked the same.
    const quad = targetQuad(targets[0], r0.H);
    const b = acquireWith(acq, img, { maxResults: 2, occupied: quad });
    expect(b.map((r) => r.target)).toEqual([1]);
    expect(acquireWith(acq, img, { maxResults: 2, exclude: [1] }).map((r) => r.target)).toEqual([0]);
  });

  it('within a time budget, tries one target per pass at least, round-robin, priority first', () => {
    const acq = new Acquirer(targets, 640, 480, K);
    const img = pairFrame();
    const seen = new Set<number>(), found = new Set<number>();
    for (let k = 0; k < 4; k++) {
      const b = acq.acquireAll(img, { maxResults: 4, budgetMs: 0 });
      expect(b.tried.length).toBe(1);
      b.tried.forEach((i) => seen.add(i));
      b.results.forEach((r) => found.add(r.target));
    }
    expect([...seen].sort()).toEqual([0, 1, 2, 3]);
    expect([...found].sort()).toEqual([0, 1]);
    // Priority targets go first, and do not move the round-robin cursor.
    const p = acq.acquireAll(img, { maxResults: 4, budgetMs: 0, priority: [3] });
    expect(p.tried).toEqual([3]);
    const next = acq.acquireAll(img, { maxResults: 4, budgetMs: 0 });
    expect(next.tried).toEqual([0]);
    // Unbudgeted: every target, priority first (unknown indices ignored), then round-robin from the cursor.
    expect(acq.acquireAll(img, { priority: [2, 9, -1] }).tried).toEqual([2, 1, 3, 0]);
  });

  it('does not let priority targets starve the others when the budget only reaches one', () => {
    const acq = new Acquirer(targets, 640, 480, K);
    const img = pairFrame();
    const tried = Array.from({ length: 6 }, () => acq.acquireAll(img, { budgetMs: 0, priority: [3] }).tried[0]);
    expect(tried).toEqual([3, 0, 3, 1, 3, 2]);
  });

  it('reports a near miss for a target seen too steeply to confirm, and none on empty frames', () => {
    const acq = new Acquirer(targets.slice(0, 3), 640, 480, K);
    const a = (65 * Math.PI) / 180, d = 0.35;
    const steep = acq.acquireAll(single([d * Math.sin(a), -d * Math.cos(a), 0]), { maxResults: 3 });
    expect(steep.results).toEqual([]);
    expect(steep.near).toEqual([0]);
    for (let s = 0; s < 20; s++) {
      const eye = [3 + 0.3 * s, -0.4, 0.1 * s];
      const empty = acq.acquireAll(single(eye, 0, [eye[0], 0, eye[2]]), { maxResults: 3 });
      expect(empty.results).toEqual([]);
      expect(empty.near).toEqual([]);
    }
  });
});

function acquireWith(acq: Acquirer, img: Uint8Array, o: Parameters<Acquirer['acquireAll']>[1]) {
  return acq.acquireAll(img, o).results;
}

const errs = (r: MultiRunResult, i: number) => r.frames.map((f) => f.targets[i].overlayErrPx).filter(Number.isFinite);
const settle = (r: MultiRunResult) => ({ ...r, frames: r.frames.slice(15) });

/**
 * A budget for the 2 x 2 grid that, on this machine at its current load,
 * fits every poster's floor (24 points) plus 100 more points for one: from
 * the costs the pipeline measures with time to spare.
 */
let calibrated = NaN;
function budgetFor100(): number {
  if (Number.isNaN(calibrated)) {
    const r = runMultiScenario(multiScenarios.grid(2, 2, 0.24, 0.19, 0.6, 1, -0.08, 0.06), { K, posters: sim, pipeline: { budget: { budgetMs: 100 } } });
    const c = r.pipeline.costs.image;
    calibrated = 4 * c.cost(24) + 100 * c.perPoint;
  }
  return calibrated;
}

describe('tracking several targets', () => {
  it('pans along a row of posters: each is found as it comes into view, tracked while there, and dropped when gone', () => {
    const scn = multiScenarios.gallery(3, 0.32, 0.4, 6);
    const r = runMultiScenario(scn, { K, posters: sim.slice(0, 3) });
    for (let i = 0; i < 3; i++) {
      const inView = r.frames.filter((f) => f.targets[i].inView > 0.9);
      expect(inView.length).toBeGreaterThan(30);
      expect(inView.filter((f) => f.targets[i].visible).length / inView.length).toBeGreaterThan(0.97);
      // Found before it is fully in view.
      const firstIn = r.frames.findIndex((f) => f.targets[i].inView > 0.9);
      const firstVis = r.frames.findIndex((f) => f.targets[i].visible);
      expect(firstVis).toBeLessThanOrEqual(firstIn);
      expect(percentile(errs(r, i), 0.95)).toBeLessThan(2);
      // Never shown while out of view.
      expect(r.frames.filter((f) => f.targets[i].inView === 0 && f.targets[i].visible && f.targets[i].status === 'tracking').length).toBe(0);
    }
    // The pan ends back at the first poster: the last is out of view and dropped.
    const last = r.frames[r.frames.length - 1];
    expect(last.targets[2].inView).toBe(0);
    expect(last.targets[2].visible).toBe(false);
    expect(last.targets[0].visible).toBe(true);
  });

  it('tracks four posters at once at full resolution when there is time', () => {
    const r = settle(runMultiScenario(multiScenarios.grid(2, 2, 0.24, 0.19, 0.6, 2, -0.08, 0.06), { K, posters: sim, pipeline: { budget: { budgetMs: 100 } } }));
    for (let i = 0; i < 4; i++) {
      expect(r.frames.every((f) => f.targets[i].visible && f.targets[i].points === 150 && f.targets[i].every === 1)).toBe(true);
      expect(percentile(errs(r, i), 0.95)).toBeLessThan(0.5);
    }
  });

  it('short of time, bounds tracking time and favours the most prominent poster, tracking every one', () => {
    // The camera is in front of poster 0 (top left); poster 2 (bottom left) is next nearest.
    const budget = budgetFor100();
    const r = settle(runMultiScenario(multiScenarios.grid(2, 2, 0.24, 0.19, 0.6, 3, -0.08, 0.06), { K, posters: sim, pipeline: { budget: { budgetMs: budget } } }));
    expect(r.frames.every((f) => f.budget === budget)).toBe(true);
    const track = r.frames.map((f) => f.trackMs);
    expect(track.reduce((a, b) => a + b, 0) / track.length).toBeLessThan(1.25 * budget);
    const pts = (i: number) => percentile(r.frames.map((f) => f.targets[i].points), 0.5);
    expect(pts(0)).toBeGreaterThan(60);
    for (const i of [1, 2, 3]) expect(pts(i)).toBeLessThan(pts(0));
    expect(pts(1)).toBe(24);
    expect(pts(3)).toBe(24);
    // Ranked by prominence: the poster in front first.
    const prio = r.frames[r.frames.length - 1].targets.map((t) => t.priority);
    expect(prio[0]).toBe(Math.max(...prio));
    for (let i = 0; i < 4; i++) {
      expect(r.frames.every((f) => f.targets[i].visible)).toBe(true);
      expect(percentile(errs(r, i), 0.95)).toBeLessThan(1);
    }
  });

  it('far short of time, updates the least prominent posters every few frames instead of dropping them', () => {
    for (const useImu of [false, true]) {
      const r = settle(runMultiScenario(multiScenarios.grid(2, 2, 0.24, 0.19, 0.6, 3, -0.08, 0.06), {
        K, posters: sim, useImu, pipeline: { budget: { budgetMs: 0.3 } },
      }));
      expect(r.frames.every((f) => f.targets[0].every === 1)).toBe(true);
      expect(r.frames.filter((f) => f.targets.some((t) => t.every > 1)).length / r.frames.length).toBeGreaterThan(0.9);
      for (let i = 0; i < 4; i++) {
        expect(r.frames.every((f) => f.targets[i].visible)).toBe(true);
        expect(percentile(errs(r, i), 0.95)).toBeLessThan(2);
      }
    }
  });

  it('lets the app raise a poster\'s priority', () => {
    const scn = multiScenarios.grid(2, 2, 0.24, 0.19, 0.6, 2, -0.08, 0.06);
    const r = runMultiScenario(scn, { K, posters: sim, pipeline: { budget: { budgetMs: budgetFor100() } } });
    const p = r.pipeline;
    expect(p.slot(0)!.points).toBeGreaterThan(60);
    expect(p.slot(3)!.points).toBe(24);
    // Weight poster 3 (the least prominent) and carry on.
    p.setPriority(3, 10);
    const renderer = createMultiSceneRenderer(scn, K, sim);
    for (let k = 0; k < 20; k++) {
      const t = 2 + k / 30;
      p.processFrame(renderer.render(t, scn.camera, { noise: 2 }).data, t);
    }
    expect(p.slot(3)!.points).toBeGreaterThan(60);
    expect(p.slot(0)!.points).toBeLessThan(p.slot(3)!.points);
  });

  it('tracks no more than maxTrack targets', () => {
    const r = runMultiScenario(multiScenarios.grid(2, 2, 0.24, 0.19, 0.6, 1), { K, posters: sim, pipeline: { maxTrack: 2 } });
    for (const f of r.frames) expect(f.targets.filter((t) => t.visible).length).toBeLessThanOrEqual(2);
    expect(r.frames[r.frames.length - 1].targets.filter((t) => t.visible).length).toBe(2);
  });

  it('with time for one target per acquisition pass, still finds every poster, and re-finds lost ones', () => {
    const r = runMultiScenario(multiScenarios.gallery(3, 0.32, 0.4, 6), { K, posters: sim.slice(0, 3), acquireBudgetMs: 0 });
    expect(r.passes.every((t) => t.length === 1)).toBe(true);
    for (let i = 0; i < 3; i++) {
      const inView = r.frames.filter((f) => f.targets[i].inView > 0.9);
      expect(inView.filter((f) => f.targets[i].visible).length / inView.length).toBeGreaterThan(0.9);
    }
    // Poster 0 is left behind on the way out and found again on the way back.
    const found0 = r.frames.filter((f) => f.acquired.includes(0)).length;
    expect(found0).toBeGreaterThanOrEqual(2);
    expect(r.frames[r.frames.length - 1].targets[0].visible).toBe(true);
  });
});

describe('worker cores with several targets', () => {
  it('acquires both targets in view, and streams poses, priorities and points for each', async () => {
    const buf = encodeTargets(targets.slice(0, 3));
    const msgs: FromTracker[] = [];
    const core = new TrackerCore((m) => msgs.push(m));
    const acq = new AcquireCore();
    const ch = new MessageChannel();
    acq.init({ type: 'init-acquire', targets: buf.slice(0), width: 640, height: 480, K, port: ch.port2 });
    core.handle({
      type: 'init', targets: buf.slice(0), width: 640, height: 480, K, maxTrack: 3, useImu: false,
      mainTimeOrigin: performance.timeOrigin, acquirePort: ch.port1, acquireBudgetMs: 50,
    });
    const r = createMultiSceneRenderer(pair, K, sim.slice(0, 2));
    let last: ResultMessage | null = null;
    const foundAt = new Map<number, number>();
    for (let k = 0; k < 20; k++) {
      core.handle({ type: 'frame', id: k, t: (k * 1000) / 30, luma: r.render(k / 30, pair.camera, { noise: 2 }).data.slice().buffer });
      for (let i = 0; i < 5; i++) await new Promise((res) => setTimeout(res, 0));
      for (const m of msgs.splice(0)) {
        if (m.type !== 'result') continue;
        const res = m as ResultMessage;
        for (const t of res.targets) if (!foundAt.has(t.index)) foundAt.set(t.index, k);
        if (!res.acquisition) last = res;
      }
    }
    // Both from the same acquisition pass.
    expect([...foundAt.keys()].sort()).toEqual([0, 1]);
    expect(foundAt.get(0)).toBe(foundAt.get(1));
    expect(last!.targets.map((t) => t.index).sort()).toEqual([0, 1]);
    expect(last!.snapshots.map((s) => s.index).sort()).toEqual([0, 1]);
    expect(last!.budget).toBeGreaterThan(0);
    for (const t of last!.targets) {
      expect(t.priority).toBeGreaterThan(0);
      expect(t.points).toBe(150);
      expect(t.every).toBe(1);
    }
    core.handle({ type: 'priority', index: 1, weight: 0 });
    ch.port1.close();
  });
});
