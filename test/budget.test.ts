import { describe, expect, it } from 'vitest';
import { CostModel, DEFAULT_BUDGET, allocate, screenPriority, type Demand } from '../src/track/budget';
import { keepMask, setLoad, DEFAULT_TRACKER, TargetTracker } from '../src/track/tracker';
import { SurfaceTracker } from '../src/deform/surface-tracker';
import { targetFor } from '../src/sim/harness';
import { Pyramid } from '../src/image/pyramid';
import { createSceneRenderer, lookAt, pose, relativePose, WALL_R, type Scenario } from '../src/sim/scene';
import { K, POSTER_W_M, getPoster } from './fixtures';

const demand = (index: number, priority: number, o: Partial<Demand> = {}): Demand => ({
  index, priority, minPoints: 24, maxPoints: 150, current: NaN, fixed: 0.2, perPoint: 0.01, ...o,
});
const costOf = (a: { cost: number }[]) => a.reduce((s, x) => s + x.cost, 0);

describe('tracking budget allocation', () => {
  it('gives every target its full points when the budget allows', () => {
    const a = allocate([demand(0, 0.5), demand(1, 0.2), demand(2, 0.1)], 20);
    expect(a.map((x) => x.points)).toEqual([150, 150, 150]);
    expect(a.every((x) => x.every === 1)).toBe(true);
  });

  it('fills the most important target first, then the next, keeping every floor', () => {
    // Floors: 3 x (0.2 + 0.24) = 1.32 ms; 1.0 ms left buys 100 more points.
    const a = allocate([demand(0, 0.1), demand(1, 0.9), demand(2, 0.4)], 2.32);
    expect(a.map((x) => x.index)).toEqual([1, 2, 0]);
    expect(a[0].points).toBe(124);
    expect(a[1].points).toBe(24);
    expect(a[2].points).toBe(24);
    expect(costOf(a)).toBeCloseTo(2.32, 6);
  });

  it('when even the floors do not fit, updates the least important targets less often, never the most important', () => {
    const d = [demand(0, 0.9), demand(1, 0.5), demand(2, 0.3), demand(3, 0.1)];
    const a = allocate(d, 1.2);
    expect(a[0].every).toBe(1);
    expect(a.slice(1).some((x) => x.every > 1)).toBe(true);
    // Least important first: rates never increase down the ranking.
    for (let i = 1; i < a.length; i++) expect(a[i].every).toBeGreaterThanOrEqual(a[i - 1].every);
    expect(costOf(a)).toBeLessThanOrEqual(1.2 + 1e-9);
    // What the slower rates free goes to the most important target; the rest stay at the floor.
    expect(a[0].points).toBeGreaterThan(24);
    expect(a.slice(1).every((x) => x.points === 24)).toBe(true);
  });

  it('stops at maxEvery: past that the floors are kept (best effort) rather than dropping targets', () => {
    const a = allocate([demand(0, 1), demand(1, 0.5), demand(2, 0.2)], 0.1);
    expect(a.map((x) => x.every)).toEqual([1, DEFAULT_BUDGET.maxEvery, DEFAULT_BUDGET.maxEvery]);
    expect(a.every((x) => x.points === 24)).toBe(true);
  });

  it('lowers a target\'s points gradually rather than all at once', () => {
    const a = allocate([demand(0, 1, { current: 150 }), demand(1, 0.5, { current: 150 })], 0.5);
    expect(a[0].points).toBe(Math.round(150 * (1 - DEFAULT_BUDGET.maxDrop)));
    // Increases are not limited (replenishing paces them).
    const b = allocate([demand(0, 1, { current: 24 })], 20);
    expect(b[0].points).toBe(150);
  });

  it('treats free work as free', () => {
    const a = allocate([demand(0, 1, { perPoint: 0, fixed: 0 }), demand(1, 0.5, { perPoint: 0, fixed: 0 })], 0);
    expect(a.map((x) => x.points)).toEqual([150, 150]);
  });
});

describe('screen priority', () => {
  const hw = 0.1, hh = 0.075;
  const at = (x: number, y: number, z: number, R: ArrayLike<number> = [1, 0, 0, 0, 1, 0, 0, 0, 1]) => screenPriority(K, R, [x, y, z], hw, hh);

  it('ranks a near, centred, fully visible target highest', () => {
    const near = at(0, 0, 0.3), far = at(0, 0, 0.9), off = at(0.08, 0.05, 0.3), edge = at(0.2, 0, 0.3);
    expect(near.inside).toBeCloseTo(1, 6);
    expect(near.offCentre).toBeLessThan(0.01);
    expect(off.inside).toBeCloseTo(1, 6);
    expect(near.score).toBeGreaterThan(far.score);
    expect(near.score).toBeGreaterThan(off.score);
    expect(off.score).toBeGreaterThan(edge.score);
    // Half the view is 0.3 tan(31.5 deg) = 0.184 m wide: 0.1 to 0.3 m is about 42% in view.
    expect(edge.inside).toBeCloseTo(0.42, 1);
  });

  it('measures coverage of the image', () => {
    // 0.2 m wide at 0.3 m: fx * 0.2 / 0.3 px wide.
    const s = at(0, 0, 0.3);
    const w = (K.fx * 0.2) / 0.3, h = (K.fy * 0.15) / 0.3;
    expect(s.coverage).toBeCloseTo((w * h) / (K.width * K.height), 3);
  });

  it('is zero out of view or behind the camera', () => {
    expect(at(2, 0, 0.3).score).toBe(0);
    expect(at(0, 0, -0.3).score).toBe(0);
  });

  it('handles a target crossing the camera plane', () => {
    // Tilted 80 degrees about y, half of it behind the camera.
    const a = (80 * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
    const r = at(0, 0, 0.02, [c, 0, s, 0, 1, 0, -s, 0, c]);
    expect(r.score).toBeGreaterThan(0);
    expect(r.inside).toBeLessThan(1);
  });
});

describe('cost model', () => {
  it('starts from its prior, then fits fixed + per-point cost from what it sees', () => {
    const m = new CostModel(0.3, 0.01);
    expect(m.cost(100)).toBeCloseTo(1.3, 9);
    for (let i = 0; i < 200; i++) {
      const n = 24 + ((i * 37) % 127);
      m.observe(n, 0.5 + 0.02 * n);
    }
    expect(m.fixed).toBeCloseTo(0.5, 2);
    expect(m.perPoint).toBeCloseTo(0.02, 4);
  });

  it('puts all the cost on the points while the allocation does not vary', () => {
    const m = new CostModel(0.3, 0.01);
    for (let i = 0; i < 50; i++) m.observe(150, 3);
    expect(m.fixed).toBe(0);
    expect(m.perPoint).toBeCloseTo(0.02, 9);
  });

  it('follows a device that slows down', () => {
    const m = new CostModel(0.1, 0.01);
    for (let i = 0; i < 100; i++) m.observe(24 + (i % 5) * 30, 0.01 * (24 + (i % 5) * 30));
    for (let i = 0; i < 200; i++) m.observe(24 + (i % 5) * 30, 0.04 * (24 + (i % 5) * 30));
    expect(m.perPoint).toBeCloseTo(0.04, 3);
  });
});

describe('tracker load', () => {
  it('scales drift correction and replenishing with the points, within the options', () => {
    const load = { maxPoints: 0, nccPerFrame: 0, replenishAttempts: 0 };
    setLoad(DEFAULT_TRACKER, load, 75);
    expect(load).toEqual({ maxPoints: 75, nccPerFrame: 25, replenishAttempts: 20 });
    setLoad(DEFAULT_TRACKER, load, 500);
    expect(load).toEqual({ maxPoints: 150, nccPerFrame: 50, replenishAttempts: 40 });
    setLoad(DEFAULT_TRACKER, load, 10);
    expect(load.nccPerFrame).toBe(8);
    expect(load.replenishAttempts).toBe(8);
  });

  it('keeps the highest-weighted points when trimming', () => {
    const w = Float64Array.of(1, 0, 0.5, 0.9, 0.2, 1, 0.5);
    const keep = keepMask(w, 7, 3, new Float64Array(7));
    expect([0, 1, 2, 3, 4, 5, 6].filter(keep)).toEqual([0, 3, 5]);
    const keepAll = keepMask(w, 7, 10, new Float64Array(7));
    expect([0, 1, 2, 3, 4, 5, 6].filter(keepAll)).toEqual([0, 2, 3, 4, 5, 6]);
    // Ties at the cut: only as many as fit, in order.
    const ties = keepMask(Float64Array.of(0.5, 0.5, 0.5, 1), 4, 2, new Float64Array(4));
    expect([0, 1, 2, 3].filter(ties)).toEqual([0, 3]);
  });

  const frames = () => {
    const poster = getPoster();
    const eye = [0.02, -0.3, 0.01];
    const scn: Scenario = { name: 's', duration: 1, camera: (t) => { const e = [eye[0] + 0.01 * t, eye[1], eye[2]]; return pose(lookAt(e, [0, 0, 0]), e); }, target: () => pose(WALL_R, [0, 0, 0]) };
    const r = createSceneRenderer(scn, { K, poster, posterWidthM: POSTER_W_M });
    const pyr = (t: number) => { const p = new Pyramid(640, 480, 3); p.build(r.render(t, scn.camera, { noise: 2 }).data); return p; };
    return { scn, pyr };
  };

  for (const kind of ['rigid', 'bending'] as const) {
    it(`${kind}: tracks with at most the allowed points, and back up to full`, () => {
      const { scn, pyr } = frames();
      const T = targetFor(getPoster(), POSTER_W_M);
      const tr = kind === 'rigid' ? new TargetTracker(T, K) : new SurfaceTracker(T, K);
      const g0 = relativePose(scn.camera(0), scn.target(0));
      let prev = pyr(0);
      expect(tr.init(prev, g0.R, g0.t)).toBe(true);
      const full = tr.o.maxPoints;
      expect(tr.n).toBeGreaterThan(0.8 * full);
      tr.setLoad(40);
      for (let k = 1; k <= 3; k++) {
        const cur = pyr(k / 30), g = relativePose(scn.camera(k / 30), scn.target(k / 30));
        expect(tr.track(prev, cur, g.R, g.t)).toBe(true);
        expect(tr.n).toBeLessThanOrEqual(40);
        expect(tr.inliers).toBeLessThanOrEqual(40);
        expect(Math.hypot(tr.t[0] - g.t[0], tr.t[1] - g.t[1], tr.t[2] - g.t[2])).toBeLessThan(0.002);
        prev = cur;
      }
      tr.setLoad(full);
      for (let k = 4; k <= 12; k++) {
        const cur = pyr(k / 30), g = relativePose(scn.camera(k / 30), scn.target(k / 30));
        expect(tr.track(prev, cur, g.R, g.t)).toBe(true);
        prev = cur;
      }
      expect(tr.n).toBeGreaterThan(0.8 * full);
    });
  }
});
