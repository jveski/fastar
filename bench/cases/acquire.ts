import type { BenchCase } from '../harness';
import { intrinsicsFromFov } from '../../src/vision/camera';
import { makePoster } from '../../src/sim/poster';
import { createSceneRenderer, lookAt, pose, WALL_R, type Scenario } from '../../src/sim/scene';
import { compileTarget } from '../../src/target/compiler';
import { Acquirer } from '../../src/track/acquire';

export function acquireCases(): BenchCase[] {
  const K = intrinsicsFromFov(640, 480, 63);
  const poster = makePoster(640, 480, 11);
  const target = compileTarget(poster, { physicalWidth: 0.2 });
  const eye = [0.08, -0.32, 0.03];
  const cam = pose(lookAt(eye, [0, 0, 0]), eye);
  const scn: Scenario = { name: 'b', duration: 1, camera: () => cam, target: () => pose(WALL_R, [0, 0, 0]) };
  const img = createSceneRenderer(scn, { K, poster, posterWidthM: 0.2 }).render(0, scn.camera, { noise: 2, blurPasses: 1 });
  // Realistic miss: the textured wall with the poster out of view.
  const away: Scenario = { name: 'away', duration: 1, camera: () => cam, target: () => pose(WALL_R, [3, 0, 0]) };
  const empty = createSceneRenderer(away, { K, poster, posterWidthM: 0.2 }).render(0, away.camera, { noise: 2, blurPasses: 1 }).data;
  return [
    {
      name: 'acquisition (target present)',
      budgetMs: 25 / 3,
      iterations: 40,
      warmup: 5,
      setup: () => {
        const acq = new Acquirer([target], 640, 480, K);
        return () => {
          if (!acq.acquire(img.data)) throw new Error('acquisition failed');
        };
      },
    },
    {
      name: 'acquisition (no target)',
      budgetMs: 25 / 3,
      iterations: 40,
      warmup: 5,
      setup: () => {
        const acq = new Acquirer([target], 640, 480, K);
        return () => acq.acquire(empty);
      },
    },
    ...multiCases(K, img.data, empty, target),
  ];
}

/**
 * Several targets: unbounded passes grow with the target count (each
 * target's matching is ~3 ms here); a budgeted pass stays bounded.
 */
function multiCases(K: ReturnType<typeof intrinsicsFromFov>, present: Uint8Array, empty: Uint8Array, first: ReturnType<typeof compileTarget>): BenchCase[] {
  const others = [23, 37, 41, 53, 67].map((s) => compileTarget(makePoster(640, 480, s), { physicalWidth: 0.2 }));
  const all = [first, ...others];
  const out: BenchCase[] = [];
  for (const n of [3, 6]) {
    for (const [what, img] of [['one in view', present], ['none in view', empty]] as const) {
      out.push({
        name: `acquisition, ${n} targets (${what})`,
        budgetMs: 25 / 3 + 3.5 * (n - 1),
        iterations: 30,
        warmup: 3,
        setup: () => {
          const acq = new Acquirer(all.slice(0, n), 640, 480, K);
          return () => acq.acquireAll(img, { maxResults: n });
        },
      });
    }
  }
  out.push({
    name: 'acquisition pass, 6 targets, 20 ms matching budget on a phone (one in view)',
    budgetMs: (25 + 20) / 3,
    iterations: 30,
    warmup: 3,
    setup: () => {
      const acq = new Acquirer(all, 640, 480, K);
      return () => acq.acquireAll(present, { maxResults: 6, budgetMs: 20 / 3 });
    },
  });
  return out;
}
