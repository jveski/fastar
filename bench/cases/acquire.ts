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
  ];
}
