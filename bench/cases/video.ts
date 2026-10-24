import type { BenchCase } from '../harness';
import { intrinsicsFromFov } from '../../src/vision/camera';
import { makeMotionClip } from '../../src/sim/motion';
import { createSceneRenderer, lookAt, pose, scenarios, WALL_R, type Scenario } from '../../src/sim/scene';
import { videoTargetFor, videoTimeAt } from '../../src/sim/harness';
import { Acquirer } from '../../src/track/acquire';
import { Pipeline } from '../../src/track/pipeline';
import type { TargetData } from '../../src/target/format';

// Video targets: a 10 s, 960x540 looping clip shown 0.6 m wide. Compiled
// lazily (in setup) so filtering out these cases skips the compile.
export function videoCases(): BenchCase[] {
  const K = intrinsicsFromFov(640, 480, 63);
  const W = 0.6;
  let clip: ReturnType<typeof makeMotionClip> | null = null;
  let target: TargetData | null = null;
  const load = () => {
    clip ??= makeMotionClip({ width: 960, height: 540, fps: 30, duration: 10, cutAt: 0.45 });
    target ??= videoTargetFor(clip, W);
    return { clip, target };
  };
  return [
    {
      name: 'video acquisition (10 s clip)',
      budgetMs: 30 / 3,
      iterations: 40,
      warmup: 5,
      setup: () => {
        const { clip, target } = load();
        const eye = [0.1, -0.75, 0.03];
        const cam = pose(lookAt(eye, [0, 0, 0]), eye);
        const scn: Scenario = { name: 'b', duration: 1, camera: () => cam, target: () => pose(WALL_R, [0, 0, 0]) };
        const img = createSceneRenderer(scn, { K, poster: clip.frame(0), posterWidthM: W, textureAt: () => clip.frame(170) })
          .render(0, scn.camera, { noise: 2, blurPasses: 1 });
        const acq = new Acquirer([target], 640, 480, K);
        return () => {
          if (!acq.acquire(img.data)?.time) throw new Error('acquisition failed');
        };
      },
    },
    {
      name: 'video tracking path per frame (incl. clock probes)',
      budgetMs: 12 / 3,
      iterations: 80,
      warmup: 5,
      setup: () => {
        const { clip, target } = load();
        const scn = scenarios.static(0.75, 4);
        const video = { clip, start: -2.01 };
        const r = createSceneRenderer(scn, {
          K, poster: clip.frame(0), posterWidthM: W, textureAt: (t) => clip.frame(Math.floor(videoTimeAt(video, t) * clip.fps)),
        });
        // Video time has to run forward, so the sequence plays once (setup + warmup + iterations frames).
        const seq = Array.from({ length: 90 }, (_, k) => r.render(k / 30, scn.camera, { noise: 2 }).data);
        const p = new Pipeline([target], K, { width: 640, height: 480 });
        const acq = new Acquirer([target], 640, 480, K);
        const res = p.processFrame(seq[0], 0);
        const a = acq.acquire(seq[0]);
        if (!a || p.applyAcquisition(a, res.frameId) === null) throw new Error('no lock');
        let k = 0;
        return () => {
          k++;
          const fr = p.processFrame(seq[k], k / 30);
          if (fr.targets[0]?.status !== 'tracking') throw new Error('lost track in benchmark');
        };
      },
    },
  ];
}
