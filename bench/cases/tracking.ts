import type { BenchCase } from '../harness';
import { intrinsicsFromFov } from '../../src/vision/camera';
import { makePoster } from '../../src/sim/poster';
import { createMultiSceneRenderer, createSceneRenderer, multiScenarios, scenarios } from '../../src/sim/scene';
import { targetFor } from '../../src/sim/harness';
import { Acquirer } from '../../src/track/acquire';
import { Pipeline } from '../../src/track/pipeline';
import { Eskf } from '../../src/fusion/eskf';
import { ImuBuffer } from '../../src/fusion/imu';
import { simulateImu, relativePose } from '../../src/sim/scene';
import { SurfaceMesh } from '../../src/deform/mesh';
import { SurfaceSolver } from '../../src/deform/solver';
import { SheetShape, bends } from '../../src/sim/bend';
import { Rng } from '../../src/math/rng';

export function trackingCases(): BenchCase[] {
  const K = intrinsicsFromFov(640, 480, 63);
  const poster = makePoster(640, 480, 11);
  const target = targetFor(poster, 0.2);
  const scn = scenarios.slowSweep();
  const r = createSceneRenderer(scn, { K, poster, posterWidthM: 0.2 });
  // Ping-pong sequence so consecutive frames are always adjacent in time.
  const fwd = Array.from({ length: 31 }, (_, k) => r.render(0.5 + k / 30, scn.camera, { noise: 2 }).data);
  const seq = [...fwd, ...fwd.slice(1, -1).reverse()];
  // A curling sheet for the bending-target path, ping-ponged like the sweep.
  const curl = scenarios.flex(-7, 0.5, 0.3, 0.35, 4);
  const rc = createSceneRenderer(curl, { K, poster, posterWidthM: 0.2 });
  const cfwd = Array.from({ length: 31 }, (_, k) => rc.render(1 + k / 30, curl.camera, { noise: 2 }).data);
  const cseq = [...cfwd, ...cfwd.slice(1, -1).reverse()];
  return [
    {
      name: 'bending-target tracking path per frame (KLT+NCC+pose+shape+ESKF)',
      budgetMs: 14 / 3,
      iterations: 300,
      warmup: 60,
      setup: () => {
        const p = new Pipeline([target], K, { width: 640, height: 480, deformable: [0] });
        const acq = new Acquirer([target], 640, 480, K);
        let k = 0, t = 0;
        const res = p.processFrame(cseq[0], 0);
        const a = acq.acquire(cseq[0]);
        if (!a || p.applyAcquisition(a, res.frameId) === null) throw new Error('no lock');
        return () => {
          k = (k + 1) % cseq.length;
          t += 1 / 30;
          const fr = p.processFrame(cseq[k], t);
          if (fr.targets[0]?.status !== 'tracking') throw new Error('lost track in benchmark');
        };
      },
    },
    {
      name: 'surface shape solve (9x7 mesh, 250 pts, 2 iterations)',
      budgetMs: 2 / 3,
      iterations: 300,
      setup: () => {
        const mesh = new SurfaceMesh(0.2, 0.15, 9, 7);
        const solver = new SurfaceSolver(mesh, 250);
        const shape = new SheetShape(bends.curl(0.2, -6), 0.1, 0.075);
        const rng = new Rng(4), n = 250;
        const idx = new Int32Array(3 * n), w = new Float32Array(3 * n), img = new Float32Array(2 * n), q = new Float64Array(3);
        for (let i = 0; i < n; i++) {
          const X = rng.range(-0.095, 0.095), Y = rng.range(-0.07, 0.07);
          mesh.bind(X, Y, idx, w, 3 * i);
          shape.point(X, Y, q);
          img[2 * i] = (K.fx * q[0]) / (q[2] + 0.35) + K.cx;
          img[2 * i + 1] = (K.fy * q[1]) / (q[2] + 0.35) + K.cy;
        }
        const V0 = new Float64Array(3 * mesh.count);
        for (let v = 0; v < mesh.count; v++) { V0[3 * v] = mesh.rest[3 * v]; V0[3 * v + 1] = mesh.rest[3 * v + 1]; V0[3 * v + 2] = 0.35; }
        const V = V0.slice();
        return () => {
          V.set(V0);
          solver.solve(K, V, V0, [0, 0, 1], idx, w, img, n);
        };
      },
    },
    {
      name: 'tracking path per frame (pyramid+KLT+NCC+pose+ESKF)',
      budgetMs: 10 / 3,
      iterations: 300,
      warmup: 60,
      setup: () => {
        const p = new Pipeline([target], K, { width: 640, height: 480 });
        const acq = new Acquirer([target], 640, 480, K);
        let k = 0, t = 0;
        const res = p.processFrame(seq[0], 0);
        const a = acq.acquire(seq[0]);
        if (!a || p.applyAcquisition(a, res.frameId) === null) throw new Error('no lock');
        return () => {
          k = (k + 1) % seq.length;
          t += 1 / 30;
          const fr = p.processFrame(seq[k], t);
          if (fr.targets[0]?.status !== 'tracking') throw new Error('lost track in benchmark');
        };
      },
    },
    ...multiTrackingCases(K),
    {
      name: 'ESKF update + re-propagation (100 ms delay)',
      budgetMs: 0.3 / 3,
      iterations: 300,
      setup: () => {
        const buf = new ImuBuffer(512);
        const imu = simulateImu(scn, {});
        for (const s of imu) buf.push(s.t, s.gyro[0], s.gyro[1], s.gyro[2], s.accel[0], s.accel[1], s.accel[2]);
        const f = new Eskf({ useImu: true, estimateTd: true });
        const C = new Float64Array(36);
        for (let i = 0; i < 6; i++) C[i * 7] = i < 3 ? 4e-6 : 2.5e-7;
        let tc = 0.5;
        const g = relativePose(scn.camera(tc), scn.target(tc));
        f.initialize(tc, g.R, g.t, C, null);
        return () => {
          tc += 1 / 30;
          if (tc > 3.5) {
            tc = 0.5;
            f.initialize(tc, g.R, g.t, C, null);
          }
          f.propagateTo(tc + 0.1, buf);
          const m = relativePose(scn.camera(tc), scn.target(tc));
          f.updatePose(tc, m.R, m.t, C, buf);
        };
      },
    },
  ];
}

/**
 * Four posters in view: at full resolution, all of them fit the default
 * budget here; with a tight budget (a slow phone) the frame stays bounded.
 */
function multiTrackingCases(K: ReturnType<typeof intrinsicsFromFov>): BenchCase[] {
  const posters = [11, 23, 37, 41].map((s) => ({ texture: makePoster(640, 480, s), widthM: 0.2 }));
  const targets = posters.map((p) => targetFor(p.texture, p.widthM));
  const scn = multiScenarios.grid(2, 2, 0.24, 0.19, 0.6, 3, -0.08, 0.06);
  const r = createMultiSceneRenderer(scn, K, posters);
  const fwd = Array.from({ length: 31 }, (_, k) => r.render(k / 30, scn.camera, { noise: 2 }).data);
  const seq = [...fwd, ...fwd.slice(1, -1).reverse()];
  const make = (trackMs?: number) => () => {
    const p = new Pipeline(targets, K, { width: 640, height: 480, maxTrack: 4, budget: trackMs ? { budgetMs: trackMs } : undefined });
    const acq = new Acquirer(targets, 640, 480, K);
    const res = p.processFrame(seq[0], 0);
    for (const a of acq.acquireAll(seq[0], { maxResults: 4 }).results) p.applyAcquisition(a, res.frameId);
    if (p.slots.length !== 4) throw new Error('no lock on all four');
    let k = 0, t = 0;
    return () => {
      k = (k + 1) % seq.length;
      t += 1 / 30;
      const fr = p.processFrame(seq[k], t);
      if (fr.targets.length !== 4 || fr.targets.some((x) => x.status !== 'tracking')) throw new Error('lost track in benchmark');
    };
  };
  return [
    { name: 'tracking path per frame, 4 targets in view', budgetMs: 8, iterations: 200, warmup: 60, setup: make() },
    { name: 'tracking path per frame, 4 targets in view, 2 ms tracking budget', budgetMs: 3, iterations: 200, warmup: 60, setup: make(2) },
  ];
}
