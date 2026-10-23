import type { BenchCase } from '../harness';
import { intrinsicsFromFov } from '../../src/vision/camera';
import { makePoster } from '../../src/sim/poster';
import { createSceneRenderer, scenarios, relativePose } from '../../src/sim/scene';
import { Pyramid } from '../../src/image/pyramid';
import { Klt } from '../../src/vision/klt';
import { NccMatcher } from '../../src/vision/ncc';
import { PoseSolver, projectPoint, poseToHomography } from '../../src/vision/pose';
import { mat3Invert } from '../../src/math/rot';
import { Rng } from '../../src/math/rng';

const K = intrinsicsFromFov(640, 480, 63);
const poster = makePoster(640, 480, 11);
const W_M = 0.2;
const s = W_M / poster.width;

function fixture() {
  const scn = scenarios.slowSweep();
  const r = createSceneRenderer(scn, { K, poster, posterWidthM: W_M });
  const t0 = 1, t1 = 1 + 1 / 30;
  const a = r.render(t0, scn.camera, { noise: 1 });
  const b = r.render(t1, scn.camera, { noise: 1 });
  const A = relativePose(scn.camera(t0), scn.target(t0));
  const B = relativePose(scn.camera(t1), scn.target(t1));
  const rng = new Rng(3);
  const tex: Array<[number, number]> = [];
  while (tex.length < 200) {
    const u = 16 + rng.int(poster.width - 32), v = 16 + rng.int(poster.height - 32);
    const d = poster.data, w = poster.width;
    const g = Math.abs(d[v * w + u + 2] - d[v * w + u - 2]) + Math.abs(d[(v + 2) * w + u] - d[(v - 2) * w + u]);
    if (g > 40) tex.push([u, v]);
  }
  return { a, b, A, B, tex };
}

export function visionCases(): BenchCase[] {
  const f = fixture();
  const toMetres = (u: number, v: number) => [(u - (poster.width - 1) / 2) * s, (v - (poster.height - 1) / 2) * s];
  return [
    {
      name: 'pyramid 640x480 x3 levels',
      budgetMs: 1 / 3,
      setup: () => {
        const p = new Pyramid(640, 480, 3);
        return () => p.build(f.a.data);
      },
    },
    {
      name: 'KLT 150 pts fwd-bwd',
      budgetMs: 1.2,
      setup: () => {
        const pa = new Pyramid(640, 480, 3).build(f.a.data);
        const pb = new Pyramid(640, 480, 3).build(f.b.data);
        const pts = new Float32Array(300), guess = new Float32Array(300), out = new Float32Array(300);
        const tmp = new Float32Array(2);
        for (let i = 0; i < 150; i++) {
          const [X, Y] = toMetres(f.tex[i][0], f.tex[i][1]);
          projectPoint(K, f.A.R, f.A.t, X, Y, tmp);
          pts[2 * i] = tmp[0]; pts[2 * i + 1] = tmp[1];
          projectPoint(K, f.B.R, f.B.t, X, Y, tmp);
          guess[2 * i] = tmp[0] + 0.7; guess[2 * i + 1] = tmp[1] - 0.5;
        }
        const status = new Uint8Array(150);
        const klt = new Klt();
        return () => {
          out.set(guess);
          klt.trackBatch(pa, pb, pts, out, 150, status);
        };
      },
    },
    {
      name: 'NCC 50 pts (warp + search)',
      budgetMs: 0.5,
      setup: () => {
        const H = poseToHomography(K, f.B.R, f.B.t, s, (poster.width - 1) / 2, (poster.height - 1) / 2, new Float64Array(9));
        const Hinv = new Float64Array(9);
        mat3Invert(H, Hinv);
        const m = new NccMatcher(4, 4);
        const cs = new Float64Array(100);
        for (let i = 0; i < 50; i++) {
          const [u, v] = f.tex[i];
          const w = H[6] * u + H[7] * v + H[8];
          cs[2 * i] = (H[0] * u + H[1] * v + H[2]) / w + 0.8;
          cs[2 * i + 1] = (H[3] * u + H[4] * v + H[5]) / w - 0.6;
        }
        return () => {
          for (let i = 0; i < 50; i++) if (m.buildTemplate(poster, Hinv, cs[2 * i], cs[2 * i + 1])) m.match(f.b);
        };
      },
    },
    {
      name: 'pose solver 150 pts',
      budgetMs: 0.5 / 3,
      setup: () => {
        const obj = new Float32Array(300), img = new Float32Array(300), tmp = new Float32Array(2);
        const rng = new Rng(9);
        for (let i = 0; i < 150; i++) {
          const [X, Y] = toMetres(f.tex[i][0], f.tex[i][1]);
          obj[2 * i] = X; obj[2 * i + 1] = Y;
          projectPoint(K, f.B.R, f.B.t, X, Y, tmp);
          img[2 * i] = tmp[0] + rng.gauss() * 0.3; img[2 * i + 1] = tmp[1] + rng.gauss() * 0.3;
        }
        const solver = new PoseSolver();
        const R = new Float64Array(9), t = new Float64Array(3);
        return () => {
          R.set(f.A.R); t.set(f.A.t);
          solver.solve(K, obj, img, 150, R, t);
        };
      },
    },
  ];
}
