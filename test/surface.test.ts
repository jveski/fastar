import { describe, expect, it } from 'vitest';
import { Surface, SurfaceFeed } from '../src/shell/surface';
import { ThreeAdapter } from '../src/adapters/three';
import type { FrameEvent, ImageTracker, Pose } from '../src/shell/image-tracker';
import { TrackerCore } from '../src/worker/tracker-core';
import { AcquireCore } from '../src/worker/acquire-core';
import type { FromTracker, ResultMessage } from '../src/worker/protocol';
import { encodeTargets } from '../src/target/format';
import { targetFor } from '../src/sim/harness';
import { createSceneRenderer, scenarios } from '../src/sim/scene';
import { K, POSTER_W_M, getPoster } from './fixtures';

/** A sheet curled about the vertical axis on a circle of radius r (app axes: y up, z out of the print). */
function curled(cols: number, rows: number, W: number, H: number, r: number): Surface {
  const p = new Float32Array(3 * cols * rows);
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const s = -W / 2 + (i * W) / (cols - 1), a = s / r, v = 3 * (j * cols + i);
      p[v] = r * Math.sin(a);
      p[v + 1] = H / 2 - (j * H) / (rows - 1);
      p[v + 2] = r * (1 - Math.cos(a));
    }
  }
  return new Surface(cols, rows, W, H, p);
}

describe('Surface', () => {
  it('samples a flat target by texture coordinates', () => {
    const s = Surface.flat(9, 7, 0.2, 0.15);
    const p = s.sample(0.25, 0.75);
    expect(p.position[0]).toBeCloseTo(-0.05, 6);
    expect(p.position[1]).toBeCloseTo(0.0375, 6);
    expect(p.position[2]).toBeCloseTo(0, 6);
    expect(p.normal[0]).toBeCloseTo(0, 9);
    expect(p.normal[1]).toBeCloseTo(0, 9);
    expect(p.normal[2]).toBeCloseTo(1, 9);
    expect(p.tangentU[0]).toBeCloseTo(1, 6);
    expect(p.tangentV[1]).toBeCloseTo(1, 6);
  });

  it('interpolates a curl smoothly, with unit tangents and the normal out of the print', () => {
    const r = 0.15, s = curled(13, 7, 0.2, 0.15, r);
    for (const u of [0.13, 0.5, 0.71, 0.97]) {
      const p = s.sample(u, 0.4);
      const a = (0.2 * u - 0.1) / r;
      // Within 0.2 mm of the circle between vertices 17 mm apart.
      expect(Math.abs(p.position[0] - r * Math.sin(a))).toBeLessThan(2e-4);
      expect(Math.abs(p.position[2] - r * (1 - Math.cos(a)))).toBeLessThan(2e-4);
      expect(Math.hypot(...p.tangentU)).toBeCloseTo(1, 2);
      expect(p.normal[0]).toBeCloseTo(-Math.sin(a), 2);
      expect(p.normal[2]).toBeCloseTo(Math.cos(a), 2);
    }
  });

  it('fills a render grid and its indices', () => {
    const s = Surface.flat(9, 7, 0.2, 0.15);
    const pos = new Float32Array(3 * 12), nrm = new Float32Array(3 * 12), uv = new Float32Array(2 * 12);
    s.fillGrid(3, 2, pos, nrm, uv);
    expect(Array.from(pos.slice(0, 3)).map((v) => +v.toFixed(6))).toEqual([-0.1, -0.075, 0]);
    expect(Array.from(uv.slice(-2))).toEqual([1, 1]);
    expect(nrm[2]).toBe(1);
    const idx = Surface.gridIndices(3, 2);
    expect(idx.length).toBe(36);
    // Counter-clockwise from the printed side: (b - a) x (d - a) along +z.
    const v = (k: number) => [pos[3 * k], pos[3 * k + 1]];
    const [a, b, d] = [v(idx[0]), v(idx[1]), v(idx[2])];
    expect((b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0])).toBeGreaterThan(0);
  });
});

describe('SurfaceFeed', () => {
  it('serves the shape measured on or before a frame, in the app axes, easing small changes only', () => {
    const f = new SurfaceFeed(3, 3, 0.2, 0.2, 0.5);
    const shape = (z: number) => Float32Array.from({ length: 27 }, (_, i) => (i % 3 === 2 ? z : i % 3 === 1 ? 0.05 : 0));
    f.push(100, shape(0.002));
    f.push(133, shape(0.0025));
    expect(f.at(90)!.positions[2]).toBeCloseTo(-0.002, 6);
    expect(f.at(100)!.positions[1]).toBeCloseTo(-0.05, 6);
    // 0.5 mm later: eased halfway.
    expect(f.at(133)!.positions[2]).toBeCloseTo(-0.00225, 6);
    // A real bend (2 cm) shows at once.
    f.push(166, shape(0.0225));
    expect(f.at(166)!.positions[2]).toBeCloseTo(-0.0225, 6);
  });
});

describe('three.js surface mesh', () => {
  it('follows the pose surface each frame', () => {
    class M4 { e = new Float64Array(16); fromArray(a: ArrayLike<number>) { for (let i = 0; i < 16; i++) this.e[i] = a[i]; return this; } identity() { return this; } }
    class Obj { matrix = new M4(); matrixAutoUpdate = true; matrixWorldNeedsUpdate = false; visible = true; children: unknown[] = []; add(o: unknown) { this.children.push(o); } }
    class Attr { needsUpdate = false; constructor(public array: Float32Array, public itemSize: number) {} setUsage() {} }
    class Geo { attrs: Record<string, Attr> = {}; index: unknown; spheres = 0; setAttribute(n: string, a: Attr) { this.attrs[n] = a; } getAttribute(n: string) { return this.attrs[n]; } setIndex(i: unknown) { this.index = i; } computeBoundingSphere() { this.spheres++; } }
    class Mesh extends Obj { constructor(public geometry: Geo, public material: unknown) { super(); } }
    let cb: ((e: FrameEvent) => void) | null = null;
    const ar = {
      targets: [{ physicalWidth: 0.2, width: 640, height: 480 }],
      onFrame: (f: (e: FrameEvent) => void) => { cb = f; return () => {}; },
      getProjectionMatrix: () => new Float32Array(16),
      get projectionMatrix() { return new Float32Array(16); },
    } as unknown as ImageTracker;
    const camera = Object.assign(new Obj(), { projectionMatrix: new M4(), near: 0.01, far: 100 });
    const ad = new ThreeAdapter(ar, { camera, scene: { add() {} }, THREE: { Group: Obj, BufferGeometry: Geo, BufferAttribute: Attr, Mesh } as never });
    const mesh = ad.surface(0, { segments: [4, 2] }) as unknown as Mesh;
    expect((ad.anchor(0) as unknown as Obj).children).toContain(mesh);
    const pos = mesh.geometry.attrs.position.array;
    expect(pos[2]).toBe(0);
    const pose: Pose = { matrix: new Float32Array(16), R: new Float64Array(9), t: new Float64Array(3), confidence: 1, surface: curled(9, 7, 0.2, 0.15, 0.12) };
    cb!({ presentedFrame: {} as FrameEvent['presentedFrame'], poses: new Map([[0, pose]]) });
    // Left edge curls out of the print.
    expect(pos[2]).toBeGreaterThan(0.01);
    expect(mesh.geometry.attrs.position.needsUpdate).toBe(true);
  });
});

describe('worker: bending targets', () => {
  it('sends the shape of a curled sheet with each result', async () => {
    const poster = getPoster();
    const buf = encodeTargets([targetFor(poster, POSTER_W_M)]);
    const msgs: FromTracker[] = [];
    const core = new TrackerCore((m) => msgs.push(m));
    const acq = new AcquireCore();
    const ch = new MessageChannel();
    acq.init({ type: 'init-acquire', targets: buf.slice(0), width: 640, height: 480, K, port: ch.port2 });
    core.handle({
      type: 'init', targets: buf.slice(0), width: 640, height: 480, K, maxTrack: 1, useImu: false,
      mainTimeOrigin: performance.timeOrigin, acquirePort: ch.port1, deformable: [0],
    });
    const scn = scenarios.curl(-6);
    const r = createSceneRenderer(scn, { K, poster, posterWidthM: POSTER_W_M });
    let last: ResultMessage['surfaces'] | undefined;
    for (let k = 0; k < 12; k++) {
      core.handle({ type: 'frame', id: k, t: (k * 1000) / 30, luma: r.render(k / 30, scn.camera, { noise: 2 }).data.slice().buffer });
      for (let i = 0; i < 5; i++) await new Promise((res) => setTimeout(res, 0));
      for (const m of msgs.splice(0)) if (m.type === 'result' && m.surfaces) last = m.surfaces;
    }
    expect(last).toBeDefined();
    const s = last![0];
    expect([s.index, s.cols, s.rows]).toEqual([0, 9, 7]);
    // Curled toward the camera (target -z) at the left and right edges relative to the middle.
    const z = (i: number, j: number) => s.vertices[3 * (j * s.cols + i) + 2];
    expect(z(0, 3) - z(4, 3)).toBeLessThan(-0.01);
    expect(z(8, 3) - z(4, 3)).toBeLessThan(-0.01);
  });
});
