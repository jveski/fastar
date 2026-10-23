import { describe, expect, it } from 'vitest';
import { ThreeAdapter } from '../src/adapters/three';
import { registerAFrame } from '../src/adapters/aframe';
import type { FrameEvent, ImageTracker, Pose } from '../src/shell/image-tracker';

class M4 {
  e = new Float64Array(16);
  fromArray(a: ArrayLike<number>) { for (let i = 0; i < 16; i++) this.e[i] = a[i]; return this; }
  copy(m: M4) { this.e.set(m.e); return this; }
  invert() { return this; }
  identity() { this.e.fill(0); this.e[0] = this.e[5] = this.e[10] = this.e[15] = 1; return this; }
}
class Obj { matrix = new M4(); matrixAutoUpdate = true; matrixWorldNeedsUpdate = false; visible = true; }

function fakeTracker() {
  let cb: ((e: FrameEvent) => void) | null = null;
  const P = Float32Array.from({ length: 16 }, (_, i) => i + 1);
  const ar = {
    onFrame: (f: (e: FrameEvent) => void) => { cb = f; return () => { cb = null; }; },
    get projectionMatrix() { return P; },
    getProjectionMatrix: () => P,
  } as unknown as ImageTracker;
  const fire = (poses: Map<number, Pose>) => cb?.({ presentedFrame: {} as FrameEvent['presentedFrame'], poses });
  return { ar, fire, P };
}

const pose = (v: number): Pose => ({ matrix: new Float32Array(16).fill(v), R: new Float64Array(9), t: new Float64Array(3), confidence: 1 });

describe('three.js adapter', () => {
  it('drives projection, anchors and visibility from frame events, then renders', () => {
    const { ar, fire, P } = fakeTracker();
    const camera = Object.assign(new Obj(), { projectionMatrix: new M4(), projectionMatrixInverse: new M4(), near: 0.01, far: 100, updateMatrixWorld() {} });
    const added: unknown[] = [];
    let renders = 0;
    const ad = new ThreeAdapter(ar, { camera, scene: { add: (o) => added.push(o) }, THREE: { Group: Obj }, render: () => renders++ });
    const a0 = ad.anchor(0), a1 = ad.anchor(1);
    expect(added).toEqual([a0, a1]);
    fire(new Map([[0, pose(7)]]));
    expect(Array.from(camera.projectionMatrix.e)).toEqual(Array.from(P));
    expect(a0.visible).toBe(true);
    expect(a0.matrixAutoUpdate).toBe(false);
    expect((a0.matrix as M4).e[0]).toBe(7);
    expect(a1.visible).toBe(false);
    expect(renders).toBe(1);
    fire(new Map());
    expect(a0.visible).toBe(false);
    ad.dispose();
  });
});

describe('A-Frame adapter', () => {
  it('registers a system and component that place target entities', () => {
    const defs: Record<string, Record<string, unknown>> = {};
    registerAFrame({ registerSystem: (n, d) => { defs[`s:${n}`] = d; }, registerComponent: (n, d) => { defs[`c:${n}`] = d; } });
    const sys = defs['s:fastar'] as Record<string, (...a: unknown[]) => unknown> & { schema: unknown };
    const { ar } = fakeTracker();
    const camEl = { object3D: new Obj() };
    const scene = {
      camera: { projectionMatrix: new M4(), projectionMatrixInverse: new M4(), near: 0.01, far: 100, el: camEl },
      canvas: { clientWidth: 400, clientHeight: 800 },
    };
    const system = Object.assign(Object.create(null), sys, { el: scene, ar, targets: new Map() });
    const target = { object3D: new Obj(), emit() {}, sceneEl: { systems: { fastar: system } } };
    const comp = defs['c:fastar-target'] as { init: (this: unknown) => void };
    comp.init.call({ el: target, data: { index: 2 } });
    expect(target.object3D.matrixAutoUpdate).toBe(false);
    (system.onFrame as (e: FrameEvent) => void)({ presentedFrame: {} as FrameEvent['presentedFrame'], poses: new Map([[2, pose(3)]]) });
    expect(target.object3D.visible).toBe(true);
    expect(target.object3D.matrix.e[5]).toBe(3);
    expect(camEl.object3D.matrix.e[0]).toBe(1);
    expect(camEl.object3D.matrix.e[13]).toBe(0);
  });
});
