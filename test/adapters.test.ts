import { describe, expect, it } from 'vitest';
import { ThreeAdapter, disposeObject } from '../src/adapters/three';
import { registerAFrame, type AFrameContent } from '../src/adapters/aframe';
import type { FrameEvent, ImageTracker, Pose } from '../src/shell/image-tracker';

class M4 {
  e = new Float64Array(16);
  fromArray(a: ArrayLike<number>) { for (let i = 0; i < 16; i++) this.e[i] = a[i]; return this; }
  copy(m: M4) { this.e.set(m.e); return this; }
  invert() { return this; }
  identity() { this.e.fill(0); this.e[0] = this.e[5] = this.e[10] = this.e[15] = 1; return this; }
}
class Obj {
  matrix = new M4(); matrixAutoUpdate = true; matrixWorldNeedsUpdate = false; visible = true;
  children: unknown[] = [];
  add(o: unknown) { this.children.push(o); }
  remove(o: unknown) { this.children = this.children.filter((c) => c !== o); }
}

type Ev = 'found' | 'lost' | 'near';

function fakeTracker() {
  let cb: ((e: FrameEvent) => void) | null = null;
  const P = Float32Array.from({ length: 16 }, (_, i) => i + 1);
  const ls: Record<Ev, Set<(i: number) => void>> = { found: new Set(), lost: new Set(), near: new Set() };
  const shown = new Set<number>();
  const ar = {
    onFrame: (f: (e: FrameEvent) => void) => { cb = f; return () => { cb = null; }; },
    get projectionMatrix() { return P; },
    getProjectionMatrix: () => P,
    targets: [0, 1, 2].map((i) => ({ name: `t${i}`, physicalWidth: 0.3, width: 640, height: 360 })),
    stats: { fps: 30, targets: [] },
    on: (e: Ev, f: (i: number) => void) => { ls[e].add(f); return () => ls[e].delete(f); },
    isShown: (i: number) => shown.has(i),
  } as unknown as ImageTracker;
  const fire = (poses: Map<number, Pose>) => cb?.({ presentedFrame: {} as FrameEvent['presentedFrame'], poses });
  const emit = (e: Ev, i: number) => {
    if (e === 'found') shown.add(i);
    if (e === 'lost') shown.delete(i);
    ls[e].forEach((f) => f(i));
  };
  return { ar, fire, P, emit };
}

const flush = async () => { for (let i = 0; i < 5; i++) await Promise.resolve(); };

/** A mesh-like object whose geometry, material and texture record disposal. */
function mesh(log: string[], name: string) {
  const o = Object.assign(new Obj(), {
    geometry: { dispose: () => log.push(`${name}.geometry`) },
    material: { map: { isTexture: true, dispose: () => log.push(`${name}.texture`) }, color: 1, dispose: () => log.push(`${name}.material`) },
    traverse(f: (o: unknown) => void) { f(o); },
  });
  return o;
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

  it('loads content onto a target\'s anchor when it is found, and removes and disposes it when unloaded', async () => {
    const { ar, emit } = fakeTracker();
    const camera = Object.assign(new Obj(), { projectionMatrix: new M4(), near: 0.01, far: 100 });
    const log: string[] = [];
    const shows: string[] = [];
    const ad = new ThreeAdapter(ar, { camera, scene: { add() {} }, THREE: { Group: Obj }, content: { budget: 1, keepAlive: 0 } });
    const m0 = mesh(log, 'm0'), m1 = mesh(log, 'm1');
    const loads: number[] = [];
    ad.content(0, () => { loads.push(0); return m0; }, { show: () => shows.push('show 0'), hide: () => shows.push('hide 0'), unload: () => log.push('m0.app') });
    ad.content(1, async ({ name }) => { loads.push(1); expect(name).toBe('t1'); return m1; });
    expect(() => ad.content(7, () => m0)).toThrow(/no target 7/);
    await flush();
    expect(loads).toEqual([]);
    expect(ad.contents!.state(0)).toBe('idle');
    emit('found', 0);
    await flush();
    expect(loads).toEqual([0]);
    const a0 = ad.anchor(0) as unknown as Obj;
    expect(a0.children).toEqual([m0]);
    expect(shows).toEqual(['show 0']);
    emit('lost', 0);
    expect(shows).toEqual(['show 0', 'hide 0']);
    // Target 1 needs the room: target 0's content is removed and disposed.
    emit('found', 1);
    await flush();
    expect(a0.children).toEqual([]);
    expect(log).toEqual(['m0.geometry', 'm0.texture', 'm0.material', 'm0.app']);
    expect((ad.anchor(1) as unknown as Obj).children).toEqual([m1]);
    ad.dispose();
    expect(log.slice(4)).toEqual(['m1.geometry', 'm1.texture', 'm1.material']);
  });

  it('disposes plain objects and arrays of materials', () => {
    const log: string[] = [];
    disposeObject({ geometry: { dispose: () => log.push('g') }, material: [{ dispose: () => log.push('a') }, { dispose: () => log.push('b') }] });
    expect(log).toEqual(['g', 'a', 'b']);
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

  /** Just enough DOM for fastar-content. */
  class Node {
    attrs = new Map<string, unknown>();
    ls = new Map<string, Set<(e?: unknown) => void>>();
    children: Node[] = [];
    parentNode: Node | null = null;
    object3D = { traverse: (f: (o: unknown) => void) => f({ geometry: { dispose: () => this.log.push(`${this.tag}.dispose`) } }) };
    [k: string]: unknown;
    constructor(readonly tag: string, readonly log: string[]) {}
    setAttribute(n: string, v: unknown) { this.attrs.set(n, v); }
    removeAttribute(n: string) { this.attrs.delete(n); this.log.push(`${this.tag}.removeAttribute ${n}`); }
    addEventListener(n: string, f: (e?: unknown) => void) { if (!this.ls.has(n)) this.ls.set(n, new Set()); this.ls.get(n)!.add(f); }
    removeEventListener(n: string, f: (e?: unknown) => void) { this.ls.get(n)?.delete(f); }
    dispatch(n: string) { [...(this.ls.get(n) ?? [])].forEach((f) => f()); }
    appendChild(c: Node) { this.children.push(c); c.parentNode = this; }
    removeChild(c: Node) { this.children = this.children.filter((x) => x !== c); c.parentNode = null; }
    play() { this.log.push(`${this.tag}.play`); return Promise.resolve(); }
    pause() { this.log.push(`${this.tag}.pause`); }
    load() { this.log.push(`${this.tag}.load`); }
  }

  it('fastar-content loads a model or a video onto its target when needed, plays videos in view, and unloads', async () => {
    const defs: Record<string, Record<string, unknown>> = {};
    registerAFrame({ registerSystem: (n, d) => { defs[`s:${n}`] = d; }, registerComponent: (n, d) => { defs[`c:${n}`] = d; } });
    const sys = defs['s:fastar'] as Record<string, (...a: unknown[]) => unknown>;
    const { ar, emit } = fakeTracker();
    const log: string[] = [];
    const created: Node[] = [];
    const doc = { createElement: (tag: string) => { const n = new Node(tag, log); created.push(n); return n; } };
    const events: string[] = [];
    const system = Object.assign(Object.create(null), sys, {
      el: {}, ar, targets: new Map(), contents: new Map(), content: null,
      data: { contentBudget: '1', keepAlive: 0 },
    });
    const comp = defs['c:fastar-content'] as Record<string, (this: unknown, ...a: unknown[]) => unknown>;
    const entity = (index: number, src: string) => {
      const el = Object.assign(new Node('a-entity', log), {
        emit: (n: string) => events.push(`${index}:${n}`), ownerDocument: doc, sceneEl: { systems: { fastar: system } },
        components: { 'fastar-target': { data: { index } } },
      });
      const c = { el, data: { src, type: 'auto', index: -1 }, load: comp.load };
      comp.init.call(c);
      return { el, c };
    };
    const model = entity(0, 'robot.glb');
    // Registered before the tracker starts, and after.
    (system.startContent as (a: unknown) => void)(ar);
    const video = entity(1, 'clip.mp4?v=2');
    expect([...system.contents.keys()]).toEqual([0, 1]);

    emit('found', 0);
    await flush();
    const child = model.el.children[0] as Node;
    expect(child.tag).toBe('a-entity');
    expect(child.attrs.get('gltf-model')).toBe('robot.glb');
    child.dispatch('model-loaded');
    await flush();
    expect(events).toEqual(['0:contentReady']);

    emit('lost', 0);
    emit('found', 1);
    await flush();
    const v = created.find((n) => n.tag === 'video')!;
    expect(v.src).toBe('clip.mp4?v=2');
    expect(v.muted).toBe(true);
    expect(v.loop).toBe(true);
    // The model made room (budget 1, out of view).
    expect(model.el.children).toEqual([]);
    expect(log).toContain('a-entity.dispose');
    expect(events).toContain('0:contentUnloaded');
    v.dispatch('loadeddata');
    await flush();
    const plane = video.el.children[0] as Node;
    expect(plane.tag).toBe('a-plane');
    expect(plane.attrs.get('width')).toBe(0.3);
    expect(plane.attrs.get('height')).toBeCloseTo(0.16875, 9);
    expect((plane.attrs.get('material') as { src: unknown }).src).toBe(v);
    expect(log).toContain('video.play');
    emit('lost', 1);
    expect(log.at(-1)).toBe('video.pause');
    (system.content as { dispose(): void }).dispose();
    expect(video.el.children).toEqual([]);
    expect(log).toContain('video.removeAttribute src');
    // Unregistering only removes the component's own entry.
    comp.remove.call(video.c);
    expect([...system.contents.keys()]).toEqual([0]);
  });

  it('fastar-content gives up a load that is no longer wanted, and reports failures', async () => {
    const defs: Record<string, Record<string, unknown>> = {};
    registerAFrame({ registerSystem: (n, d) => { defs[`s:${n}`] = d; }, registerComponent: (n, d) => { defs[`c:${n}`] = d; } });
    const comp = defs['c:fastar-content'] as { load(this: unknown, s: AbortSignal, meta: unknown): Promise<AFrameContent> };
    const log: string[] = [];
    const created: Node[] = [];
    const doc = { createElement: (tag: string) => { const n = new Node(tag, log); created.push(n); return n; } };
    const el = Object.assign(new Node('a-entity', log), { ownerDocument: doc });
    const meta = { physicalWidth: 0.2, width: 4, height: 3 };
    const ctrl = new AbortController();
    const p = comp.load.call({ el, data: { src: 'm.gltf', type: 'auto', index: 0 } }, ctrl.signal, meta);
    expect(el.children.length).toBe(1);
    ctrl.abort();
    await expect(p).rejects.toMatchObject({ name: 'AbortError' });
    expect(el.children.length).toBe(0);
    // A video (by type, whatever the URL) that fails to load: rejected, and its element released.
    const q = comp.load.call({ el, data: { src: 'stream', type: 'video', index: 0 } }, new AbortController().signal, meta);
    const video = created.find((n) => n.tag === 'video')!;
    video.dispatch('error');
    await expect(q).rejects.toThrow(/could not load/);
    expect(log).toContain('video.removeAttribute src');
    expect(el.children.length).toBe(0);
    // A model that fails.
    const r = comp.load.call({ el, data: { src: 'bad.glb', type: 'gltf', index: 0 } }, new AbortController().signal, meta);
    (el.children[0] as Node).dispatch('model-error');
    await expect(r).rejects.toThrow(/could not load/);
    expect(el.children.length).toBe(0);
  });
});
