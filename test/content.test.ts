import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ContentManager, deviceContentBudget, type ContentOptions, type ContentSource } from '../src/shell/content';
import { ImageTracker } from '../src/shell/image-tracker';
import { encodeTargets } from '../src/target/format';
import { compileTarget } from '../src/target/compiler';
import { makePoster } from '../src/sim/poster';
import type { ResultMessage } from '../src/worker/protocol';

type Ev = 'found' | 'lost' | 'near';

/** A tracker stand-in the test drives by hand. */
class FakeSource implements ContentSource {
  readonly targets = Array.from({ length: 6 }, (_, i) => ({ name: `t${i}` }));
  readonly stats = { fps: 30, targets: [] as Array<{ index: number; priority?: number }> };
  readonly shown = new Set<number>();
  private readonly ls: Record<Ev, Set<(i: number) => void>> = { found: new Set(), lost: new Set(), near: new Set() };
  on(e: Ev, cb: (i: number) => void) { this.ls[e].add(cb); return () => this.ls[e].delete(cb); }
  isShown(i: number) { return this.shown.has(i); }
  found(i: number) { this.shown.add(i); this.ls.found.forEach((f) => f(i)); }
  lost(i: number) { this.shown.delete(i); this.ls.lost.forEach((f) => f(i)); }
  near(i: number) { this.ls.near.forEach((f) => f(i)); }
  get listeners() { return this.ls.found.size + this.ls.lost.size + this.ls.near.size; }
}

/** Loads that resolve when the test says so. */
function loader() {
  const pending = new Map<number, { resolve: (v: string) => void; reject: (e: unknown) => void; signal: AbortSignal }>();
  const log: string[] = [];
  const o: ContentOptions<string> = {
    load: (i, { signal, name }) => new Promise<string>((resolve, reject) => {
      log.push(`load ${i} ${name}`);
      pending.set(i, { resolve, reject, signal });
    }),
    unload: (i, v) => log.push(`unload ${i} ${v}`),
    show: (i) => log.push(`show ${i}`),
    hide: (i) => log.push(`hide ${i}`),
  };
  const finish = async (i: number) => { pending.get(i)!.resolve(`v${i}`); pending.delete(i); await flush(); };
  return { o, pending, log, finish };
}

const flush = async () => { for (let i = 0; i < 5; i++) await Promise.resolve(); };

describe('content manager', () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });

  it('loads content when its target is found, shows it, and hides it when lost', async () => {
    const src = new FakeSource(), L = loader();
    const m = new ContentManager(src, { ...L.o, budget: 4 }, () => Date.now());
    expect(m.state(0)).toBe('idle');
    src.found(0);
    await flush();
    expect(m.state(0)).toBe('loading');
    expect(L.log).toEqual(['load 0 t0']);
    await L.finish(0);
    expect(m.state(0)).toBe('ready');
    expect(m.get(0)).toBe('v0');
    expect(L.log).toEqual(['load 0 t0', 'show 0']);
    src.lost(0);
    expect(L.log.at(-1)).toBe('hide 0');
    // Found again: shown straight away, not reloaded.
    src.found(0);
    expect(L.log.slice(-1)).toEqual(['show 0']);
    expect(L.log.filter((x) => x.startsWith('load')).length).toBe(1);
    m.dispose();
  });

  it('keeps content out of view while there is room, and unloads the least recently seen when room is needed', async () => {
    const src = new FakeSource(), L = loader();
    const m = new ContentManager(src, { ...L.o, budget: 2, keepAlive: 1000 }, () => Date.now());
    const unloaded: number[] = [];
    m.on('unload', (i) => unloaded.push(i));
    for (const i of [0, 1]) { src.found(i); await flush(); await L.finish(i); }
    src.lost(0);
    vi.advanceTimersByTime(500);
    src.lost(1);
    // Within room: nothing unloaded, even long after.
    vi.advanceTimersByTime(5000);
    expect(unloaded).toEqual([]);
    expect(m.used).toBe(2);
    // A third target needs room: the one out of view longest goes.
    src.found(2);
    await flush();
    expect(unloaded).toEqual([0]);
    expect(m.state(0)).toBe('idle');
    expect(m.state(1)).toBe('ready');
    await L.finish(2);
    expect(m.used).toBe(2);
    m.dispose();
  });

  it('never unloads content in view or within keepAlive: goes over budget instead', async () => {
    const src = new FakeSource(), L = loader();
    const m = new ContentManager(src, { ...L.o, budget: 1, keepAlive: 3000 }, () => Date.now());
    src.found(0); await flush(); await L.finish(0);
    src.found(1); await flush(); await L.finish(1);
    expect(m.used).toBe(2);
    expect(m.state(0)).toBe('ready');
    src.lost(0);
    vi.advanceTimersByTime(1000);
    src.found(2); await flush(); await L.finish(2);
    // 0 is still within keepAlive.
    expect(m.state(0)).toBe('ready');
    vi.advanceTimersByTime(2500);
    // Past keepAlive and over budget: unloaded by the lost timer.
    expect(m.state(0)).toBe('idle');
    expect(m.state(1)).toBe('ready');
    expect(m.state(2)).toBe('ready');
    m.dispose();
  });

  it('starts loading on a near miss, and abandons it if the target is not found in time', async () => {
    const src = new FakeSource(), L = loader();
    const m = new ContentManager(src, { ...L.o, budget: 4, nearTimeout: 2000 }, () => Date.now());
    src.near(3);
    await flush();
    expect(m.state(3)).toBe('loading');
    const signal = L.pending.get(3)!.signal;
    vi.advanceTimersByTime(2001);
    expect(signal.aborted).toBe(true);
    expect(m.state(3)).toBe('idle');
    // A load that finishes after its abort is freed at once.
    L.pending.get(3)!.resolve('late');
    await flush();
    expect(L.log).toContain('unload 3 late');
    expect(m.state(3)).toBe('idle');
    m.dispose();
  });

  it('a near miss followed by the target being found: the load carries on and the content shows', async () => {
    const src = new FakeSource(), L = loader();
    const m = new ContentManager(src, { ...L.o, budget: 4, nearTimeout: 2000 }, () => Date.now());
    src.near(1);
    await flush();
    vi.advanceTimersByTime(500);
    src.found(1);
    vi.advanceTimersByTime(3000);
    expect(m.state(1)).toBe('loading');
    await L.finish(1);
    expect(L.log).toEqual(['load 1 t1', 'show 1']);
    m.dispose();
  });

  it('only prefetches near misses into free room', async () => {
    const src = new FakeSource(), L = loader();
    const m = new ContentManager(src, { ...L.o, budget: 1 }, () => Date.now());
    src.found(0); await flush(); await L.finish(0);
    src.near(1);
    await flush();
    expect(m.state(1)).toBe('idle');
    expect(m.state(0)).toBe('ready');
    m.dispose();
  });

  it('aborts loads for targets that left when the room is needed for one in view', async () => {
    const src = new FakeSource(), L = loader();
    const m = new ContentManager(src, { ...L.o, budget: 1, concurrency: 2 }, () => Date.now());
    src.found(0);
    await flush();
    src.lost(0);
    const signal = L.pending.get(0)!.signal;
    src.found(1);
    await flush();
    expect(signal.aborted).toBe(true);
    expect(m.state(0)).toBe('idle');
    expect(m.state(1)).toBe('loading');
    m.dispose();
  });

  it('loads in view first, most prominent first, a few at a time', async () => {
    const src = new FakeSource(), L = loader();
    src.stats.targets = [{ index: 0, priority: 0.1 }, { index: 1, priority: 0.2 }, { index: 2, priority: 0.9 }];
    const m = new ContentManager(src, { ...L.o, budget: 8, concurrency: 1, preload: [5] }, () => Date.now());
    // The preload starts first (nothing else is wanted yet).
    await flush();
    expect(L.log).toEqual(['load 5 t5']);
    src.near(4);
    src.found(0); src.found(1); src.found(2);
    await L.finish(5);
    await L.finish(2);
    await L.finish(1);
    await L.finish(0);
    await L.finish(4);
    expect(L.log.filter((x) => x.startsWith('load'))).toEqual(['load 5 t5', 'load 2 t2', 'load 1 t1', 'load 0 t0', 'load 4 t4']);
    m.dispose();
  });

  it('keeps preloaded content however long it is out of view', async () => {
    const src = new FakeSource(), L = loader();
    const m = new ContentManager(src, { ...L.o, budget: 1, keepAlive: 10, preload: [0] }, () => Date.now());
    await flush(); await L.finish(0);
    src.found(1); await flush(); await L.finish(1);
    vi.advanceTimersByTime(10000);
    expect(m.state(0)).toBe('ready');
    m.dispose();
  });

  it('reports errors and retries on the next sighting', async () => {
    const src = new FakeSource(), L = loader();
    const errors: number[] = [];
    const m = new ContentManager(src, { ...L.o, budget: 2, onError: (i) => errors.push(i) }, () => Date.now());
    src.found(0);
    await flush();
    L.pending.get(0)!.reject(new Error('404'));
    await flush();
    expect(m.state(0)).toBe('error');
    expect(errors).toEqual([0]);
    src.lost(0);
    src.found(0);
    await flush();
    expect(m.state(0)).toBe('loading');
    m.dispose();
  });

  it('only manages the targets given, and more added later; picks up targets already in view', async () => {
    const src = new FakeSource(), L = loader();
    src.shown.add(2);
    const m = new ContentManager(src, { ...L.o, budget: 4, targets: [2] }, () => Date.now());
    await flush();
    expect(L.log).toEqual(['load 2 t2']);
    src.found(1);
    await flush();
    expect(m.state(1)).toBe('idle');
    m.add(1);
    await flush();
    expect(m.state(1)).toBe('loading');
    expect(m.list().map((e) => e.index).sort()).toEqual([1, 2]);
    m.add(99);
    expect(m.list().length).toBe(2);
    m.dispose();
  });

  it('weighs content by size', async () => {
    const src = new FakeSource(), L = loader();
    const m = new ContentManager(src, { ...L.o, budget: 3, keepAlive: 0, size: (i) => (i === 0 ? 3 : 1) }, () => Date.now());
    src.found(0); await flush(); await L.finish(0);
    expect(m.used).toBe(3);
    src.lost(0);
    vi.advanceTimersByTime(10);
    src.found(1); await flush();
    expect(m.state(0)).toBe('idle');
    m.dispose();
  });

  it('auto budget: from the device, lowered while the frame rate suffers and raised back after', async () => {
    expect(deviceContentBudget({ deviceMemory: 8, hardwareConcurrency: 8 })).toBe(8);
    expect(deviceContentBudget({ deviceMemory: 4, hardwareConcurrency: 8 })).toBe(4);
    expect(deviceContentBudget({ deviceMemory: 2, hardwareConcurrency: 8 })).toBe(2);
    expect(deviceContentBudget({ deviceMemory: 8, hardwareConcurrency: 4 })).toBe(3);
    // iOS reports no memory.
    expect(deviceContentBudget({ hardwareConcurrency: 6 })).toBe(3);
    const src = new FakeSource(), L = loader();
    const m = new ContentManager(src, { ...L.o, budget: 'auto', keepAlive: 0 }, () => Date.now());
    const base = m.budget;
    expect(base).toBe(deviceContentBudget());
    for (const i of [0, 1]) { src.found(i); await flush(); await L.finish(i); }
    src.lost(1);
    vi.advanceTimersByTime(5000);
    src.stats.fps = 12;
    vi.advanceTimersByTime(3000);
    expect(m.budget).toBe(base - 1);
    vi.advanceTimersByTime(30000);
    // Never below what is in view.
    expect(m.budget).toBeGreaterThanOrEqual(1);
    expect(m.state(0)).toBe('ready');
    if (base - 1 < 2) expect(m.state(1)).toBe('idle');
    src.stats.fps = 30;
    vi.advanceTimersByTime(60000);
    expect(m.budget).toBe(base);
    m.dispose();
  });

  it('auto budget follows the tracker\'s throttling level where it has one', async () => {
    const src = new (class extends FakeSource { performance = { level: 0 }; })(), L = loader();
    const m = new ContentManager(src, { ...L.o, budget: 'auto', keepAlive: 0 }, () => Date.now());
    const base = m.budget;
    src.found(0);
    await flush();
    await L.finish(0);
    src.performance.level = 2;
    // The frame rate is ignored: the level already accounts for it.
    src.stats.fps = 5;
    vi.advanceTimersByTime(1000);
    expect(m.budget).toBe(Math.max(1, base - 2));
    src.performance.level = 3;
    vi.advanceTimersByTime(1000);
    expect(m.budget).toBe(Math.max(1, base - 3));
    // Never below what is in view.
    expect(m.state(0)).toBe('ready');
    src.performance.level = 0;
    vi.advanceTimersByTime(1000);
    expect(m.budget).toBe(base);
    m.dispose();
  });

  it('dispose aborts loads, unloads content and stops listening', async () => {
    const src = new FakeSource(), L = loader();
    const m = new ContentManager(src, { ...L.o, budget: 4 }, () => Date.now());
    src.found(0); await flush(); await L.finish(0);
    src.found(1); await flush();
    const signal = L.pending.get(1)!.signal;
    m.dispose();
    expect(signal.aborted).toBe(true);
    expect(L.log).toContain('unload 0 v0');
    expect(src.listeners).toBe(0);
    src.found(2);
    await flush();
    expect(L.log.some((x) => x === 'load 2 t2')).toBe(false);
  });
});

describe('ImageTracker target events and helpers', () => {
  const buf = encodeTargets([11, 23, 37].map((s, i) => compileTarget(makePoster(320, 240, s), { physicalWidth: 0.2, name: `p${i}` })));
  const result = (o: Partial<ResultMessage>): ResultMessage => ({
    type: 'result', frameId: 1, t: 0, doneAt: 0, targets: [], found: [], lost: [], timings: { pyramid: 0, track: 0, fusion: 0, total: 0, klt: 0, ncc: 0, solve: 0 },
    budget: 12, load: { stride: 1, overload: 0, interval: 33.3, acquireBusy: 0, prep: 0 }, snapshots: [], td: 0, tdCalibrated: false, clocks: [], ...o,
  });

  it('tracks every target by default, finds targets by name, and reports near misses at most once a second', async () => {
    const ar = await ImageTracker.create({ targets: buf.slice(0) });
    expect((ar as unknown as { o: { maxTrack: number } }).o.maxTrack).toBe(3);
    expect(ar.targetIndex('p1')).toBe(1);
    expect(ar.targetIndex('nope')).toBe(-1);
    const near: number[] = [];
    ar.on('near', (i) => near.push(i));
    const feed = (m: ResultMessage) => (ar as unknown as { onResultMsg(m: ResultMessage): void }).onResultMsg(m);
    feed(result({ near: [2], acquisition: true }));
    feed(result({ near: [2], acquisition: true }));
    expect(near).toEqual([2]);
    expect(ar.isShown(2)).toBe(false);
    expect(ar.stats.budgetMs).toBe(0);
    feed(result({}));
    expect(ar.stats.budgetMs).toBe(12);
  });

  it('honours maxTrack and hands out a content manager wired to its events', async () => {
    const ar = await ImageTracker.create({ targets: buf.slice(0), maxTrack: 1 });
    expect((ar as unknown as { o: { maxTrack: number } }).o.maxTrack).toBe(1);
    const loads: number[] = [];
    const m = ar.content({ load: (i) => { loads.push(i); return i; }, budget: 2 });
    (ar as unknown as { listeners: { found: Set<(i: number) => void> } }).listeners.found.forEach((cb) => cb(1));
    await flush();
    expect(loads).toEqual([1]);
    expect(m.get(1)).toBe(1);
    m.dispose();
  });

  it('reports its throttling state, with the tracking load from the worker', async () => {
    const ar = await ImageTracker.create({ targets: buf.slice(0) });
    expect(ar.performance.level).toBe(0);
    expect(ar.performance.bottleneck).toBe('none');
    const off = ar.on('performance', () => {});
    expect(typeof off).toBe('function');
    off();
    const feed = (m: ResultMessage) => (ar as unknown as { onResultMsg(m: ResultMessage): void }).onResultMsg(m);
    feed(result({ load: { stride: 2, overload: 0.4, interval: 33.3, acquireBusy: 0.25, prep: 1 } }));
    expect(ar.performance.stride).toBe(2);
    expect(ar.performance.trackingOverload).toBe(0.4);
    expect(ar.performance.acquireBusy).toBe(0.25);
    // The copy path sends the worker one frame in `stride`.
    expect((ar as unknown as { stride: number }).stride).toBe(2);
  });
});
