// Lazy content for tracked targets: each target's content (a model, a video,
// anything the app loads) is loaded when the target is found or nearly
// found, kept while it is in view, and unloaded once it has been out of view
// a while and room is needed. How much stays loaded follows the device:
// more on fast phones and computers, less on slow ones.

export type ContentState = 'idle' | 'loading' | 'ready' | 'error';

export interface ContentLoadContext {
  /** Aborted when the load is no longer wanted (the target left before it finished, and room is needed). */
  signal: AbortSignal;
  /** The target's name (from the targets file). */
  name: string;
}

export interface ContentOptions<T> {
  /** Load target `index`'s content. */
  load(index: number, ctx: ContentLoadContext): Promise<T> | T;
  /** Free it (dispose GPU resources, release media). */
  unload?(index: number, value: T): void;
  /** The target came into view with its content loaded (e.g. play a video). */
  show?(index: number, value: T): void;
  /** The target went out of view (e.g. pause it). */
  hide?(index: number, value: T): void;
  /** Targets that have content (default: all). */
  targets?: number[];
  /** Weight of loaded content against the budget (default 1, e.g. a large model 3). */
  size?(index: number, value: T): number;
  /**
   * How much content to keep loaded (sum of sizes): a number, or 'auto'
   * (from the device's memory and cores, lowered while the frame rate
   * suffers). Content in view always stays, even over budget.
   */
  budget?: number | 'auto';
  /** Out of view this long (ms), content may be unloaded to make room (default 3000). */
  keepAlive?: number;
  /** Always loaded, from the start (e.g. the most likely target). */
  preload?: number[];
  /** A load started on a near miss is abandoned if the target is not found within this (ms, default 4000). */
  nearTimeout?: number;
  /** Loads at a time (default 2). */
  concurrency?: number;
  onError?(index: number, error: unknown): void;
}

/** What content management needs from the tracker (ImageTracker provides it). */
export interface ContentSource {
  readonly targets: ReadonlyArray<{ name: string }>;
  on(event: 'found' | 'lost' | 'near', cb: (index: number) => void): () => void;
  isShown(index: number): boolean;
  readonly stats: { fps: number; targets: ReadonlyArray<{ index: number; priority?: number }> };
  /** Throttling level (ImageTracker.performance); with it, the 'auto' budget follows the level instead of the frame rate. */
  readonly performance?: { level: number };
}

export type ContentEvent = 'ready' | 'unload' | 'error';

interface Entry<T> {
  index: number;
  state: ContentState;
  value: T | undefined;
  size: number;
  ctrl: AbortController | null;
  visible: boolean;
  /** Last time in view, or of the near miss that started the load (ms); -Infinity if never. */
  lastSeen: number;
  pinned: boolean;
  /** Loading or loaded on a near miss only. */
  speculative: boolean;
  queued: boolean;
  nearTimer: ReturnType<typeof setTimeout> | null;
  error: unknown;
}

/** Frame-rate samples (1 s apart) below this share of the best before shrinking the budget, and how many. */
const SLOW_SHARE = 0.7;
const SLOW_SAMPLES = 3;
const FAST_SHARE = 0.9;
const FAST_SAMPLES = 10;

/** Content budget for this device from its memory (GB) and cores; conservative where unknown (iOS). */
export function deviceContentBudget(nav: { deviceMemory?: number; hardwareConcurrency?: number } | undefined =
  typeof navigator !== 'undefined' ? (navigator as { deviceMemory?: number; hardwareConcurrency?: number }) : undefined): number {
  const mem = nav?.deviceMemory, cores = nav?.hardwareConcurrency;
  let b = mem === undefined ? 3 : mem <= 2 ? 2 : mem <= 4 ? 4 : 8;
  if (cores && cores <= 4) b = Math.min(b, 3);
  return b;
}

export class ContentManager<T> {
  private readonly entries = new Map<number, Entry<T>>();
  private readonly listeners: Record<ContentEvent, Set<(index: number) => void>> = { ready: new Set(), unload: new Set(), error: new Set() };
  private readonly offs: Array<() => void> = [];
  private readonly timers = new Set<ReturnType<typeof setTimeout>>();
  private readonly auto: boolean;
  private readonly base: number;
  private limit: number;
  private loading = 0;
  private fpsTimer: ReturnType<typeof setInterval> | null = null;
  private bestFps = 0;
  private slow = 0;
  private fast = 0;
  private disposed = false;
  private readonly keepAlive: number;
  private readonly nearTimeout: number;
  private readonly concurrency: number;

  constructor(private readonly src: ContentSource, private readonly o: ContentOptions<T>, private readonly now: () => number = () => Date.now()) {
    this.auto = o.budget === undefined || o.budget === 'auto';
    this.base = this.auto ? deviceContentBudget() : Math.max(0, o.budget as number);
    this.limit = this.base;
    this.keepAlive = o.keepAlive ?? 3000;
    this.nearTimeout = o.nearTimeout ?? 4000;
    this.concurrency = Math.max(1, o.concurrency ?? 2);
    const indices = o.targets ?? src.targets.map((_, i) => i);
    for (const i of indices) this.addEntry(i);
    this.offs.push(src.on('found', (i) => this.onFound(i)), src.on('lost', (i) => this.onLost(i)), src.on('near', (i) => this.onNear(i)));
    for (const e of this.entries.values()) this.sync(e);
    if (this.auto) this.fpsTimer = setInterval(() => this.sampleFps(), 1000);
    this.pump();
  }

  /** Manage target `index`'s content too (one not in `targets` at construction). */
  add(index: number, preload = false): void {
    const e = this.addEntry(index);
    if (!e || this.disposed) return;
    if (preload) e.pinned = true;
    this.sync(e);
    this.pump();
  }

  private addEntry(i: number): Entry<T> | null {
    if (i < 0 || i >= this.src.targets.length) return null;
    let e = this.entries.get(i);
    if (!e) {
      e = {
        index: i, state: 'idle', value: undefined, size: 0, ctrl: null, visible: false, lastSeen: -Infinity,
        pinned: !!this.o.preload?.includes(i), speculative: false, queued: false, nearTimer: null, error: undefined,
      };
      this.entries.set(i, e);
    }
    return e;
  }

  /** Catch up with a target already in view, or preloaded. */
  private sync(e: Entry<T>): void {
    if (!e.visible && this.src.isShown(e.index)) { e.visible = true; e.lastSeen = this.now(); }
    if (e.visible || e.pinned) this.request(e);
  }

  /** Loaded content of target `index`, if ready. */
  get(index: number): T | undefined {
    return this.entries.get(index)?.value;
  }

  state(index: number): ContentState {
    return this.entries.get(index)?.state ?? 'idle';
  }

  /** Current budget (sum of content sizes kept loaded). */
  get budget(): number {
    return this.limit;
  }

  /** Budget in use: loaded content sizes, plus one per load in progress. */
  get used(): number {
    let u = 0;
    for (const e of this.entries.values()) u += e.state === 'ready' ? e.size : e.state === 'loading' ? 1 : 0;
    return u;
  }

  /** Every target with content: its state and whether it is in view. */
  list(): Array<{ index: number; state: ContentState; visible: boolean }> {
    return [...this.entries.values()].map((e) => ({ index: e.index, state: e.state, visible: e.visible }));
  }

  on(event: ContentEvent, cb: (index: number) => void): () => void {
    this.listeners[event].add(cb);
    return () => this.listeners[event].delete(cb);
  }

  /** Stop: abort loads, unload everything, and stop following the tracker. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.offs.forEach((f) => f());
    if (this.fpsTimer) clearInterval(this.fpsTimer);
    this.timers.forEach((t) => clearTimeout(t));
    for (const e of this.entries.values()) {
      if (e.nearTimer) clearTimeout(e.nearTimer);
      if (e.state === 'loading') this.abort(e);
      else if (e.state === 'ready') this.unload(e);
    }
  }

  // --- tracker events ---

  private onFound(i: number): void {
    const e = this.entries.get(i);
    if (!e || this.disposed) return;
    e.visible = true;
    e.speculative = false;
    e.lastSeen = this.now();
    this.clearNear(e);
    if (e.state === 'ready') this.o.show?.(i, e.value as T);
    else this.request(e);
    this.pump();
  }

  private onLost(i: number): void {
    const e = this.entries.get(i);
    if (!e || this.disposed || !e.visible) return;
    e.visible = false;
    e.lastSeen = this.now();
    if (e.state === 'ready') this.o.hide?.(i, e.value as T);
    this.later(this.keepAlive + 1, () => this.enforce());
  }

  private onNear(i: number): void {
    const e = this.entries.get(i);
    if (!e || this.disposed || e.visible || e.state !== 'idle') return;
    e.speculative = true;
    e.lastSeen = this.now();
    this.request(e);
    this.clearNear(e);
    e.nearTimer = setTimeout(() => {
      e.nearTimer = null;
      if (e.visible || !e.speculative) return;
      // Not found after all: abandon the load, or leave the content to the usual eviction.
      if (e.state === 'loading') this.abort(e);
      e.queued = false;
      e.lastSeen = this.now();
      this.later(this.keepAlive + 1, () => this.enforce());
      this.pump();
    }, this.nearTimeout);
    this.pump();
  }

  // --- loading ---

  private request(e: Entry<T>): void {
    if (e.state === 'loading' || e.state === 'ready') return;
    e.queued = true;
  }

  /** Rank for the load queue: in view first (most important first), then preloads, then near misses. */
  private rank(e: Entry<T>): number {
    if (e.visible) {
      const p = this.src.stats.targets.find((x) => x.index === e.index)?.priority ?? 0;
      return 3 + Math.min(1, p);
    }
    return e.pinned ? 2 : 1;
  }

  private pump(): void {
    if (this.disposed) return;
    for (;;) {
      if (this.loading >= this.concurrency) return;
      let next: Entry<T> | null = null;
      for (const e of this.entries.values()) {
        if (!e.queued || e.state === 'loading' || e.state === 'ready') continue;
        if (!next || this.rank(e) > this.rank(next)) next = e;
      }
      if (!next) return;
      // Room for it: make some if needed. Content in view (or preloaded)
      // loads even over budget; a near miss only into free room.
      if (this.used + 1 > this.limit) this.evict(1, !next.visible && !next.pinned);
      if (this.used + 1 > this.limit && !next.visible && !next.pinned) {
        next.queued = false;
        next.speculative = false;
        continue;
      }
      this.start(next);
    }
  }

  private start(e: Entry<T>): void {
    e.queued = false;
    e.state = 'loading';
    e.error = undefined;
    const ctrl = (e.ctrl = new AbortController());
    this.loading++;
    const name = this.src.targets[e.index]?.name ?? String(e.index);
    Promise.resolve()
      .then(() => this.o.load(e.index, { signal: ctrl.signal, name }))
      .then((value) => {
        if (ctrl.signal.aborted || this.disposed) {
          // Finished anyway: free it.
          try { this.o.unload?.(e.index, value); } catch { /* app cleanup failed */ }
          return;
        }
        this.loading--;
        e.ctrl = null;
        e.state = 'ready';
        e.value = value;
        e.size = Math.max(0, this.o.size?.(e.index, value) ?? 1);
        this.emit('ready', e.index);
        if (e.visible) this.o.show?.(e.index, value);
        this.enforce();
        this.pump();
      }, (err: unknown) => {
        if (ctrl.signal.aborted || this.disposed) return;
        this.loading--;
        e.ctrl = null;
        e.state = 'error';
        e.error = err;
        this.o.onError?.(e.index, err);
        this.emit('error', e.index);
        this.pump();
      });
  }

  private abort(e: Entry<T>): void {
    if (e.state !== 'loading') return;
    e.ctrl?.abort();
    e.ctrl = null;
    e.state = 'idle';
    e.speculative = false;
    this.loading--;
  }

  private unload(e: Entry<T>): void {
    const v = e.value as T;
    e.state = 'idle';
    e.value = undefined;
    e.size = 0;
    e.speculative = false;
    try { this.o.unload?.(e.index, v); } catch (err) { this.o.onError?.(e.index, err); }
    this.emit('unload', e.index);
  }

  // --- budget ---

  /**
   * Free room until `need` more fits: unload content out of view for longer
   * than keepAlive, least recently seen first; then (if `soft` is false)
   * abort loads for targets out of view.
   */
  private evict(need: number, soft = false): void {
    const t = this.now();
    const out = [...this.entries.values()]
      .filter((e) => e.state === 'ready' && !e.visible && !e.pinned && t - e.lastSeen >= this.keepAlive)
      .sort((a, b) => a.lastSeen - b.lastSeen);
    for (const e of out) {
      if (this.used + need <= this.limit) return;
      this.unload(e);
    }
    if (soft) return;
    const pending = [...this.entries.values()]
      .filter((e) => e.state === 'loading' && !e.visible && !e.pinned)
      .sort((a, b) => a.lastSeen - b.lastSeen);
    for (const e of pending) {
      if (this.used + need <= this.limit) return;
      this.abort(e);
    }
  }

  private enforce(): void {
    if (this.disposed) return;
    if (this.used > this.limit) this.evict(0);
  }

  /** 'auto': lower the budget while the device is throttled (or the frame rate suffers), raise it back once it recovers. */
  private sampleFps(): void {
    const perf = this.src.performance;
    if (perf) {
      let visible = 0;
      for (const e of this.entries.values()) if (e.visible) visible++;
      const want = Math.min(this.base, Math.max(1, visible, this.base - perf.level));
      if (want < this.limit) {
        this.limit = want;
        this.enforce();
      } else if (want > this.limit) {
        this.limit = want;
        this.pump();
      }
      return;
    }
    const fps = this.src.stats.fps;
    if (!(fps > 0)) return;
    this.bestFps = Math.max(fps, this.bestFps * 0.995);
    if (fps < SLOW_SHARE * this.bestFps) { this.slow++; this.fast = 0; }
    else if (fps >= FAST_SHARE * this.bestFps) { this.fast++; this.slow = 0; }
    else { this.slow = 0; this.fast = 0; }
    let visible = 0;
    for (const e of this.entries.values()) if (e.visible) visible++;
    if (this.slow >= SLOW_SAMPLES && this.limit > Math.max(1, visible)) {
      this.limit--;
      this.slow = 0;
      this.enforce();
    } else if (this.fast >= FAST_SAMPLES && this.limit < this.base) {
      this.limit++;
      this.fast = 0;
      this.pump();
    }
  }

  private later(ms: number, f: () => void): void {
    const t = setTimeout(() => { this.timers.delete(t); f(); }, ms);
    this.timers.add(t);
  }

  private clearNear(e: Entry<T>): void {
    if (e.nearTimer) clearTimeout(e.nearTimer);
    e.nearTimer = null;
  }

  private emit(event: ContentEvent, index: number): void {
    this.listeners[event].forEach((cb) => cb(index));
  }
}
