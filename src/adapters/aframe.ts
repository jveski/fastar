// A-Frame integration (no dependency on A-Frame; pass the global AFRAME).
//
//   <a-scene fastar="targets: targets.bin" vr-mode-ui="enabled: false">
//     <a-camera look-controls="enabled: false" wasd-controls="enabled: false"></a-camera>
//     <a-entity fastar-target="index: 0"><a-box scale="0.05 0.05 0.05"></a-box></a-entity>
//     <a-entity fastar-target="index: 1" fastar-content="src: robot.glb"></a-entity>
//     <a-entity fastar-target="index: 2" fastar-content="src: clip.mp4"></a-entity>
//   </a-scene>
//
// Every target in the file is tracked (maxTrack limits it). fastar-content
// loads a glTF model or a video onto its target only once the target is
// found (or nearly found), plays videos only while their target is in view,
// and unloads content that has been out of view a while when room is needed
// (the scene's contentBudget and keepAlive).
//
// Target entities emit `targetFound`, `targetLost` and `targetNear`, and
// `contentReady` / `contentUnloaded`. The scene emits `fastar-ready`,
// `fastar-started`, `fastar-error`, and `fastar-performance` (detail: the
// tracker's PerformanceState) when the device's throttling level changes.
import { ImageTracker, type FrameEvent } from '../shell/image-tracker';
import { MotionSensors } from '../shell/sensors';
import { ContentManager } from '../shell/content';
import type { TargetMeta } from '../target/format';
import { disposeObject } from './three';

interface AFrameLike {
  registerSystem(name: string, def: Record<string, unknown>): void;
  registerComponent(name: string, def: Record<string, unknown>): void;
}

/** The DOM bits used to build content (an element, or the document). */
interface NodeLike {
  setAttribute(name: string, value: unknown): void;
  removeAttribute?(name: string): void;
  addEventListener(name: string, f: (e?: unknown) => void, o?: unknown): void;
  removeEventListener?(name: string, f: (e?: unknown) => void): void;
  appendChild?(c: unknown): unknown;
  parentNode?: { removeChild(c: unknown): unknown } | null;
  object3D?: unknown;
}

interface DocLike {
  createElement(tag: string): NodeLike & Record<string, unknown>;
}

interface El extends NodeLike {
  object3D: { matrix: { fromArray(a: ArrayLike<number>): unknown; identity?(): unknown }; matrixAutoUpdate: boolean; matrixWorldNeedsUpdate: boolean; visible: boolean };
  emit(name: string, detail?: unknown): void;
  sceneEl: SceneEl;
  addEventListener(n: string, f: () => void, o?: unknown): void;
  ownerDocument?: DocLike;
  components?: Record<string, { data?: { index?: number } } | undefined>;
}

interface SceneEl extends El {
  camera: { projectionMatrix: { fromArray(a: ArrayLike<number>): unknown }; projectionMatrixInverse?: { copy(m: unknown): { invert(): unknown } }; near: number; far: number; el?: El };
  canvas?: HTMLCanvasElement;
  hasLoaded: boolean;
  systems: Record<string, FastarSystem>;
}

/** Loaded content of one target. */
export interface AFrameContent {
  show(): void;
  hide(): void;
  unload(): void;
}

interface ContentComponent {
  el: El;
  data: { src: string; type: 'auto' | 'gltf' | 'video'; index: number };
  load(signal: AbortSignal, meta: TargetMeta): Promise<AFrameContent>;
}

interface FastarSystem {
  el: SceneEl;
  data: { targets: string; maxTrack: number; autoStart: boolean; useImu: boolean; contentBudget: string; keepAlive: number };
  ar: ImageTracker | null;
  video: HTMLVideoElement | null;
  targets: Map<number, El>;
  contents: Map<number, ContentComponent>;
  content: ContentManager<AFrameContent> | null;
  start(): Promise<void>;
  register(index: number, el: El): void;
  unregister(index: number): void;
  registerContent(index: number, c: ContentComponent): void;
  unregisterContent(index: number, c: ContentComponent): void;
  startContent(ar: ImageTracker): void;
  onFrame(e: FrameEvent): void;
}

const VIDEO = /\.(mp4|m4v|webm|mov|ogv)(\?|#|$)/i;

export function registerAFrame(AFRAME: AFrameLike): void {
  AFRAME.registerSystem('fastar', {
    schema: {
      targets: { type: 'string', default: 'targets.bin' },
      /** 0: every target in the file. */
      maxTrack: { type: 'int', default: 0 },
      autoStart: { type: 'boolean', default: true },
      useImu: { type: 'boolean', default: true },
      /** 'auto' or a number of items kept loaded by fastar-content. */
      contentBudget: { type: 'string', default: 'auto' },
      keepAlive: { type: 'number', default: 3000 },
    },

    init(this: FastarSystem) {
      this.ar = null;
      this.video = null;
      this.targets = new Map();
      this.contents = new Map();
      this.content = null;
      const begin = () => {
        if (!this.data.autoStart) return;
        // iOS needs a tap for motion permission (and often for camera playback).
        if (MotionSensors.needsPermission()) document.addEventListener('pointerup', () => void this.start(), { once: true });
        else void this.start();
      };
      if (this.el.hasLoaded) begin();
      else this.el.addEventListener('loaded', begin, { once: true });
    },

    async start(this: FastarSystem) {
      if (this.ar) return;
      try {
        const video = document.createElement('video');
        Object.assign(video.style, {
          position: 'fixed', inset: '0', width: '100%', height: '100%', objectFit: 'cover', zIndex: '-2',
        });
        document.body.prepend(video);
        this.video = video;
        const canvas = this.el.canvas;
        if (canvas) canvas.style.background = 'transparent';
        const ar = await ImageTracker.create({ targets: this.data.targets, maxTrack: this.data.maxTrack > 0 ? this.data.maxTrack : undefined });
        this.ar = ar;
        this.el.emit('fastar-ready');
        ar.on('found', (i) => this.targets.get(i)?.emit('targetFound'));
        ar.on('lost', (i) => this.targets.get(i)?.emit('targetLost'));
        ar.on('near', (i) => this.targets.get(i)?.emit('targetNear'));
        ar.on('performance', (p) => this.el.emit('fastar-performance', p));
        ar.onFrame((e) => this.onFrame(e));
        this.startContent(ar);
        await ar.start({ video, useImu: this.data.useImu, requestMotionPermission: true });
        this.el.emit('fastar-started');
      } catch (e) {
        this.el.emit('fastar-error', { error: e });
      }
    },

    /** Content for the fastar-content entities, loaded as their targets come into view. */
    startContent(this: FastarSystem, ar: ImageTracker) {
      const b = this.data.contentBudget;
      const m = (this.content = new ContentManager<AFrameContent>(ar, {
        targets: [],
        budget: b === 'auto' || b === '' ? 'auto' : Number(b),
        keepAlive: this.data.keepAlive,
        load: (i, ctx) => {
          const c = this.contents.get(i);
          if (!c) throw new Error(`fastar: no content for target ${i}`);
          return c.load(ctx.signal, ar.targets[i]);
        },
        show: (_, v) => v.show(),
        hide: (_, v) => v.hide(),
        unload: (_, v) => v.unload(),
        onError: (i, err) => console.warn(`fastar: content for target ${i} failed to load`, err),
      }));
      m.on('ready', (i) => this.contents.get(i)?.el.emit('contentReady'));
      m.on('unload', (i) => this.contents.get(i)?.el.emit('contentUnloaded'));
      for (const i of this.contents.keys()) m.add(i);
    },

    register(this: FastarSystem, index: number, el: El) {
      el.object3D.matrixAutoUpdate = false;
      el.object3D.visible = false;
      this.targets.set(index, el);
    },

    unregister(this: FastarSystem, index: number) {
      this.targets.delete(index);
    },

    registerContent(this: FastarSystem, index: number, c: ContentComponent) {
      this.contents.set(index, c);
      this.content?.add(index);
    },

    unregisterContent(this: FastarSystem, index: number, c: ContentComponent) {
      if (this.contents.get(index) === c) this.contents.delete(index);
    },

    onFrame(this: FastarSystem, e: FrameEvent) {
      const cam = this.el.camera;
      const canvas = this.el.canvas;
      if (cam && this.ar) {
        const w = canvas?.clientWidth || innerWidth, h = canvas?.clientHeight || innerHeight;
        cam.projectionMatrix.fromArray(this.ar.getProjectionMatrix(w, h, cam.near, cam.far, 'cover'));
        cam.projectionMatrixInverse?.copy(cam.projectionMatrix).invert();
        // Poses are camera-relative: keep the camera entity at the origin.
        const ce = cam.el?.object3D;
        if (ce) {
          ce.matrixAutoUpdate = false;
          ce.matrix.identity?.();
          ce.matrixWorldNeedsUpdate = true;
        }
      }
      for (const [i, el] of this.targets) {
        const p = e.poses.get(i);
        el.object3D.visible = !!p;
        if (p) {
          el.object3D.matrix.fromArray(p.matrix);
          el.object3D.matrixWorldNeedsUpdate = true;
        }
      }
    },
  });

  AFRAME.registerComponent('fastar-target', {
    schema: { index: { type: 'int', default: 0 } },
    init(this: { el: El; data: { index: number } }) {
      this.el.sceneEl.systems.fastar?.register(this.data.index, this.el);
    },
    remove(this: { el: El; data: { index: number } }) {
      this.el.sceneEl.systems.fastar?.unregister(this.data.index);
    },
  });

  AFRAME.registerComponent('fastar-content', {
    schema: {
      src: { type: 'string', default: '' },
      type: { type: 'string', default: 'auto', oneOf: ['auto', 'gltf', 'video'] },
      /** Target index (default: this entity's fastar-target). */
      index: { type: 'int', default: -1 },
    },
    init(this: ContentComponent) {
      this.el.sceneEl.systems.fastar?.registerContent(contentIndex(this), this);
    },
    remove(this: ContentComponent) {
      this.el.sceneEl.systems.fastar?.unregisterContent(contentIndex(this), this);
    },
    load(this: ContentComponent, signal: AbortSignal, meta: TargetMeta): Promise<AFrameContent> {
      const { src, type } = this.data;
      const doc = (this.el.ownerDocument ?? document) as unknown as DocLike;
      return (type === 'video' || (type === 'auto' && VIDEO.test(src)))
        ? loadVideo(doc, this.el, src, meta, signal)
        : loadModel(doc, this.el, src, signal);
    },
  });
}

function contentIndex(c: ContentComponent): number {
  return c.data.index >= 0 ? c.data.index : c.el.components?.['fastar-target']?.data?.index ?? 0;
}

function aborted(): Error {
  const e = new Error('fastar: content load aborted');
  e.name = 'AbortError';
  return e;
}

/** Settle on `ok` / `fail` events of `node`, or reject on abort (after `cleanup`). */
function whenLoaded(node: NodeLike, ok: string, fail: string, signal: AbortSignal, cleanup: () => void): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const done = () => {
      node.removeEventListener?.(ok, onOk);
      node.removeEventListener?.(fail, onFail);
      signal.removeEventListener('abort', onAbort);
    };
    const onOk = () => { done(); resolve(); };
    const onFail = (e?: unknown) => { done(); cleanup(); reject(e instanceof Error ? e : new Error(`fastar: could not load content (${fail})`)); };
    const onAbort = () => { done(); cleanup(); reject(aborted()); };
    if (signal.aborted) return onAbort();
    node.addEventListener(ok, onOk);
    node.addEventListener(fail, onFail);
    signal.addEventListener('abort', onAbort);
  });
}

/** A glTF model in a child entity (A-Frame's gltf-model). */
async function loadModel(doc: DocLike, el: El, src: string, signal: AbortSignal): Promise<AFrameContent> {
  const child = doc.createElement('a-entity');
  const detach = () => child.parentNode?.removeChild(child);
  child.setAttribute('gltf-model', src);
  const loaded = whenLoaded(child, 'model-loaded', 'model-error', signal, detach);
  el.appendChild?.(child);
  await loaded;
  return {
    show() {},
    hide() {},
    unload() {
      disposeObject(child.object3D);
      detach();
    },
  };
}

/** A looping, muted video on a plane covering the target, playing only while the target is in view. */
async function loadVideo(doc: DocLike, el: El, src: string, meta: TargetMeta, signal: AbortSignal): Promise<AFrameContent> {
  const v = doc.createElement('video') as NodeLike & { [k: string]: unknown; play?(): Promise<void>; pause?(): void; load?(): void };
  v.crossOrigin = 'anonymous';
  v.loop = true;
  v.muted = true;
  v.playsInline = true;
  v.setAttribute('playsinline', '');
  v.preload = 'auto';
  const release = () => {
    v.pause?.();
    v.removeAttribute?.('src');
    v.load?.();
  };
  const ready = whenLoaded(v, 'loadeddata', 'error', signal, release);
  v.src = src;
  v.load?.();
  await ready;
  const plane = doc.createElement('a-plane');
  const w = meta.physicalWidth, h = (meta.physicalWidth * meta.height) / meta.width;
  plane.setAttribute('width', w);
  plane.setAttribute('height', h);
  plane.setAttribute('material', { shader: 'flat', src: v });
  el.appendChild?.(plane);
  return {
    show() { void v.play?.()?.catch?.(() => {}); },
    hide() { v.pause?.(); },
    unload() {
      release();
      disposeObject(plane.object3D);
      plane.parentNode?.removeChild(plane);
    },
  };
}
