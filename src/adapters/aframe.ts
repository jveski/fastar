// A-Frame integration (no dependency on A-Frame; pass the global AFRAME).
//
//   <a-scene fastar="targets: targets.bin; maxTrack: 1" vr-mode-ui="enabled: false">
//     <a-camera look-controls="enabled: false" wasd-controls="enabled: false"></a-camera>
//     <a-entity fastar-target="index: 0"><a-box scale="0.05 0.05 0.05"></a-box></a-entity>
//   </a-scene>
//
// Target entities emit `targetFound` / `targetLost`. The scene emits
// `fastar-ready`, `fastar-started` and `fastar-error`.
import { ImageTracker, type FrameEvent } from '../shell/image-tracker';
import { MotionSensors } from '../shell/sensors';

interface AFrameLike {
  registerSystem(name: string, def: Record<string, unknown>): void;
  registerComponent(name: string, def: Record<string, unknown>): void;
}

interface El {
  object3D: { matrix: { fromArray(a: ArrayLike<number>): unknown; identity?(): unknown }; matrixAutoUpdate: boolean; matrixWorldNeedsUpdate: boolean; visible: boolean };
  emit(name: string, detail?: unknown): void;
  sceneEl: SceneEl;
  addEventListener(n: string, f: () => void, o?: unknown): void;
}

interface SceneEl extends El {
  camera: { projectionMatrix: { fromArray(a: ArrayLike<number>): unknown }; projectionMatrixInverse?: { copy(m: unknown): { invert(): unknown } }; near: number; far: number; el?: El };
  canvas?: HTMLCanvasElement;
  hasLoaded: boolean;
  systems: Record<string, FastarSystem>;
}

interface FastarSystem {
  el: SceneEl;
  data: { targets: string; maxTrack: number; autoStart: boolean; useImu: boolean };
  ar: ImageTracker | null;
  video: HTMLVideoElement | null;
  targets: Map<number, El>;
  start(): Promise<void>;
  register(index: number, el: El): void;
  unregister(index: number): void;
  onFrame(e: FrameEvent): void;
}

export function registerAFrame(AFRAME: AFrameLike): void {
  AFRAME.registerSystem('fastar', {
    schema: {
      targets: { type: 'string', default: 'targets.bin' },
      maxTrack: { type: 'int', default: 1 },
      autoStart: { type: 'boolean', default: true },
      useImu: { type: 'boolean', default: true },
    },

    init(this: FastarSystem) {
      this.ar = null;
      this.video = null;
      this.targets = new Map();
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
        const ar = await ImageTracker.create({ targets: this.data.targets, maxTrack: this.data.maxTrack });
        this.ar = ar;
        this.el.emit('fastar-ready');
        ar.on('found', (i) => this.targets.get(i)?.emit('targetFound'));
        ar.on('lost', (i) => this.targets.get(i)?.emit('targetLost'));
        ar.onFrame((e) => this.onFrame(e));
        await ar.start({ video, useImu: this.data.useImu, requestMotionPermission: true });
        this.el.emit('fastar-started');
      } catch (e) {
        this.el.emit('fastar-error', { error: e });
      }
    },

    register(this: FastarSystem, index: number, el: El) {
      el.object3D.matrixAutoUpdate = false;
      el.object3D.visible = false;
      this.targets.set(index, el);
    },

    unregister(this: FastarSystem, index: number) {
      this.targets.delete(index);
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
}
