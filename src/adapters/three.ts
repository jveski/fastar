// three.js adapter. Duck-typed so the library does not depend on three:
// pass your THREE namespace (for Group) and camera/scene/renderer.
import type { ImageTracker, FrameEvent } from '../shell/image-tracker';

interface Matrix4Like {
  fromArray(a: ArrayLike<number>): unknown;
  copy?(m: Matrix4Like): Matrix4Like;
  invert?(): unknown;
  identity?(): unknown;
}

export interface Object3DLike {
  matrix: Matrix4Like;
  matrixAutoUpdate: boolean;
  matrixWorldNeedsUpdate: boolean;
  visible: boolean;
  add?(o: unknown): unknown;
}

export interface CameraLike extends Object3DLike {
  projectionMatrix: Matrix4Like;
  projectionMatrixInverse?: Matrix4Like;
  near: number;
  far: number;
  updateMatrixWorld?(force?: boolean): void;
}

interface TextureLike {
  image: unknown;
  needsUpdate: boolean;
  colorSpace?: string;
  offset: { set(x: number, y: number): unknown };
  repeat: { set(x: number, y: number): unknown };
  dispose?(): void;
}

export interface ThreeAdapterOptions {
  camera: CameraLike;
  scene?: { add(o: unknown): unknown; background?: unknown };
  /**
   * THREE namespace, used to create anchor groups (and, with syncDisplay,
   * textures for the camera background).
   */
  THREE?: { Group: new () => Object3DLike; Texture?: new (image?: unknown) => TextureLike; SRGBColorSpace?: string };
  /** Called after matrices are updated, inside the video-frame callback. */
  render?: (e: FrameEvent) => void;
  /** Size of the rendered view in CSS pixels (defaults to the video element box). */
  viewSize?: () => [number, number];
}

/**
 * Keeps a three.js camera and per-target anchors in sync with the tracker.
 * The camera sits at the origin; anchors carry the target pose (metres,
 * y up, z out of the target, origin at the target centre). When frames carry
 * an image (start({ syncDisplay: true })) it becomes `scene.background`,
 * cropped like the video's object-fit, so background and overlay always come
 * from the same frame.
 */
export class ThreeAdapter {
  private readonly anchors = new Map<number, Object3DLike>();
  private readonly textures = new Map<unknown, TextureLike>();
  private readonly off: () => void;

  constructor(private readonly ar: ImageTracker, private readonly o: ThreeAdapterOptions) {
    const cam = o.camera;
    cam.matrixAutoUpdate = false;
    cam.matrix.identity?.();
    cam.matrixWorldNeedsUpdate = true;
    this.off = ar.onFrame((e) => this.update(e));
  }

  /** Anchor object for target `index` (created via THREE.Group if not supplied). */
  anchor(index: number, obj?: Object3DLike): Object3DLike {
    let a = this.anchors.get(index);
    if (obj && a !== obj) {
      a = obj;
      this.anchors.set(index, a);
    }
    if (!a) {
      if (!this.o.THREE) throw new Error('fastar: pass THREE in ThreeAdapter options or supply an object');
      a = new this.o.THREE.Group();
      this.o.scene?.add(a);
      this.anchors.set(index, a);
    }
    a.matrixAutoUpdate = false;
    a.visible = false;
    return a;
  }

  private update(e: FrameEvent): void {
    const cam = this.o.camera;
    const [w, h] = this.o.viewSize?.() ?? [0, 0];
    const P = w && h ? this.ar.getProjectionMatrix(w, h, cam.near, cam.far) : this.ar.projectionMatrix;
    cam.projectionMatrix.fromArray(P);
    if (cam.projectionMatrixInverse?.copy) cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert?.();
    cam.updateMatrixWorld?.(true);
    if (e.image) this.setBackground(e.image, w || e.image.width, h || e.image.height);
    for (const [i, a] of this.anchors) {
      const p = e.poses.get(i);
      a.visible = !!p;
      if (p) {
        a.matrix.fromArray(p.matrix);
        a.matrixWorldNeedsUpdate = true;
      }
    }
    this.o.render?.(e);
  }

  private setBackground(image: { width: number; height: number }, viewW: number, viewH: number): void {
    const T = this.o.THREE?.Texture, scene = this.o.scene;
    if (!T || !scene) return;
    let tex = this.textures.get(image);
    if (!tex) {
      tex = new T(image);
      if (this.o.THREE?.SRGBColorSpace) tex.colorSpace = this.o.THREE.SRGBColorSpace;
      this.textures.set(image, tex);
    }
    tex.needsUpdate = true;
    // object-fit as UV crop (cover) or shrink (contain) of the full-view background quad.
    const fit = this.ar.fitMode;
    let rx = 1, ry = 1;
    if (fit !== 'fill') {
      const ia = image.width / image.height, va = viewW / viewH;
      if ((fit === 'cover') === ia > va) rx = va / ia;
      else ry = ia / va;
    }
    tex.repeat.set(rx, ry);
    tex.offset.set((1 - rx) / 2, (1 - ry) / 2);
    scene.background = tex;
  }

  dispose(): void {
    this.off();
    for (const t of this.textures.values()) t.dispose?.();
    this.textures.clear();
  }
}
