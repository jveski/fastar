// three.js adapter. Duck-typed so the library does not depend on three:
// pass your THREE namespace (for Group) and camera/scene/renderer.
import type { ImageTracker, FrameEvent } from '../shell/image-tracker';
import { Surface } from '../shell/surface';
import { ContentManager, type ContentLoadContext, type ContentOptions } from '../shell/content';

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
  remove?(o: unknown): unknown;
  traverse?(f: (o: unknown) => void): void;
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

interface AttributeLike {
  array: ArrayLike<number> & { set(a: ArrayLike<number>, offset?: number): void };
  needsUpdate: boolean;
  setUsage?(usage: unknown): unknown;
}

interface GeometryLike {
  setAttribute(name: string, attr: unknown): unknown;
  setIndex(index: unknown): unknown;
  getAttribute(name: string): AttributeLike;
  computeBoundingSphere?(): void;
  dispose?(): void;
}

/** What surface meshes need from the THREE namespace. */
interface ThreeSurfaceLike {
  BufferGeometry: new () => GeometryLike;
  BufferAttribute: new (array: ArrayLike<number>, itemSize: number) => AttributeLike;
  Mesh: new (geometry: GeometryLike, material?: unknown) => Object3DLike;
  DynamicDrawUsage?: unknown;
}

export interface SurfaceMeshOptions {
  /** Material for the mesh (three's default if omitted). */
  material?: unknown;
  /** Render grid resolution across and down the target (default 32 x 24). */
  segments?: [number, number];
}

export interface ThreeAdapterOptions {
  camera: CameraLike;
  scene?: { add(o: unknown): unknown; background?: unknown };
  /**
   * THREE namespace, used to create anchor groups (and, with syncDisplay,
   * textures for the camera background).
   */
  THREE?: { Group: new () => Object3DLike; Texture?: new (image?: unknown) => TextureLike; SRGBColorSpace?: string } & Partial<ThreeSurfaceLike>;
  /** Called after matrices are updated, inside the video-frame callback. */
  render?: (e: FrameEvent) => void;
  /** Size of the rendered view in CSS pixels (defaults to the video element box). */
  viewSize?: () => [number, number];
  /** How much lazily loaded content (see content()) stays loaded, and for how long. */
  content?: Pick<ContentOptions<unknown>, 'budget' | 'keepAlive' | 'nearTimeout' | 'concurrency' | 'onError'>;
}

export interface ThreeContentOptions<T> {
  /** Load ahead, and keep loaded. */
  preload?: boolean;
  /** Weight against the content budget (default 1). */
  size?: number;
  /** The target came into view with this content (e.g. play a video texture). */
  show?(value: T): void;
  /** The target went out of view. */
  hide?(value: T): void;
  /** Extra cleanup when unloaded, after the object is removed and its geometries, materials and textures disposed. */
  unload?(value: T): void;
}

interface Registered {
  load(ctx: ContentLoadContext): Promise<Object3DLike> | Object3DLike;
  o: ThreeContentOptions<Object3DLike>;
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
  private readonly meshes: Array<{ index: number; geometry: GeometryLike; segX: number; segY: number }> = [];
  private readonly off: () => void;
  private readonly loaders = new Map<number, Registered>();
  private manager: ContentManager<Object3DLike> | null = null;

  constructor(private readonly ar: ImageTracker, private readonly o: ThreeAdapterOptions) {
    const cam = o.camera;
    cam.matrixAutoUpdate = false;
    cam.matrix.identity?.();
    cam.matrixWorldNeedsUpdate = true;
    this.off = ar.onFrame((e) => this.update(e));
  }

  /**
   * Lazily loaded content for target `index`: `load` runs when the target is
   * found (or nearly found), and the object it returns is added to the
   * target's anchor. Once the target has been out of view a while and room
   * is needed, the object is removed and disposed (see the `content` option
   * for the budget).
   */
  content<T extends Object3DLike>(index: number, load: (ctx: ContentLoadContext) => Promise<T> | T, o: ThreeContentOptions<T> = {}): void {
    if (!this.ar.targets[index]) throw new Error(`fastar: no target ${index}`);
    this.loaders.set(index, { load, o: o as ThreeContentOptions<Object3DLike> });
    if (!this.manager) {
      const m = (this.manager = new ContentManager<Object3DLike>(this.ar, {
        ...this.o.content,
        targets: [],
        load: (i, ctx) => this.loaders.get(i)!.load(ctx),
        size: (i) => this.loaders.get(i)?.o.size ?? 1,
        show: (i, v) => this.loaders.get(i)?.o.show?.(v),
        hide: (i, v) => this.loaders.get(i)?.o.hide?.(v),
        unload: (i, v) => {
          this.anchors.get(i)?.remove?.(v);
          disposeObject(v);
          this.loaders.get(i)?.o.unload?.(v);
        },
      }));
      m.on('ready', (i) => {
        const v = m.get(i);
        if (v) (this.anchors.get(i) ?? this.anchor(i)).add?.(v);
      });
    }
    this.manager.add(index, !!o.preload);
  }

  /** The content manager behind content() (state, budget, events), once content is registered. */
  get contents(): ContentManager<Object3DLike> | null {
    return this.manager;
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

  /**
   * A mesh covering target `index` that follows its shape when it bends
   * (create with the deformable option), parented to the target's anchor.
   * Texture coordinates map the target image onto it; normals are smooth.
   */
  surface(index: number, o: SurfaceMeshOptions = {}): Object3DLike {
    const T = this.o.THREE;
    if (!T?.BufferGeometry || !T.BufferAttribute || !T.Mesh) throw new Error('fastar: pass THREE (with BufferGeometry, BufferAttribute and Mesh) in ThreeAdapter options');
    const meta = this.ar.targets[index];
    if (!meta) throw new Error(`fastar: no target ${index}`);
    const [segX, segY] = o.segments ?? [32, 24];
    const n = (segX + 1) * (segY + 1);
    const geometry = new T.BufferGeometry();
    const pos = new T.BufferAttribute(new Float32Array(3 * n), 3), nrm = new T.BufferAttribute(new Float32Array(3 * n), 3);
    pos.setUsage?.(T.DynamicDrawUsage);
    nrm.setUsage?.(T.DynamicDrawUsage);
    geometry.setAttribute('position', pos);
    geometry.setAttribute('normal', nrm);
    geometry.setAttribute('uv', new T.BufferAttribute(new Float32Array(2 * n), 2));
    geometry.setIndex(new T.BufferAttribute(Surface.gridIndices(segX, segY), 1));
    const W = meta.physicalWidth, H = (W * meta.height) / meta.width;
    this.fill(geometry, Surface.flat(2, 2, W, H), segX, segY, true);
    const mesh = new T.Mesh(geometry, o.material);
    this.anchor(index).add?.(mesh);
    this.meshes.push({ index, geometry, segX, segY });
    return mesh;
  }

  private fill(g: GeometryLike, s: Surface, segX: number, segY: number, uvs = false): void {
    const pos = g.getAttribute('position'), nrm = g.getAttribute('normal');
    s.fillGrid(segX, segY, pos.array as Float32Array, nrm.array as Float32Array, uvs ? g.getAttribute('uv').array as Float32Array : null);
    pos.needsUpdate = nrm.needsUpdate = true;
    g.computeBoundingSphere?.();
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
    for (const m of this.meshes) {
      const s = e.poses.get(m.index)?.surface;
      if (s) this.fill(m.geometry, s, m.segX, m.segY);
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
    this.manager?.dispose();
    this.manager = null;
    for (const t of this.textures.values()) t.dispose?.();
    this.textures.clear();
    for (const m of this.meshes) m.geometry.dispose?.();
    this.meshes.length = 0;
  }
}

interface Disposable { dispose?(): void }

/** Dispose the geometries, materials and textures under a three.js object. */
export function disposeObject(root: unknown): void {
  const visit = (o: unknown) => {
    const n = o as { geometry?: Disposable; material?: Disposable | Disposable[] };
    n.geometry?.dispose?.();
    const mats = Array.isArray(n.material) ? n.material : n.material ? [n.material] : [];
    for (const m of mats) {
      for (const v of Object.values(m)) if ((v as { isTexture?: boolean } | null)?.isTexture) (v as Disposable).dispose?.();
      m.dispose?.();
    }
  };
  const r = root as Object3DLike;
  if (r.traverse) r.traverse(visit);
  else visit(root);
}
