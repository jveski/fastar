// Ray-cast renderer for planar scenes with rolling shutter, motion blur,
// blur and sensor noise. Test-only; not part of the browser bundle.
import { createGray, blur5, type GrayImage } from '../image/gray';
import { pyrDown, PyrScratch } from '../image/pyramid';
import type { Intrinsics } from '../vision/camera';
import { Rng } from '../math/rng';
import { mat3MulAtB, type Mat3, type Vec3 } from '../math/rot';

export interface RigidPose {
  R: Mat3; // world-from-frame rotation
  p: Vec3; // frame origin in world
}

export interface TexturedPlane {
  /** Plane frame: x right, y down, z into the plane; origin at texture centre. */
  pose: (t: number) => RigidPose;
  texture: GrayImage;
  /** Time-varying texture (a playing video); same size as `texture`. */
  textureAt?: (t: number) => GrayImage;
  metersPerTexel: number;
  tile?: boolean;
  /** If set, the plane is only rendered when this returns true. */
  active?: (t: number) => boolean;
  /** Constant intensity instead of texture sampling. */
  flat?: number;
}

export interface RenderOptions {
  /** Rolling-shutter readout time (top to bottom) in seconds. */
  readout?: number;
  /** Exposure time in seconds (motion blur). */
  exposure?: number;
  exposureSamples?: number;
  /** Gaussian noise std in grey levels. */
  noise?: number;
  /** Extra optical blur passes of the [1 4 6 4 1] kernel. */
  blurPasses?: number;
  seed?: number;
  background?: number;
}

interface PreparedPlane extends TexturedPlane {
  mips: GrayImage[];
}

export class Renderer {
  private readonly planes: PreparedPlane[];
  private readonly acc: Float32Array;
  private readonly rng: Rng;
  private readonly tmp: Uint16Array;

  constructor(readonly K: Intrinsics, planes: TexturedPlane[], seed = 1) {
    this.planes = planes.map((p) => ({ ...p, mips: buildMips(p.texture) }));
    this.acc = new Float32Array(K.width * K.height);
    this.rng = new Rng(seed);
    this.tmp = new Uint16Array(K.width * K.height);
  }

  /**
   * Render a frame whose centre row is exposed at time t (mid-exposure).
   * cameraPose gives world-from-camera (OpenCV camera axes).
   */
  render(t: number, cameraPose: (t: number) => RigidPose, opts: RenderOptions = {}, out?: GrayImage): GrayImage {
    const K = this.K;
    const W = K.width, H = K.height;
    const img = out ?? createGray(W, H);
    const readout = opts.readout ?? 0;
    const exposure = opts.exposure ?? 0;
    const ns = exposure > 0 ? Math.max(1, opts.exposureSamples ?? 4) : 1;
    const acc = this.acc;
    acc.fill(0);
    const bg = opts.background ?? 128;
    for (let s = 0; s < ns; s++) {
      const dtExp = ns === 1 ? 0 : (s / (ns - 1) - 0.5) * exposure;
      for (let y = 0; y < H; y++) {
        const tr = t + dtExp + (y / (H - 1) - 0.5) * readout;
        this.renderRow(y, tr, cameraPose(tr), bg, acc);
      }
    }
    const d = img.data;
    const inv = 1 / ns;
    const noise = opts.noise ?? 0;
    for (let i = 0; i < W * H; i++) {
      let v = acc[i] * inv;
      if (noise > 0) v += this.rng.gauss() * noise;
      d[i] = v < 0 ? 0 : v > 255 ? 255 : (v + 0.5) | 0;
    }
    const passes = opts.blurPasses ?? 0;
    for (let i = 0; i < passes; i++) blur5(img, img, this.tmp);
    return img;
  }

  private renderRow(y: number, t: number, cam: RigidPose, bg: number, acc: Float32Array): void {
    const K = this.K;
    const W = K.width;
    const dy = (y - K.cy) / K.fy;
    const nPlanes = this.planes.length;
    // Per-plane camera-in-plane transforms for this row.
    const tf: Array<{ R: Float64Array; o: Float64Array; pl: PreparedPlane; mips: GrayImage[] } | null> = [];
    for (let k = 0; k < nPlanes; k++) {
      const pl = this.planes[k];
      if (pl.active && !pl.active(t)) { tf.push(null); continue; }
      const pp = pl.pose(t);
      const R = mat3MulAtB(pp.R, cam.R, new Float64Array(9));
      const ex = cam.p[0] - pp.p[0], ey = cam.p[1] - pp.p[1], ez = cam.p[2] - pp.p[2];
      const o = Float64Array.of(
        pp.R[0] * ex + pp.R[3] * ey + pp.R[6] * ez,
        pp.R[1] * ex + pp.R[4] * ey + pp.R[7] * ez,
        pp.R[2] * ex + pp.R[5] * ey + pp.R[8] * ez,
      );
      tf.push({ R, o, pl, mips: pl.textureAt ? mipsOf(pl.textureAt(t)) : pl.mips });
    }
    const row = y * W;
    for (let x = 0; x < W; x++) {
      const dx = (x - K.cx) / K.fx;
      let bestL = Infinity, val = bg;
      for (let k = 0; k < nPlanes; k++) {
        const f = tf[k];
        if (!f) continue;
        const R = f.R, o = f.o;
        const px = R[0] * dx + R[1] * dy + R[2];
        const py = R[3] * dx + R[4] * dy + R[5];
        const pz = R[6] * dx + R[7] * dy + R[8];
        if (Math.abs(pz) < 1e-9) continue;
        const lam = -o[2] / pz;
        if (lam <= 0 || lam >= bestL) continue;
        const X = o[0] + lam * px, Y = o[1] + lam * py;
        const pl = f.pl;
        const tex = pl.texture;
        const mpt = pl.metersPerTexel;
        const u = X / mpt + (tex.width - 1) / 2, v = Y / mpt + (tex.height - 1) / 2;
        if (!pl.tile && (u < -0.5 || v < -0.5 || u > tex.width - 0.5 || v > tex.height - 0.5)) continue;
        bestL = lam;
        if (pl.flat !== undefined) { val = pl.flat; continue; }
        const cosv = Math.abs(pz) / Math.sqrt(px * px + py * py + pz * pz);
        const foot = lam / K.fx / Math.max(cosv, 0.15) / mpt;
        const lev = foot > 1 ? Math.log2(foot) : 0;
        val = sampleTrilinear(f.mips, lev, u, v, !!pl.tile);
      }
      acc[row + x] += val;
    }
  }
}

const mipCache = new WeakMap<GrayImage, GrayImage[]>();
function mipsOf(tex: GrayImage): GrayImage[] {
  let m = mipCache.get(tex);
  if (!m) mipCache.set(tex, (m = buildMips(tex)));
  return m;
}

function buildMips(tex: GrayImage): GrayImage[] {
  const mips = [tex];
  let cur = tex;
  while (cur.width > 8 && cur.height > 8) {
    const next = createGray(cur.width >> 1, cur.height >> 1);
    pyrDown(cur, next, new PyrScratch(cur.width, cur.height));
    mips.push(next);
    cur = next;
  }
  return mips;
}

function sampleTrilinear(mips: GrayImage[], lev: number, u: number, v: number, tile: boolean): number {
  const maxL = mips.length - 1;
  if (lev >= maxL) return sampleLevel(mips[maxL], u / (1 << maxL), v / (1 << maxL), tile);
  const l0 = Math.floor(lev), a = lev - l0;
  const s0 = sampleLevel(mips[l0], u / (1 << l0), v / (1 << l0), tile);
  if (a < 1e-3) return s0;
  const s1 = sampleLevel(mips[l0 + 1], u / (1 << (l0 + 1)), v / (1 << (l0 + 1)), tile);
  return s0 + (s1 - s0) * a;
}

function sampleLevel(img: GrayImage, u: number, v: number, tile: boolean): number {
  const w = img.width, h = img.height, d = img.data;
  const x0f = Math.floor(u), y0f = Math.floor(v);
  const ax = u - x0f, ay = v - y0f;
  let x0: number, x1: number, y0: number, y1: number;
  if (tile) {
    x0 = ((x0f % w) + w) % w; x1 = (x0 + 1) % w;
    y0 = ((y0f % h) + h) % h; y1 = (y0 + 1) % h;
  } else {
    x0 = x0f < 0 ? 0 : x0f >= w ? w - 1 : x0f;
    x1 = x0f + 1 < 0 ? 0 : x0f + 1 >= w ? w - 1 : x0f + 1;
    y0 = y0f < 0 ? 0 : y0f >= h ? h - 1 : y0f;
    y1 = y0f + 1 < 0 ? 0 : y0f + 1 >= h ? h - 1 : y0f + 1;
  }
  const a = d[y0 * w + x0], b = d[y0 * w + x1], c = d[y1 * w + x0], e = d[y1 * w + x1];
  const top = a + (b - a) * ax, bot = c + (e - c) * ax;
  return top + (bot - top) * ay;
}
