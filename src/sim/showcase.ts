// "Marbled ink": a generative, seamlessly looping clip for the video example
// and tests, where the whole picture keeps churning. Two detailed textures
// (a mosaic of cells and a heap of ink shapes) are pulled through swirling
// flows that surge and reverse, flood over each other along moving fronts,
// and change colour all the time. Nothing on screen looks the same a second
// later, yet every patch only moves and bends a little from frame to frame,
// which is what the video compiler can follow. Every motion has a whole
// number of cycles per loop, so the last frame flows into the first.
//
// Colour is laid on as chroma around a fixed luma, so the colour churn
// leaves the brightness pattern (all the tracker sees) to the flow.
//
// Drawn in design coordinates (960x540) scaled to the output size.
import { createGray, rgbaToGray, type GrayImage } from '../image/gray';
import { Rng } from '../math/rng';
import type { VideoClip } from './motion';

export interface ShowcaseOptions {
  width?: number;
  fps?: number;
  seed?: number;
}

/** A periodic term: amp * sin(2 pi cycles t / loop + phase). */
export type Wave = [amp: number, cycles: number, phase: number];

/** What an overlay needs to move with the clip (JSON-friendly). */
export interface ShowcaseScene {
  designWidth: number;
  designHeight: number;
  loop: number;
  /** Content rotation at the screen centre (rad, counter-clockwise as seen) = sum of waves. */
  spin: Wave[];
  /** Content scale at the screen centre = exp(-(sum of waves)). */
  zoom: Wave[];
  /** Hue turns per loop: chroma rotates by 2 pi hueCycles t / loop (plus local ripples). */
  hueCycles: number;
}

export interface ShowcaseClip extends VideoClip {
  /** RGBA pixels of frame i (not cached). */
  frameRgba(i: number): Uint8ClampedArray;
  /** Scale from design pixels (960x540) to output pixels. */
  scale: number;
  scene: ShowcaseScene;
  /** Where the screen point (x, y) (design px) samples layer k's texture at time t (texture design px). */
  warp(k: number, x: number, y: number, t: number): [number, number];
}

export const DESIGN_W = 960;
export const DESIGN_H = 540;
export const LOOP = 12;
const CX = DESIGN_W / 2, CY = DESIGN_H / 2;
/** Source textures repeat every TILE design px (more than the screen diagonal). */
const TILE = 1536;
const HUE_CYCLES = 6;
/** Flow fields are evaluated on a grid this many output px apart and interpolated. */
const GRID = 8;

const TAU = 2 * Math.PI;
export const waveSum = (ws: Wave[], t: number): number => {
  let v = 0;
  for (const [a, n, ph] of ws) v += a * Math.sin((TAU * n * t) / LOOP + ph);
  return v;
};

// --- textures ------------------------------------------------------------

/** Periodic T x T texture: luma and chroma (Cb, Cr in -1..1 times saturation). */
interface Texture {
  T: number;
  Y: Float32Array;
  cb: Float32Array;
  cr: Float32Array;
}

const newTexture = (T: number): Texture => ({ T, Y: new Float32Array(T * T), cb: new Float32Array(T * T), cr: new Float32Array(T * T) });

/** Bright or dark, rarely in between: strong edges wherever two meet. */
const contrastLuma = (rng: Rng, bright: boolean) => (bright ? rng.range(165, 240) : rng.range(18, 90));

function randomChroma(rng: Rng): [number, number] {
  const h = rng.range(0, TAU), s = rng.range(0.55, 1);
  return [s * Math.cos(h), s * Math.sin(h)];
}

/**
 * Voronoi mosaic: cells of random size with dark seams, many holding a dot
 * of the opposite brightness. `sc` is texture px per design px.
 */
function mosaic(T: number, sc: number, rng: Rng): Texture {
  const tex = newTexture(T);
  const spacing = 38 * sc, n = Math.round((T / spacing) ** 2);
  const sx = new Float64Array(n), sy = new Float64Array(n), lum = new Float32Array(n), dotR = new Float32Array(n), dotL = new Float32Array(n);
  const ca = new Float32Array(n), cbb = new Float32Array(n), da = new Float32Array(n), db = new Float32Array(n);
  const ox = new Float32Array(n), oy = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    sx[i] = rng.range(0, T); sy[i] = rng.range(0, T);
    const bright = rng.next() < 0.5;
    lum[i] = contrastLuma(rng, bright);
    [ca[i], cbb[i]] = randomChroma(rng);
    dotR[i] = rng.next() < 0.6 ? rng.range(0.14, 0.3) * spacing : 0;
    dotL[i] = contrastLuma(rng, !bright);
    [da[i], db[i]] = randomChroma(rng);
    const a = rng.range(0, TAU), d = rng.range(0, 0.15) * spacing;
    ox[i] = d * Math.cos(a); oy[i] = d * Math.sin(a);
  }
  const G = Math.max(1, Math.floor(T / (2.5 * spacing))), cs = T / G;
  const buckets: number[][] = Array.from({ length: G * G }, () => []);
  for (let i = 0; i < n; i++) buckets[Math.min(G - 1, (sy[i] / cs) | 0) * G + Math.min(G - 1, (sx[i] / cs) | 0)].push(i);
  const half = T / 2;
  const wrapD = (d: number) => (d > half ? d - T : d < -half ? d + T : d);
  const seam = 1.3 * sc;
  const near: number[] = [];
  for (let y = 0; y < T; y++) {
    const by = Math.min(G - 1, (y / cs) | 0);
    for (let x = 0; x < T; x++) {
      const bx = Math.min(G - 1, (x / cs) | 0);
      near.length = 0;
      for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
        for (const k of buckets[((by + j + G) % G) * G + ((bx + i + G) % G)]) near.push(k);
      }
      const px = x + 0.5, py = y + 0.5;
      let best = -1, bd = Infinity;
      for (const k of near) {
        const dx = wrapD(px - sx[k]), dy = wrapD(py - sy[k]), d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = k; }
      }
      // Distance to the nearest cell edge (bisector with each neighbour).
      const bxk = wrapD(px - sx[best]), byk = wrapD(py - sy[best]);
      let edge = Infinity;
      for (const k of near) {
        if (k === best) continue;
        const ex = wrapD(sx[k] - sx[best]), ey = wrapD(sy[k] - sy[best]), el = Math.hypot(ex, ey);
        if (el < 1e-6) continue;
        const e = el / 2 - (bxk * ex + byk * ey) / el;
        if (e < edge) edge = e;
      }
      let Y = lum[best], u = ca[best], v = cbb[best];
      if (dotR[best] > 0) {
        const a = cover(dotR[best] - Math.hypot(bxk - ox[best], byk - oy[best]));
        Y += (dotL[best] - Y) * a; u += (da[best] - u) * a; v += (db[best] - v) * a;
      }
      const a = cover(seam - edge);
      Y += (10 - Y) * a; u *= 1 - a; v *= 1 - a;
      const o = y * T + x;
      tex.Y[o] = Y; tex.cb[o] = u; tex.cr[o] = v;
    }
  }
  return tex;
}

/** Heap of ink shapes, large first: discs, rings, shards, strokes and crescents. */
function ink(T: number, sc: number, rng: Rng): Texture {
  const tex = newTexture(T);
  tex.Y.fill(30);
  const area = (TILE * TILE) / 1e4;
  const classes: Array<[count: number, r0: number, r1: number]> = [
    [Math.round(area * 0.45), 38, 80],
    [Math.round(area * 2.8), 14, 38],
    [Math.round(area * 11), 5, 14],
  ];
  for (const [count, r0, r1] of classes) {
    for (let s = 0; s < count; s++) {
      const r = rng.range(r0, r1) * sc, cx = rng.range(0, T), cy = rng.range(0, T);
      const L = contrastLuma(rng, rng.next() < 0.5), [u, v] = randomChroma(rng);
      const kind = rng.int(5);
      let sdf: (x: number, y: number) => number;
      if (kind === 0) {
        sdf = (x, y) => Math.hypot(x, y) - r;
      } else if (kind === 1) {
        const w = rng.range(0.15, 0.35) * r;
        sdf = (x, y) => Math.abs(Math.hypot(x, y) - (r - w)) - w;
      } else if (kind === 2) {
        const k = 3 + rng.int(3), angles = Array.from({ length: k }, () => rng.range(0, TAU)).sort((a, b) => a - b);
        const nx: number[] = [], ny: number[] = [], off: number[] = [];
        for (let i = 0; i < k; i++) {
          const a0 = angles[i], a1 = angles[(i + 1) % k] + (i + 1 === k ? TAU : 0);
          if (a1 - a0 >= Math.PI) continue; // keep it convex
          const x0 = r * Math.cos(a0), y0 = r * Math.sin(a0), x1 = r * Math.cos(a1), y1 = r * Math.sin(a1);
          const ex = x1 - x0, ey = y1 - y0, l = Math.hypot(ex, ey);
          nx.push(ey / l); ny.push(-ex / l); off.push((x0 * ey - y0 * ex) / l);
        }
        if (!nx.length) continue;
        sdf = (x, y) => {
          let m = -Infinity;
          for (let i = 0; i < nx.length; i++) m = Math.max(m, x * nx[i] + y * ny[i] - off[i]);
          return m;
        };
      } else if (kind === 3) {
        const a = rng.range(0, Math.PI), dx = Math.cos(a) * r, dy = Math.sin(a) * r, w = rng.range(0.18, 0.4) * r;
        sdf = (x, y) => {
          const h = Math.max(-1, Math.min(1, (x * dx + y * dy) / (r * r)));
          return Math.hypot(x - h * dx, y - h * dy) - w;
        };
      } else {
        const a = rng.range(0, TAU), d = rng.range(0.35, 0.7) * r;
        const bx = d * Math.cos(a), by = d * Math.sin(a), r2 = rng.range(0.7, 0.95) * r;
        sdf = (x, y) => Math.max(Math.hypot(x, y) - r, r2 - Math.hypot(x - bx, y - by));
      }
      const X0 = Math.floor(cx - r - 2), X1 = Math.ceil(cx + r + 2), Y0 = Math.floor(cy - r - 2), Y1 = Math.ceil(cy + r + 2);
      for (let Y = Y0; Y <= Y1; Y++) {
        const row = (((Y % T) + T) % T) * T;
        for (let X = X0; X <= X1; X++) {
          const a = cover(-sdf(X + 0.5 - cx, Y + 0.5 - cy));
          if (a <= 0) continue;
          const o = row + (((X % T) + T) % T);
          tex.Y[o] += (L - tex.Y[o]) * a; tex.cb[o] += (u - tex.cb[o]) * a; tex.cr[o] += (v - tex.cr[o]) * a;
        }
      }
    }
  }
  return tex;
}

/** Antialiased coverage from a signed distance (px, positive inside). */
const cover = (d: number) => (d >= 0.5 ? 1 : d <= -0.5 ? 0 : d + 0.5);

/** Periodic 1-2-1 blur of the luma: softer edges alias less when the flow squeezes them. */
function soften(tex: Texture): void {
  const { T, Y } = tex, tmp = new Float32Array(T * T);
  for (let y = 0; y < T; y++) {
    const r = y * T;
    for (let x = 0; x < T; x++) tmp[r + x] = 0.25 * Y[r + ((x + T - 1) % T)] + 0.5 * Y[r + x] + 0.25 * Y[r + ((x + 1) % T)];
  }
  for (let y = 0; y < T; y++) {
    const u = ((y + T - 1) % T) * T, r = y * T, d = ((y + 1) % T) * T;
    for (let x = 0; x < T; x++) Y[r + x] = 0.25 * tmp[u + x] + 0.5 * tmp[r + x] + 0.25 * tmp[d + x];
  }
}

// --- flow ------------------------------------------------------------------

/**
 * A swirl: points within ~radius of its (moving) centre turn by an angle
 * that surges and reverses, and spread or gather by exp(zoom).
 */
interface Vortex {
  cx: number;
  cy: number;
  /** Centre path: a circle of this radius, cycles per loop and phase. */
  path: [r: number, n: number, ph: number];
  radius: number;
  angle: Wave[];
  zoom: Wave[];
}

interface LayerFlow {
  vortices: Vortex[];
  /** Tiles the texture travels per loop, in x and y (whole numbers, so it loops). */
  travel: [number, number];
}

/**
 * The flow of each layer: a swirl at the centre (shared, and what the
 * overlay follows) and a field of counter-rotating swirls over the rest of
 * the screen, while the texture travels underneath. Every part of the
 * picture moves differently from its surroundings, so no part of it can be
 * mistaken for the screen itself moving.
 */
function makeFlows(rng: Rng): { layers: LayerFlow[]; centre: Vortex } {
  const centre: Vortex = {
    cx: CX, cy: CY, path: [0, 0, 0], radius: 210,
    angle: [[1.4, 1, rng.range(0, TAU)], [0.35, 3, rng.range(0, TAU)]],
    zoom: [[0.22, 2, rng.range(0, TAU)], [0.08, 3, rng.range(0, TAU)]],
  };
  const layers: LayerFlow[] = [];
  const SP = 250;
  for (let k = 0; k < 2; k++) {
    const vortices: Vortex[] = [];
    const off = k * SP / 2;
    for (let gy = 0; gy < 4; gy++) {
      for (let gx = 0; gx < 6; gx++) {
        const radius = rng.range(140, 185);
        const path: [number, number, number] = [rng.range(25, 60), rng.next() < 0.5 ? -1 : 1, rng.range(0, TAU)];
        const cx = -140 + off + gx * SP + rng.range(-40, 40), cy = -120 + off + gy * SP + rng.range(-40, 40);
        if (cx > DESIGN_W + 160 || cy > DESIGN_H + 160) continue;
        // Clear of the centre swirl, so the centre turns as the scene says.
        if (Math.hypot(cx - CX, cy - CY) - path[0] < 1.8 * radius + 0.7 * centre.radius) continue;
        const n1 = 1 + rng.int(2), sign = (gx + gy) % 2 ? 1 : -1;
        vortices.push({
          cx, cy, path, radius,
          angle: [[(sign * rng.range(1, 1.5)) / n1, n1, rng.range(0, TAU)], [rng.range(0.2, 0.4), 3 + rng.int(2), rng.range(0, TAU)]],
          zoom: [[rng.range(0.1, 0.2), 1 + rng.int(3), rng.range(0, TAU)]],
        });
      }
    }
    layers.push({ vortices, travel: k ? [0, 1] : [1, 0] });
  }
  return { layers, centre };
}

function applyVortex(v: Vortex, t: number, x: number, y: number, out: Float64Array): void {
  const a = (TAU * v.path[1] * t) / LOOP + v.path[2];
  const vx = v.cx + v.path[0] * Math.cos(a), vy = v.cy + v.path[0] * Math.sin(a);
  const dx = x - vx, dy = y - vy, d2 = (dx * dx + dy * dy) / (v.radius * v.radius);
  if (d2 > 9) { out[0] = x; out[1] = y; return; }
  const f = Math.exp(-d2), th = waveSum(v.angle, t) * f, z = Math.exp(waveSum(v.zoom, t) * f);
  const c = z * Math.cos(th), s = z * Math.sin(th);
  out[0] = vx + c * dx - s * dy;
  out[1] = vy + s * dx + c * dy;
}

// --- clip --------------------------------------------------------------------

export function makeShowcaseClip(o: ShowcaseOptions = {}): ShowcaseClip {
  const W = o.width ?? 960, s = W / DESIGN_W, H = Math.round(DESIGN_H * s);
  const fps = o.fps ?? 30;
  const N = Math.round(LOOP * fps);
  const seed = o.seed ?? 5;
  const rng = new Rng(seed * 7919 + 1);
  const T = Math.round(TILE * s), ts = T / TILE; // texture px per design px
  const textures = [mosaic(T, ts, rng), ink(T, ts, rng)];
  textures.forEach(soften);
  const { layers, centre } = makeFlows(rng);

  // Fronts where one texture floods over the other: plane waves sweeping
  // across the (swirled) screen, with the balance between the two shifting.
  const fronts = Array.from({ length: 3 }, () => {
    const a = rng.range(0, TAU), k = TAU / rng.range(420, 720);
    return { kx: k * Math.cos(a), ky: k * Math.sin(a), n: (1 + rng.int(2)) * (rng.next() < 0.5 ? -1 : 1), ph: rng.range(0, TAU), amp: rng.range(0.6, 1) };
  });
  const ampSum = fronts.reduce((acc, f) => acc + f.amp, 0);
  const balance: Wave[] = [[0.35, 1, rng.range(0, TAU)], [0.15, 2, rng.range(0, TAU)]];
  // Colour ripples: hue waves over the screen on top of the steady hue turn.
  const ripples = Array.from({ length: 2 }, () => {
    const a = rng.range(0, TAU), k = TAU / rng.range(300, 600);
    return { kx: k * Math.cos(a), ky: k * Math.sin(a), n: (2 + rng.int(3)) * (rng.next() < 0.5 ? -1 : 1), ph: rng.range(0, TAU), amp: rng.range(0.7, 1.1) };
  });

  const tmp = new Float64Array(2);
  /** Layer k's texture position (design px, unwrapped) for screen point (x, y) at time t. */
  const warp = (k: number, x: number, y: number, t: number): [number, number] => {
    const L = layers[k];
    tmp[0] = x; tmp[1] = y;
    for (const v of L.vortices) applyVortex(v, t, tmp[0], tmp[1], tmp);
    applyVortex(centre, t, tmp[0], tmp[1], tmp);
    const u = (TILE * t) / LOOP;
    return [tmp[0] + L.travel[0] * u + k * 517, tmp[1] + L.travel[1] * u + k * 311];
  };

  // Per-frame grids (flow, front mask, hue rotation) at GRID px spacing.
  const gw = Math.ceil(W / GRID) + 1, gh = Math.ceil(H / GRID) + 1, gn = gw * gh;
  const gu = [new Float32Array(gn), new Float32Array(gn)], gv = [new Float32Array(gn), new Float32Array(gn)];
  const gm = new Float32Array(gn), gc = new Float32Array(gn), gs = new Float32Array(gn);
  const fillGrids = (t: number) => {
    const bal = waveSum(balance, t), hue0 = (TAU * HUE_CYCLES * t) / LOOP;
    for (let j = 0; j < gh; j++) {
      for (let i = 0; i < gw; i++) {
        const g = j * gw + i, x = (i * GRID) / s, y = (j * GRID) / s;
        for (let k = 0; k < 2; k++) {
          const [u, v] = warp(k, x, y, t);
          gu[k][g] = u * ts; gv[k][g] = v * ts;
        }
        // Fronts follow the first layer's swirls, so they curl with the picture.
        tmp[0] = x; tmp[1] = y;
        for (const v of layers[0].vortices) applyVortex(v, t, tmp[0], tmp[1], tmp);
        let m = bal;
        for (const f of fronts) m += (f.amp / ampSum) * Math.sin(f.kx * tmp[0] + f.ky * tmp[1] + (TAU * f.n * t) / LOOP + f.ph);
        gm[g] = m;
        let h = hue0;
        for (const r of ripples) h += r.amp * Math.sin(r.kx * x + r.ky * y + (TAU * r.n * t) / LOOP + r.ph);
        gc[g] = Math.cos(h); gs[g] = Math.sin(h);
      }
    }
  };

  const EDGE = 0.045; // half-width of a front, in mask units
  const CHROMA = 120;
  const frameRgba = (i: number): Uint8ClampedArray => {
    const t = (((i + 0.5) / fps) % LOOP + LOOP) % LOOP;
    fillGrids(t);
    const out = new Uint8ClampedArray(W * H * 4);
    const ta = textures[0], tb = textures[1];
    const sample = (tex: Texture, u: number, v: number, res: Float64Array) => {
      u -= T * Math.floor(u / T); v -= T * Math.floor(v / T);
      let x0 = u | 0, y0 = v | 0;
      if (x0 >= T) x0 = T - 1;
      if (y0 >= T) y0 = T - 1;
      const ax = u - x0, ay = v - y0, x1 = x0 + 1 === T ? 0 : x0 + 1, y1 = y0 + 1 === T ? 0 : y0 + 1;
      const a = y0 * T + x0, b = y0 * T + x1, c = y1 * T + x0, d = y1 * T + x1;
      const w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
      res[0] = tex.Y[a] * w00 + tex.Y[b] * w10 + tex.Y[c] * w01 + tex.Y[d] * w11;
      res[1] = tex.cb[a] * w00 + tex.cb[b] * w10 + tex.cb[c] * w01 + tex.cb[d] * w11;
      res[2] = tex.cr[a] * w00 + tex.cr[b] * w10 + tex.cr[c] * w01 + tex.cr[d] * w11;
    };
    const A = new Float64Array(3), B = new Float64Array(3);
    for (let y = 0; y < H; y++) {
      const fy = (y + 0.5) / GRID, j = Math.min(gh - 2, fy | 0), ay = fy - j;
      for (let x = 0; x < W; x++) {
        const fx = (x + 0.5) / GRID, ii = Math.min(gw - 2, fx | 0), ax = fx - ii;
        const g = j * gw + ii;
        const w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
        const lerp = (a: Float32Array) => a[g] * w00 + a[g + 1] * w10 + a[g + gw] * w01 + a[g + gw + 1] * w11;
        const m = lerp(gm);
        const e = (m + EDGE) / (2 * EDGE), wB = e <= 0 ? 0 : e >= 1 ? 1 : e * e * (3 - 2 * e);
        let Y: number, u: number, v: number;
        if (wB < 1) sample(ta, lerp(gu[0]), lerp(gv[0]), A);
        if (wB > 0) sample(tb, lerp(gu[1]), lerp(gv[1]), B);
        if (wB <= 0) { Y = A[0]; u = A[1]; v = A[2]; }
        else if (wB >= 1) { Y = B[0]; u = B[1]; v = B[2]; }
        else { Y = A[0] + (B[0] - A[0]) * wB; u = A[1] + (B[1] - A[1]) * wB; v = A[2] + (B[2] - A[2]) * wB; }
        // Turn the chroma by the hue field and add it around the luma, as
        // much as fits without clipping (so the luma stays exactly Y).
        const hc = lerp(gc), hs = lerp(gs), hl = Math.hypot(hc, hs) || 1;
        const cb = (u * hc - v * hs) / hl, cr = (u * hs + v * hc) / hl;
        const dR = 1.402 * cr, dG = -0.344136 * cb - 0.714136 * cr, dB = 1.772 * cb;
        let k = CHROMA;
        if (dR > 0) k = Math.min(k, (255 - Y) / dR); else if (dR < 0) k = Math.min(k, -Y / dR);
        if (dG > 0) k = Math.min(k, (255 - Y) / dG); else if (dG < 0) k = Math.min(k, -Y / dG);
        if (dB > 0) k = Math.min(k, (255 - Y) / dB); else if (dB < 0) k = Math.min(k, -Y / dB);
        const o = (y * W + x) * 4;
        out[o] = Y + k * dR; out[o + 1] = Y + k * dG; out[o + 2] = Y + k * dB; out[o + 3] = 255;
      }
    }
    return out;
  };

  const cache = new Map<number, GrayImage>();
  const frame = (i: number): GrayImage => {
    i = ((i % N) + N) % N;
    let g = cache.get(i);
    if (!g) cache.set(i, (g = rgbaToGray(frameRgba(i), W, H, createGray(W, H))));
    return g;
  };
  const scene: ShowcaseScene = { designWidth: DESIGN_W, designHeight: DESIGN_H, loop: LOOP, spin: centre.angle, zoom: centre.zoom, hueCycles: HUE_CYCLES };
  return { width: W, height: H, fps, frameCount: N, duration: N / fps, frame, frameRgba, scale: s, scene, warp };
}
