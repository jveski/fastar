import { blur5, createGray, resizeArea, type GrayImage } from '../image/gray';
import { FastDetector } from '../vision/fast';
import { FeatureSet, describeLevel, hamming, ORB_BORDER } from '../vision/orb';
import { buildKmTree } from '../vision/kmtree';
import { encodeTargets, type QualityReport, type TargetData, type TargetImageInfo } from './format';

export interface CompileOptions {
  name?: string;
  /** Physical width of the printed target in metres. */
  physicalWidth: number;
  /** Long side of the base image (larger inputs are downscaled). */
  maxDim?: number;
  /** Smallest pyramid level long side. */
  minLevelSize?: number;
  maxPerLevel?: number;
  fastThreshold?: number;
  trackPointCount?: number;
}

const LEVEL_STEP = Math.pow(2, 1 / 3);
const TRACK_SIZES = [256, 512, 1024];

export function compileTarget(input: GrayImage, o: CompileOptions): TargetData {
  if (!(o.physicalWidth > 0)) throw new Error('physicalWidth must be > 0 (metres)');
  const maxDim = o.maxDim ?? 1024;
  const long0 = Math.max(input.width, input.height);
  const base = long0 > maxDim
    ? resizeArea(input, Math.round((input.width * maxDim) / long0), Math.round((input.height * maxDim) / long0))
    : input;
  const W = base.width, H = base.height, long = Math.max(W, H);

  // Matching data over a 2^(1/3) scale pyramid.
  const feats = new FeatureSet(4096);
  const det = new FastDetector(W, H, 60000, o.maxPerLevel ?? 500);
  const levelScales: number[] = [];
  const perLevel: number[] = [];
  const minLevel = o.minLevelSize ?? 100;
  for (let k = 0; ; k++) {
    const s = Math.pow(LEVEL_STEP, -k);
    const lw = Math.round(W * s), lh = Math.round(H * s);
    if (Math.max(lw, lh) < minLevel || Math.min(lw, lh) < 2 * ORB_BORDER + 8) break;
    const lev = k === 0 ? base : resizeArea(base, lw, lh);
    const smooth = blur5(lev, createGray(lw, lh));
    const n = det.detect(lev, {
      threshold: o.fastThreshold ?? 20,
      maxPoints: o.maxPerLevel ?? 500,
      border: ORB_BORDER,
      cell: Math.max(16, Math.round(Math.max(lw, lh) / 12)),
    });
    const before = feats.count;
    const scale = W / lw;
    describeLevel(lev, smooth, det.x, det.y, n, scale, 0.5, feats);
    levelScales.push(scale);
    perLevel.push(feats.count - before);
  }
  const kp = feats.kp.slice(0, feats.count * 4);
  const desc = feats.desc.slice(0, feats.count * 8);
  const tree = buildKmTree(desc, feats.count);

  // Tracking images at 256/512/1024 (capped by the base size).
  const images: Uint8Array[] = [];
  const infos: TargetImageInfo[] = [];
  for (const S of TRACK_SIZES) {
    if (S < long) {
      const img = resizeArea(base, Math.round((W * S) / long), Math.round((H * S) / long));
      images.push(img.data);
      infos.push({ width: img.width, height: img.height, scale: W / img.width });
    } else {
      images.push(base.data.slice());
      infos.push({ width: W, height: H, scale: 1 });
      break;
    }
  }

  // Tracking points chosen on the ~512 px image.
  const ti = Math.min(1, infos.length - 1);
  const timg = createGray(infos[ti].width, infos[ti].height, images[ti]);
  const tp = selectTrackingPoints(timg, o.trackPointCount ?? 200);
  const trackPoints = new Float32Array(tp.length);
  const ts = infos[ti].scale;
  for (let i = 0; i < tp.length; i += 2) {
    trackPoints[i] = (tp[i] + 0.5) * ts - 0.5;
    trackPoints[i + 1] = (tp[i + 1] + 0.5) * ts - 0.5;
  }

  const quality = qualityReport(kp, desc, feats.count, perLevel, W, H, trackPoints.length / 2);
  return {
    meta: {
      name: o.name ?? 'target',
      width: W,
      height: H,
      physicalWidth: o.physicalWidth,
      levelScales,
      images: infos,
      quality,
    },
    kp, desc, tree, images, trackPoints,
  };
}

export function compileTargets(list: Array<{ image: GrayImage } & CompileOptions>): { buffer: ArrayBuffer; targets: TargetData[] } {
  const targets = list.map((t) => compileTarget(t.image, t));
  return { buffer: encodeTargets(targets), targets };
}

/**
 * Shi-Tomasi points (min eigenvalue over a 9x9 window), spread on a grid,
 * rejecting points whose patch resembles nearby patches (self-NCC).
 */
export function selectTrackingPoints(img: GrayImage, count: number): number[] {
  const w = img.width, h = img.height;
  const margin = 14;
  const { cands, score } = cornerCandidates(img, margin);
  cands.sort((a, b) => score[b] - score[a]);
  const cell = Math.max(8, Math.floor(Math.sqrt(((w - 2 * margin) * (h - 2 * margin)) / (count * 1.2))));
  const gw = Math.ceil(w / cell);
  const used = new Map<number, number>();
  const out: number[] = [];
  for (let pass = 0; pass < 2 && out.length < 2 * count; pass++) {
    for (const p of cands) {
      if (out.length >= 2 * count) break;
      if (p < 0) continue;
      const x = p % w, y = (p / w) | 0;
      const c = ((y / cell) | 0) * gw + ((x / cell) | 0);
      if ((used.get(c) ?? 0) >= (pass === 0 ? 1 : 3)) continue;
      if (selfSimilar(img, x, y)) continue;
      // Keep points apart.
      let near = false;
      for (let k = 0; k < out.length; k += 2) if (Math.abs(out[k] - x) < 5 && Math.abs(out[k + 1] - y) < 5) { near = true; break; }
      if (near) continue;
      used.set(c, (used.get(c) ?? 0) + 1);
      out.push(x, y);
    }
  }
  return out;
}

/**
 * Shi-Tomasi score (min eigenvalue of the 9x9 structure tensor) and the 5x5
 * local maxima above ~5 grey levels/px in the weakest direction.
 */
export function cornerCandidates(img: GrayImage, margin: number): { cands: number[]; score: Float32Array } {
  const w = img.width, h = img.height, d = img.data;
  const R = 4;
  const n = w * h;
  const ixx = new Float64Array((w + 1) * (h + 1)), iyy = new Float64Array((w + 1) * (h + 1)), ixy = new Float64Array((w + 1) * (h + 1));
  const W1 = w + 1;
  for (let y = 0; y < h; y++) {
    let sxx = 0, syy = 0, sxy = 0;
    for (let x = 0; x < w; x++) {
      let gx = 0, gy = 0;
      if (x > 0 && x < w - 1 && y > 0 && y < h - 1) {
        const p = y * w + x;
        gx = (d[p + 1] - d[p - 1]) * 0.5;
        gy = (d[p + w] - d[p - w]) * 0.5;
      }
      sxx += gx * gx; syy += gy * gy; sxy += gx * gy;
      const o = (y + 1) * W1 + x + 1;
      ixx[o] = ixx[o - W1] + sxx; iyy[o] = iyy[o - W1] + syy; ixy[o] = ixy[o - W1] + sxy;
    }
  }
  const box = (S: Float64Array, x: number, y: number) => {
    const a = (y - R) * W1 + (x - R), b = (y - R) * W1 + (x + R + 1), c = (y + R + 1) * W1 + (x - R), e = (y + R + 1) * W1 + (x + R + 1);
    return S[e] - S[b] - S[c] + S[a];
  };
  const score = new Float32Array(n);
  for (let y = margin; y < h - margin; y++) {
    for (let x = margin; x < w - margin; x++) {
      const a = box(ixx, x, y), c = box(iyy, x, y), b = box(ixy, x, y);
      const tr = (a + c) / 2;
      score[y * w + x] = tr - Math.sqrt(Math.max(0, tr * tr - (a * c - b * b)));
    }
  }
  const cands: number[] = [];
  for (let y = margin; y < h - margin; y++) {
    for (let x = margin; x < w - margin; x++) {
      const p = y * w + x, s = score[p];
      if (s < 81 * 25) continue; // weakest gradient direction >= ~5 grey levels/px
      let isMax = true;
      for (let j = -2; j <= 2 && isMax; j++) for (let i = -2; i <= 2; i++) {
        if ((i || j) && score[p + j * w + i] > s) { isMax = false; break; }
      }
      if (isMax) cands.push(p);
    }
  }
  return { cands, score };
}

/** True if the 9x9 patch at (x, y) matches a shifted patch 3..8 px away with NCC > 0.85. */
export function selfSimilar(img: GrayImage, x: number, y: number): boolean {
  const w = img.width, d = img.data;
  const R = 4, N = 81;
  let s = 0, s2 = 0;
  for (let j = -R; j <= R; j++) for (let i = -R; i <= R; i++) { const v = d[(y + j) * w + x + i]; s += v; s2 += v * v; }
  const m = s / N, va = s2 - s * m;
  if (va <= 0) return true;
  for (const r of [3, 5, 8]) {
    for (let k = 0; k < 16; k++) {
      const a = (k * Math.PI) / 8;
      const ox = Math.round(r * Math.cos(a)), oy = Math.round(r * Math.sin(a));
      const cx = x + ox, cy = y + oy;
      if (cx - R < 0 || cy - R < 0 || cx + R >= img.width || cy + R >= img.height) continue;
      let t = 0, t2 = 0, st = 0;
      for (let j = -R; j <= R; j++) for (let i = -R; i <= R; i++) {
        const u = d[(cy + j) * w + cx + i], v = d[(y + j) * w + x + i];
        t += u; t2 += u * u; st += u * v;
      }
      const vb = t2 - (t * t) / N;
      if (vb <= 0) continue;
      const ncc = (st - (s * t) / N) / Math.sqrt(va * vb);
      if (ncc > 0.85) return true;
    }
  }
  return false;
}

function qualityReport(
  kp: Float32Array, desc: Uint32Array, n: number, perLevel: number[], W: number, H: number, trackingPoints: number,
): QualityReport {
  const grid = new Uint8Array(64);
  for (let i = 0; i < n; i++) {
    const gx = Math.min(7, Math.max(0, Math.floor((kp[4 * i] / W) * 8)));
    const gy = Math.min(7, Math.max(0, Math.floor((kp[4 * i + 1] / H) * 8)));
    grid[gy * 8 + gx] = 1;
  }
  const coverage = grid.reduce((a, b) => a + b, 0) / 64;
  const selfSim = selfSimilarity(kp, desc, n, W, H);
  const warnings: string[] = [];
  if (n < 150) warnings.push(`few features (${n}); use a more detailed image`);
  if (coverage < 0.5) warnings.push(`features cover only ${(coverage * 100).toFixed(0)}% of the target`);
  if (selfSim > 0.25) warnings.push(`repetitive texture (${(selfSim * 100).toFixed(0)}% of features have look-alikes)`);
  if (trackingPoints < 60) warnings.push(`only ${trackingPoints} tracking points; tracking may be unstable`);
  return { features: n, featuresPerLevel: perLevel, coverage, selfSimilarity: selfSim, trackingPoints, warnings };
}

/** Fraction of a deterministic sample of descriptors with a near-identical twin elsewhere on the image. */
export function selfSimilarity(kp: Float32Array, desc: Uint32Array, n: number, W: number, H: number): number {
  const sample = Math.min(n, 300);
  let similar = 0;
  const minSep = 0.05 * Math.max(W, H);
  for (let k = 0; k < sample; k++) {
    const i = Math.floor((k * n) / sample);
    let best = 257;
    for (let j = 0; j < n; j++) {
      if (Math.hypot(kp[4 * i] - kp[4 * j], kp[4 * i + 1] - kp[4 * j + 1]) < minSep) continue;
      const dd = hamming(desc, 8 * i, desc, 8 * j);
      if (dd < best) best = dd;
    }
    if (best < 24) similar++;
  }
  return sample ? similar / sample : 0;
}
