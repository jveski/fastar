// Tracking budget: every visible target is tracked, but the time spent per
// frame is bounded. Each target's tracking work scales with its number of
// active points, so points are what gets shared out: every target gets a
// floor, then the most important targets (largest, most in view, nearest
// the centre) get the rest in turn. When even the floors do not fit, the
// least important targets are updated only every 2nd or 3rd frame (the
// filter carries them in between).
import type { Intrinsics } from '../vision/camera';

export interface BudgetOptions {
  /** Tracking time per frame (ms). Default: `share` of the frame interval, less building the pyramid. */
  budgetMs?: number;
  /** Share of the frame interval spent tracking (the rest is left to the page and acquisition). */
  share: number;
  /** Most frames between updates of the least important targets. */
  maxEvery: number;
  /** Largest share of its points a target may lose in one frame (so its quality fades rather than jumps). */
  maxDrop: number;
  /** Priority bonus of the most important target of the last frame, so near-ties do not swap every frame. */
  stickiness: number;
}

export const DEFAULT_BUDGET: BudgetOptions = { share: 0.6, maxEvery: 3, maxDrop: 0.25, stickiness: 1.25 };

/**
 * Cost of tracking a target for a frame, ms = fixed + perPoint * points,
 * fitted online (exponentially forgotten least squares) on this device.
 */
export class CostModel {
  private w = 0;
  private sn = 0;
  private st = 0;
  private snn = 0;
  private snt = 0;

  constructor(private readonly priorFixed: number, private readonly priorPerPoint: number, private readonly forget = 0.05) {}

  observe(points: number, ms: number): void {
    if (!(ms >= 0) || !(points >= 0)) return;
    const k = 1 - this.forget;
    this.w = this.w * k + 1;
    this.sn = this.sn * k + points;
    this.st = this.st * k + ms;
    this.snn = this.snn * k + points * points;
    this.snt = this.snt * k + points * ms;
  }

  get samples(): number {
    return this.w;
  }

  get perPoint(): number {
    return this.fit()[1];
  }

  get fixed(): number {
    return this.fit()[0];
  }

  /** Predicted ms for `points`. */
  cost(points: number): number {
    const [f, c] = this.fit();
    return f + c * points;
  }

  private fit(): [number, number] {
    if (this.w < 3) return [this.priorFixed, this.priorPerPoint];
    const mn = this.sn / this.w, mt = this.st / this.w;
    const vn = this.snn / this.w - mn * mn, cnt = this.snt / this.w - mn * mt;
    // Allocations that hardly vary say nothing about the split: all per point.
    if (vn > (0.1 * mn) ** 2 + 1) {
      const c = cnt / vn, f = mt - c * mn;
      if (c > 0 && f >= 0) return [f, c];
    }
    return [0, Math.max(1e-5, mt / Math.max(1, mn))];
  }
}

export interface Demand {
  index: number;
  /** Importance (>= 0); see screenPriority. */
  priority: number;
  minPoints: number;
  maxPoints: number;
  /** Points allocated last frame (NaN if none). */
  current: number;
  fixed: number;
  perPoint: number;
}

export interface Allocation {
  index: number;
  points: number;
  /** Track on every `every`-th frame. */
  every: number;
  /** Predicted ms per frame (amortized over `every`). */
  cost: number;
}

/** Share out `budgetMs` among the demands (see the file comment). Sorted by priority, highest first. */
export function allocate(demands: Demand[], budgetMs: number, o: BudgetOptions = DEFAULT_BUDGET): Allocation[] {
  const d = demands.slice().sort((a, b) => b.priority - a.priority || a.index - b.index);
  const n = d.length;
  const every = new Array<number>(n).fill(1);
  const floor = (i: number) => (d[i].fixed + d[i].perPoint * d[i].minPoints) / every[i];
  const floors = () => {
    let s = 0;
    for (let i = 0; i < n; i++) s += floor(i);
    return s;
  };
  // Floors over budget: update the least important targets less often, one
  // step at a time from the bottom (never the most important one).
  for (let level = 2; level <= o.maxEvery && floors() > budgetMs; level++) {
    for (let i = n - 1; i >= 1 && floors() > budgetMs; i--) every[i] = level;
  }
  let left = budgetMs - floors();
  const out: Allocation[] = [];
  for (let i = 0; i < n; i++) {
    const x = d[i], per = x.perPoint / every[i];
    let points = x.minPoints;
    if (per <= 0) points = x.maxPoints;
    else if (left > 0) {
      const extra = Math.min(x.maxPoints - x.minPoints, left / per);
      points += extra;
      left -= extra * per;
    }
    if (Number.isFinite(x.current)) points = Math.max(points, x.current * (1 - o.maxDrop));
    points = Math.round(Math.min(x.maxPoints, Math.max(x.minPoints, points)));
    out.push({ index: x.index, points, every: every[i], cost: (x.fixed + x.perPoint * points) / every[i] });
  }
  return out;
}

export interface StrideOptions {
  /** Most camera frames per tracked frame. */
  maxStride: number;
  /** Share of tracked frames after which camera frames were dropped that counts as overload. */
  overload: number;
  /** Least time (s) between changes. */
  minDwell: number;
  /** Time (s) the device must have room for every frame before going back to it, and its cap. */
  hold: number;
  maxHold: number;
  /** Overloaded again this soon (s) after going back: wait twice as long next time. */
  relapse: number;
  /** Share of the frame interval a frame must fit in at full rate to go back to it. */
  fit: number;
}

export const DEFAULT_STRIDE: StrideOptions = { maxStride: 2, overload: 0.35, minDwell: 1, hold: 2, maxHold: 30, relapse: 10, fit: 0.75 };

/**
 * How many camera frames per tracked frame. A device that cannot keep up
 * with every frame drops whichever frame arrives while it is busy, so the
 * tracked frames come at uneven gaps (33, 66, 33, 100 ms...), which shows
 * as stutter. Tracking every 2nd frame on purpose keeps them even (and
 * leaves twice the time for each). Goes back to every frame only once the
 * work for it would fit with room to spare for a while, waiting longer
 * each time that turns out wrong.
 */
export class StrideControl {
  stride = 1;
  /** Share of recent tracked frames after which camera frames were dropped (0 to 1). */
  overload = 0;
  private readonly o: StrideOptions;
  private changedAt = -Infinity;
  private backAt = -Infinity;
  private fitSince = NaN;
  private hold: number;

  constructor(o: Partial<StrideOptions> = {}) {
    this.o = { ...DEFAULT_STRIDE, ...o };
    this.hold = this.o.hold;
  }

  /**
   * A tracked frame at time t (s), `gap` camera frames after the previous
   * one; `fullRateMs` is what a frame would cost at every frame (ms), and
   * `intervalMs` the camera frame interval.
   */
  observe(t: number, gap: number, fullRateMs: number, intervalMs: number): void {
    const o = this.o;
    if (gap >= 1) this.overload += 0.15 * ((gap > this.stride ? 1 : 0) - this.overload);
    if (t - this.changedAt < o.minDwell) return;
    if (this.stride < o.maxStride && this.overload > o.overload) {
      if (t - this.backAt < o.relapse) this.hold = Math.min(o.maxHold, 2 * this.hold);
      this.change(t, this.stride + 1);
    } else if (this.stride > 1) {
      if (!(fullRateMs <= o.fit * intervalMs)) this.fitSince = NaN;
      else if (Number.isNaN(this.fitSince)) this.fitSince = t;
      else if (t - this.fitSince >= this.hold) {
        this.change(t, this.stride - 1);
        this.backAt = t;
      }
    }
  }

  private change(t: number, stride: number): void {
    this.stride = stride;
    this.changedAt = t;
    this.overload = 0;
    this.fitSince = NaN;
  }
}

export interface ScreenInfo {
  /** Share of the image the target covers. */
  coverage: number;
  /** Share of the target (as projected) inside the image. */
  inside: number;
  /** Distance of the visible part's centre from the image centre, 0 (centre) to 1 (corner). */
  offCentre: number;
  /** Importance from the above, 0 (out of view) to 1 (filling the view). */
  score: number;
}

const NEAR = 1e-3;

/**
 * How prominent a target of half-size (hw, hh) metres, at pose (R, t)
 * (OpenCV: x right, y down, z into the target), is in the image: its outline
 * clipped to the image, by size first, then how much of it is in view and
 * how central it is.
 */
export function screenPriority(K: Intrinsics, R: ArrayLike<number>, t: ArrayLike<number>, hw: number, hh: number): ScreenInfo {
  // Corners in the camera, clipped to the near plane.
  let poly: number[] = [];
  for (const [X, Y] of [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]]) {
    poly.push(R[0] * X + R[1] * Y + t[0], R[3] * X + R[4] * Y + t[1], R[6] * X + R[7] * Y + t[2]);
  }
  poly = clipNear(poly);
  const none: ScreenInfo = { coverage: 0, inside: 0, offCentre: 1, score: 0 };
  if (poly.length < 9) return none;
  const img: number[] = [];
  for (let i = 0; i < poly.length; i += 3) img.push((K.fx * poly[i]) / poly[i + 2] + K.cx, (K.fy * poly[i + 1]) / poly[i + 2] + K.cy);
  const full = Math.abs(area(img));
  const clipped = clipRect(img, K.width, K.height);
  const a = Math.abs(area(clipped));
  if (!(a > 0) || !(full > 0)) return none;
  const coverage = Math.min(1, a / (K.width * K.height));
  const inside = Math.min(1, a / full);
  const [cx, cy] = centroid(clipped);
  const offCentre = Math.min(1, Math.hypot(cx - K.width / 2, cy - K.height / 2) / Math.hypot(K.width / 2, K.height / 2));
  const score = Math.sqrt(coverage) * (0.5 + 0.5 * inside) * (1 - 0.5 * offCentre * offCentre);
  return { coverage, inside, offCentre, score };
}

function clipNear(p: number[]): number[] {
  const out: number[] = [];
  const n = p.length / 3;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const za = p[3 * i + 2], zb = p[3 * j + 2];
    const ina = za >= NEAR, inb = zb >= NEAR;
    if (ina) out.push(p[3 * i], p[3 * i + 1], za);
    if (ina !== inb) {
      const s = (NEAR - za) / (zb - za);
      out.push(p[3 * i] + s * (p[3 * j] - p[3 * i]), p[3 * i + 1] + s * (p[3 * j + 1] - p[3 * i + 1]), NEAR);
    }
  }
  return out;
}

/** Sutherland-Hodgman against [0, w] x [0, h]. */
function clipRect(p: number[], w: number, h: number): number[] {
  const edges: Array<[number, number, number]> = [[0, 1, 0], [0, -1, w], [1, 1, 0], [1, -1, h]];
  let poly = p;
  for (const [axis, sign, off] of edges) {
    const n = poly.length / 2;
    if (!n) break;
    const out: number[] = [];
    const f = (i: number) => sign * poly[2 * i + axis] + off;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n, fa = f(i), fb = f(j);
      if (fa >= 0) out.push(poly[2 * i], poly[2 * i + 1]);
      if ((fa >= 0) !== (fb >= 0)) {
        const s = fa / (fa - fb);
        out.push(poly[2 * i] + s * (poly[2 * j] - poly[2 * i]), poly[2 * i + 1] + s * (poly[2 * j + 1] - poly[2 * i + 1]));
      }
    }
    poly = out;
  }
  return poly;
}

function area(p: number[]): number {
  const n = p.length / 2;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    s += p[2 * i] * p[2 * j + 1] - p[2 * j] * p[2 * i + 1];
  }
  return s / 2;
}

function centroid(p: number[]): [number, number] {
  const n = p.length / 2;
  let a = 0, cx = 0, cy = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const c = p[2 * i] * p[2 * j + 1] - p[2 * j] * p[2 * i + 1];
    a += c;
    cx += (p[2 * i] + p[2 * j]) * c;
    cy += (p[2 * i + 1] + p[2 * j + 1]) * c;
  }
  if (Math.abs(a) < 1e-9) {
    let sx = 0, sy = 0;
    for (let i = 0; i < n; i++) { sx += p[2 * i]; sy += p[2 * i + 1]; }
    return [sx / n, sy / n];
  }
  return [cx / (3 * a), cy / (3 * a)];
}
