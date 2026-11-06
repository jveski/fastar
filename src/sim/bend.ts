// Bent paper for the simulator: a developable sheet whose cross-section
// across one in-plane direction follows a curvature profile while it stays
// straight along the other (a generalized cylinder). The bend never stretches
// the sheet, so every texel has an exact ground-truth position. Test-only.

export interface BendSpec {
  /** Direction the sheet bends across (radians from target x); it stays straight perpendicular to it. */
  angle: number;
  /**
   * Curvature (1/m) at signed arc length s (m) from the target centre along
   * the bend direction. Positive bends away from the camera (target +z, into
   * the target), negative toward it.
   */
  curvature: (s: number) => number;
}

/** A bent sheet at one instant: flat target metres <-> 3D target-frame points. */
export class SheetShape {
  readonly e1x: number;
  readonly e1y: number;
  private readonly s0: number;
  private readonly ds: number;
  private readonly n: number;
  private readonly x: Float64Array;
  private readonly z: Float64Array;
  private readonly phi: Float64Array;

  /** halfW, halfH: half the sheet's size (m). */
  constructor(readonly spec: BendSpec, readonly halfW: number, readonly halfH: number, samples = 1024) {
    const c = Math.cos(spec.angle), s = Math.sin(spec.angle);
    this.e1x = c;
    this.e1y = s;
    // Arc-length range the sheet spans along the bend direction.
    const L = Math.abs(halfW * c) + Math.abs(halfH * s) + 1e-3;
    const half = samples >> 1;
    const n = (this.n = 2 * half + 1);
    const ds = (this.ds = L / half);
    this.s0 = -half * ds;
    const x = (this.x = new Float64Array(n)), z = (this.z = new Float64Array(n)), phi = (this.phi = new Float64Array(n));
    const k = spec.curvature;
    for (let i = half + 1; i < n; i++) {
      const sm = this.s0 + (i - 0.5) * ds;
      phi[i] = phi[i - 1] + k(sm) * ds;
      const pm = 0.5 * (phi[i] + phi[i - 1]);
      x[i] = x[i - 1] + ds * Math.cos(pm);
      z[i] = z[i - 1] + ds * Math.sin(pm);
    }
    for (let i = half - 1; i >= 0; i--) {
      const sm = this.s0 + (i + 0.5) * ds;
      phi[i] = phi[i + 1] - k(sm) * ds;
      const pm = 0.5 * (phi[i] + phi[i + 1]);
      x[i] = x[i + 1] - ds * Math.cos(pm);
      z[i] = z[i + 1] - ds * Math.sin(pm);
    }
  }

  /** Cross-section at arc length a: [x, z, heading]. */
  private curve(a: number, out: Float64Array): void {
    let f = (a - this.s0) / this.ds;
    if (f < 0) f = 0;
    if (f > this.n - 1.000001) f = this.n - 1.000001;
    const i = f | 0, w = f - i;
    out[0] = this.x[i] + (this.x[i + 1] - this.x[i]) * w;
    out[1] = this.z[i] + (this.z[i + 1] - this.z[i]) * w;
    out[2] = this.phi[i] + (this.phi[i + 1] - this.phi[i]) * w;
  }

  private readonly cv = new Float64Array(3);

  /** 3D target-frame position of flat target point (X, Y) (m). */
  point(X: number, Y: number, out: Float64Array | Float32Array, o = 0): void {
    const c = this.e1x, s = this.e1y;
    const a = X * c + Y * s, b = -X * s + Y * c;
    const cv = this.cv;
    this.curve(a, cv);
    out[o] = cv[0] * c - b * s;
    out[o + 1] = cv[0] * s + b * c;
    out[o + 2] = cv[1];
  }

  /** Unit normal (target +z side) at flat point (X, Y). */
  normal(X: number, Y: number, out: Float64Array | Float32Array, o = 0): void {
    const c = this.e1x, s = this.e1y;
    this.curve(X * c + Y * s, this.cv);
    const ph = this.cv[2];
    out[o] = -Math.sin(ph) * c;
    out[o + 1] = -Math.sin(ph) * s;
    out[o + 2] = Math.cos(ph);
  }

  /**
   * Nearest hit of the ray o + lam d (target frame, lam > 0) with the sheet
   * inside its bounds. Writes [lam, X, Y, cos(incidence)] into out; returns
   * false on a miss.
   */
  intersect(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, out: Float64Array): boolean {
    const c = this.e1x, s = this.e1y;
    const oa = ox * c + oy * s, ob = -ox * s + oy * c;
    const da = dx * c + dy * s, db = -dx * s + dy * c;
    const d2 = da * da + dz * dz;
    if (d2 < 1e-18) return false;
    const x = this.x, z = this.z, n = this.n;
    const f = (i: number) => da * (z[i] - oz) - dz * (x[i] - oa);
    const STEP = 16;
    let best = Infinity, bs = 0, bb = 0, bi = 0, bw = 0;
    let fa = f(0);
    for (let i0 = 0; i0 < n - 1; i0 += STEP) {
      const i1 = Math.min(n - 1, i0 + STEP);
      const fb = f(i1);
      if ((fa <= 0) !== (fb <= 0)) {
        // Bisect to the crossing segment.
        let lo = i0, hi = i1, flo = fa;
        while (hi - lo > 1) {
          const m = (lo + hi) >> 1, fm = f(m);
          if ((flo <= 0) === (fm <= 0)) { lo = m; flo = fm; } else hi = m;
        }
        const fhi = f(hi);
        const w = flo === fhi ? 0 : flo / (flo - fhi);
        const px = x[lo] + (x[hi] - x[lo]) * w, pz = z[lo] + (z[hi] - z[lo]) * w;
        const lam = ((px - oa) * da + (pz - oz) * dz) / d2;
        if (lam > 0 && lam < best) {
          const sa = this.s0 + (lo + w) * this.ds, b = ob + lam * db;
          const X = sa * c - b * s, Y = sa * s + b * c;
          if (Math.abs(X) <= this.halfW && Math.abs(Y) <= this.halfH) {
            best = lam; bs = X; bb = Y; bi = lo; bw = w;
          }
        }
      }
      fa = fb;
    }
    if (best === Infinity) return false;
    const ph = this.phi[bi] + (this.phi[bi + 1] - this.phi[bi]) * bw;
    const nx = -Math.sin(ph) * c, ny = -Math.sin(ph) * s, nz = Math.cos(ph);
    out[0] = best;
    out[1] = bs;
    out[2] = bb;
    out[3] = Math.abs(nx * dx + ny * dy + nz * dz) / Math.sqrt(dx * dx + dy * dy + dz * dz);
    return true;
  }
}

/**
 * A bend that changes over time, memoized per time so a frame's rows share
 * one shape. spec(t) returns null while the sheet is flat.
 */
export function animatedBend(spec: (t: number) => BendSpec | null, halfW: number, halfH: number): (t: number) => SheetShape | null {
  const cache = new Map<number, SheetShape | null>();
  return (t) => {
    let s = cache.get(t);
    if (s === undefined) {
      const sp = spec(t);
      s = sp ? new SheetShape(sp, halfW, halfH) : null;
      if (cache.size > 64) cache.clear();
      cache.set(t, s);
    }
    return s;
  };
}

/** Common profiles. Curvatures in 1/m; s in m. */
export const bends = {
  /** Uniform curl: the whole sheet on one circular arc. */
  curl: (angle: number, k: number): BendSpec => ({ angle, curvature: () => k }),
  /** Flat up to `at` (m from the centre along the bend direction), then curling with curvature k (a lifted edge or corner). */
  lift: (angle: number, at: number, k: number, soft = 0.01): BendSpec => ({
    angle,
    curvature: (s) => k * smoothstep((s - at) / soft),
  }),
  /** Wave across the sheet. */
  wave: (angle: number, k: number, wavelength: number): BendSpec => ({
    angle,
    curvature: (s) => k * Math.sin((2 * Math.PI * s) / wavelength),
  }),
};

function smoothstep(x: number): number {
  const v = x < 0 ? 0 : x > 1 ? 1 : x;
  return v * v * (3 - 2 * v);
}
