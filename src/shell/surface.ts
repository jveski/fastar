// A bending target's shape as seen by the app: a grid of vertices in the
// target's frame (the frame of Pose.matrix), with smooth sampling and helpers
// to fill render geometry.
import { sampleGrid } from '../deform/grid';

/** A point on a surface: position and unit normal, and tangents per metre along the sheet. */
export interface SurfacePoint {
  position: Float32Array;
  normal: Float32Array;
  /** Change of position per metre moved right along the sheet (unit length while it does not stretch). */
  tangentU: Float32Array;
  /** Change of position per metre moved up along the sheet. */
  tangentV: Float32Array;
}

export function surfacePoint(): SurfacePoint {
  return { position: new Float32Array(3), normal: new Float32Array(3), tangentU: new Float32Array(3), tangentV: new Float32Array(3) };
}

/**
 * The shape of a bending target in one frame. Coordinates are those of
 * Pose.matrix: metres, origin at the centre of the flat target, x right, y up,
 * z out of the printed side. The surface is the smooth (Catmull-Rom)
 * interpolation of the vertex grid; (u, v) address it like the target
 * image's texture coordinates: u from 0 at the left edge to 1 at the right,
 * v from 0 at the bottom to 1 at the top.
 */
export class Surface {
  private readonly pos = new Float64Array(3);
  private readonly tu = new Float64Array(3);
  private readonly tv = new Float64Array(3);

  /**
   * cols x rows vertices, row-major from the top-left corner of the target
   * ([x, y, z] each); width x height: the flat target's size (m).
   */
  constructor(
    readonly cols: number,
    readonly rows: number,
    readonly width: number,
    readonly height: number,
    readonly positions: Float32Array,
  ) {}

  /** Surface point at texture coordinates (u, v); outside [0, 1] the sheet is extended smoothly. */
  sample(u: number, v: number, out: SurfacePoint = surfacePoint()): SurfacePoint {
    const gx = u * (this.cols - 1), gy = (1 - v) * (this.rows - 1);
    sampleGrid(this.positions, this.cols, this.rows, gx, gy, this.pos, this.tu, this.tv);
    // Per grid step to per metre; v runs up, rows down.
    const su = (this.cols - 1) / this.width, sv = -(this.rows - 1) / this.height;
    const p = out.position, a = out.tangentU, b = out.tangentV, n = out.normal;
    for (let k = 0; k < 3; k++) {
      p[k] = this.pos[k];
      a[k] = this.tu[k] * su;
      b[k] = this.tv[k] * sv;
    }
    const nx = a[1] * b[2] - a[2] * b[1], ny = a[2] * b[0] - a[0] * b[2], nz = a[0] * b[1] - a[1] * b[0];
    const l = Math.hypot(nx, ny, nz) || 1;
    n[0] = nx / l; n[1] = ny / l; n[2] = nz / l;
    return out;
  }

  /**
   * Fill a render grid of (segX + 1) x (segY + 1) vertices, row-major from
   * the bottom-left (as gridIndices expects): positions and optionally
   * normals ([x, y, z] each) and texture coordinates ([u, v]).
   */
  fillGrid(segX: number, segY: number, positions: Float32Array, normals?: Float32Array | null, uvs?: Float32Array | null): void {
    const pt = scratch;
    let k = 0;
    for (let j = 0; j <= segY; j++) {
      const v = j / segY;
      for (let i = 0; i <= segX; i++, k++) {
        const u = i / segX;
        this.sample(u, v, pt);
        positions.set(pt.position, 3 * k);
        if (normals) normals.set(pt.normal, 3 * k);
        if (uvs) { uvs[2 * k] = u; uvs[2 * k + 1] = v; }
      }
    }
  }

  /** Triangle indices for a fillGrid grid, wound counter-clockwise seen from the printed side. */
  static gridIndices(segX: number, segY: number): Uint16Array | Uint32Array {
    const n = (segX + 1) * (segY + 1);
    const out = n > 65535 ? new Uint32Array(6 * segX * segY) : new Uint16Array(6 * segX * segY);
    let k = 0;
    for (let j = 0; j < segY; j++) {
      for (let i = 0; i < segX; i++) {
        const a = j * (segX + 1) + i, b = a + 1, c = a + segX + 1, d = c + 1;
        out[k++] = a; out[k++] = b; out[k++] = d;
        out[k++] = a; out[k++] = d; out[k++] = c;
      }
    }
    return out;
  }

  /** A flat surface of the given size (what a rigid target looks like). */
  static flat(cols: number, rows: number, width: number, height: number): Surface {
    const p = new Float32Array(3 * cols * rows);
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const v = 3 * (j * cols + i);
        p[v] = -width / 2 + (i * width) / (cols - 1);
        p[v + 1] = height / 2 - (j * height) / (rows - 1);
      }
    }
    return new Surface(cols, rows, width, height, p);
  }
}

const scratch = surfacePoint();

/** Shapes from the worker for one target, served per presented frame with light smoothing. */
export class SurfaceFeed {
  private readonly history: Array<{ t: number; vertices: Float32Array }> = [];
  private shown: Float32Array | null = null;
  private shownTime = -Infinity;

  /**
   * keep: share of a small shape change still pending after one frame (0 =
   * no smoothing); changes over a few millimetres pass at once.
   */
  constructor(readonly cols: number, readonly rows: number, readonly width: number, readonly height: number, private readonly keep = 0.5) {}

  /** A shape measured on the frame captured at t (ms): OpenCV target axes (x right, y down, z into the target). */
  push(t: number, vertices: Float32Array): void {
    const h = this.history;
    // Stored in the app's axes (y up, z out of the target).
    for (let i = 0; i < vertices.length; i += 3) { vertices[i + 1] = -vertices[i + 1]; vertices[i + 2] = -vertices[i + 2]; }
    h.push({ t, vertices });
    if (h.length > 8) h.shift();
  }

  /** The shape to show with the frame captured at t (ms): the newest measured on or before it. */
  at(t: number): Surface | null {
    const h = this.history;
    if (!h.length) return null;
    let e = h[0];
    for (const x of h) if (x.t <= t + 0.5) e = x;
    const n = e.vertices.length;
    let s = this.shown;
    if (!s || s.length !== n || !(t - this.shownTime >= 0 && t - this.shownTime < 250)) {
      s = this.shown = e.vertices.slice();
    } else {
      for (let i = 0; i < n; i += 3) {
        const dx = e.vertices[i] - s[i], dy = e.vertices[i + 1] - s[i + 1], dz = e.vertices[i + 2] - s[i + 2];
        // Below 1 mm: mostly noise, eased; from 5 mm on: real bending, shown at once.
        const d = Math.hypot(dx, dy, dz), r = Math.min(1, Math.max(0, (d - 0.001) / 0.004));
        const a = 1 - this.keep * (1 - r);
        s[i] += a * dx; s[i + 1] += a * dy; s[i + 2] += a * dz;
      }
    }
    this.shownTime = t;
    return new Surface(this.cols, this.rows, this.width, this.height, s.slice());
  }
}
