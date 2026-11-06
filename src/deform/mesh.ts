// Triangle mesh over a target, for tracking it while it bends: a regular grid
// of nx x ny vertices spanning the target (target metres, origin at the
// centre, x right, y down), each cell split into two triangles along its
// (i, j)-(i+1, j+1) diagonal. Points on the target are bound to a triangle by
// barycentric weights, which stay fixed however the mesh deforms.
import type { TargetMeta } from '../target/format';

export class SurfaceMesh {
  readonly count: number;
  readonly faceCount: number;
  /** Rest (flat) vertex positions [X, Y, 0], target metres. */
  readonly rest: Float64Array;
  /** Vertex indices [a, b, c] per face. */
  readonly faces: Int32Array;
  /** Per face: inverse of the rest edge matrix [r_b - r_a, r_c - r_a] (2x2 row-major). */
  readonly faceInv: Float64Array;
  /** Cell size (m). */
  readonly cw: number;
  readonly ch: number;

  constructor(readonly width: number, readonly height: number, readonly nx: number, readonly ny: number) {
    if (nx < 3 || ny < 3) throw new Error('fastar: surface grid needs at least 3 x 3 vertices');
    this.count = nx * ny;
    this.cw = width / (nx - 1);
    this.ch = height / (ny - 1);
    const rest = (this.rest = new Float64Array(3 * this.count));
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const v = j * nx + i;
        rest[3 * v] = -width / 2 + i * this.cw;
        rest[3 * v + 1] = -height / 2 + j * this.ch;
      }
    }
    this.faceCount = 2 * (nx - 1) * (ny - 1);
    const faces = (this.faces = new Int32Array(3 * this.faceCount));
    const inv = (this.faceInv = new Float64Array(4 * this.faceCount));
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const v00 = j * nx + i, v10 = v00 + 1, v01 = v00 + nx, v11 = v01 + 1;
        const f = 2 * (j * (nx - 1) + i);
        faces.set([v00, v10, v11], 3 * f);
        faces.set([v00, v11, v01], 3 * f + 3);
      }
    }
    for (let f = 0; f < this.faceCount; f++) {
      const a = faces[3 * f], b = faces[3 * f + 1], c = faces[3 * f + 2];
      const e1x = rest[3 * b] - rest[3 * a], e1y = rest[3 * b + 1] - rest[3 * a + 1];
      const e2x = rest[3 * c] - rest[3 * a], e2y = rest[3 * c + 1] - rest[3 * a + 1];
      const det = e1x * e2y - e2x * e1y;
      inv[4 * f] = e2y / det; inv[4 * f + 1] = -e2x / det;
      inv[4 * f + 2] = -e1y / det; inv[4 * f + 3] = e1x / det;
    }
  }

  /** Mesh for a target with `long` vertices along its long side and square-ish cells. */
  static forTarget(meta: TargetMeta, long = 9): SurfaceMesh {
    const W = meta.physicalWidth, H = (W * meta.height) / meta.width;
    const short = (n: number) => Math.max(3, Math.round(((long - 1) * n) + 1));
    return W >= H ? new SurfaceMesh(W, H, long, short(H / W)) : new SurfaceMesh(W, H, short(W / H), long);
  }

  /** Grid coordinates (vertex (i, j) at (i, j)) of flat target point (X, Y). */
  gridX(X: number): number {
    return (X + this.width / 2) / this.cw;
  }

  gridY(Y: number): number {
    return (Y + this.height / 2) / this.ch;
  }

  /** Face containing flat target point (X, Y) (the nearest border face outside the mesh). */
  faceAt(X: number, Y: number): number {
    const gx = this.gridX(X), gy = this.gridY(Y);
    let i = Math.floor(gx), j = Math.floor(gy);
    if (i < 0) i = 0; else if (i > this.nx - 2) i = this.nx - 2;
    if (j < 0) j = 0; else if (j > this.ny - 2) j = this.ny - 2;
    return 2 * (j * (this.nx - 1) + i) + (gx - i >= gy - j ? 0 : 1);
  }

  /**
   * Bind flat point (X, Y): its face's vertex indices into idx[o..o+2] and
   * barycentric weights into w[o..o+2]. Returns the face.
   */
  bind(X: number, Y: number, idx: Int32Array, w: Float32Array | Float64Array, o: number): number {
    const f = this.faceAt(X, Y);
    const a = this.faces[3 * f], r = this.rest, inv = this.faceInv;
    const dx = X - r[3 * a], dy = Y - r[3 * a + 1];
    const al = inv[4 * f] * dx + inv[4 * f + 1] * dy, be = inv[4 * f + 2] * dx + inv[4 * f + 3] * dy;
    idx[o] = a; idx[o + 1] = this.faces[3 * f + 1]; idx[o + 2] = this.faces[3 * f + 2];
    w[o] = 1 - al - be; w[o + 1] = al; w[o + 2] = be;
    return f;
  }
}
