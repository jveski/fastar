// Smooth evaluation of a deformed vertex grid (shared by the tracking worker
// and the main thread). Vertices are [x, y, z] per vertex, row-major (nx per
// row). The surface is the Catmull-Rom interpolation of the grid: it passes
// through every vertex and is smooth across cells; past the border the grid
// is extended linearly.

/** Vertex coordinate c of (i, j), extended linearly one vertex beyond the border. */
function at(V: ArrayLike<number>, nx: number, ny: number, i: number, j: number, c: number): number {
  if (i < 0) return 2 * at(V, nx, ny, 0, j, c) - at(V, nx, ny, 1, j, c);
  if (i >= nx) return 2 * at(V, nx, ny, nx - 1, j, c) - at(V, nx, ny, nx - 2, j, c);
  if (j < 0) return 2 * at(V, nx, ny, i, 0, c) - at(V, nx, ny, i, 1, c);
  if (j >= ny) return 2 * at(V, nx, ny, i, ny - 1, c) - at(V, nx, ny, i, ny - 2, c);
  return V[3 * (j * nx + i) + c];
}

const wu = new Float64Array(4), du = new Float64Array(4), wv = new Float64Array(4), dv = new Float64Array(4);

function weights(t: number, w: Float64Array, d: Float64Array): void {
  const t2 = t * t, t3 = t2 * t;
  w[0] = 0.5 * (-t3 + 2 * t2 - t);
  w[1] = 0.5 * (3 * t3 - 5 * t2 + 2);
  w[2] = 0.5 * (-3 * t3 + 4 * t2 + t);
  w[3] = 0.5 * (t3 - t2);
  d[0] = 0.5 * (-3 * t2 + 4 * t - 1);
  d[1] = 0.5 * (9 * t2 - 10 * t);
  d[2] = 0.5 * (-9 * t2 + 8 * t + 1);
  d[3] = 0.5 * (3 * t2 - 2 * t);
}

/**
 * Surface point at grid coordinates (gx, gy) (vertex (i, j) is at (i, j)),
 * into pos; with tu/tv, also the derivatives along gx and gy (per grid step).
 */
export function sampleGrid(
  V: ArrayLike<number>, nx: number, ny: number, gx: number, gy: number,
  pos: Float64Array | Float32Array, tu?: Float64Array | Float32Array, tv?: Float64Array | Float32Array, o = 0,
): void {
  let i = Math.floor(gx), j = Math.floor(gy);
  if (i < 0) i = 0; else if (i > nx - 2) i = nx - 2;
  if (j < 0) j = 0; else if (j > ny - 2) j = ny - 2;
  weights(gx - i, wu, du);
  weights(gy - j, wv, dv);
  // Inner vertices (no border extension) take the fast path.
  const inner = i >= 1 && j >= 1 && i + 2 < nx && j + 2 < ny;
  for (let c = 0; c < 3; c++) {
    let p = 0, a = 0, b = 0;
    for (let r = 0; r < 4; r++) {
      let row = 0, rowD = 0;
      for (let q = 0; q < 4; q++) {
        const x = inner ? V[3 * ((j - 1 + r) * nx + i - 1 + q) + c] : at(V, nx, ny, i - 1 + q, j - 1 + r, c);
        row += wu[q] * x;
        rowD += du[q] * x;
      }
      p += wv[r] * row;
      a += wv[r] * rowD;
      b += dv[r] * row;
    }
    pos[o + c] = p;
    if (tu) tu[o + c] = a;
    if (tv) tv[o + c] = b;
  }
}
