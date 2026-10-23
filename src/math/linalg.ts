// Small dense linear algebra on row-major Float64Arrays. Sizes are tiny
// (<= 19), so straightforward loops beat any library and allocate nothing
// when callers pass scratch buffers.

/** In-place Cholesky (lower). Returns false if A is not positive definite. */
export function cholesky(A: Float64Array, n: number): boolean {
  for (let j = 0; j < n; j++) {
    let d = A[j * n + j];
    for (let k = 0; k < j; k++) d -= A[j * n + k] * A[j * n + k];
    if (!(d > 0)) return false;
    d = Math.sqrt(d);
    A[j * n + j] = d;
    const inv = 1 / d;
    for (let i = j + 1; i < n; i++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = s * inv;
    }
    for (let i = 0; i < j; i++) A[i * n + j] = 0;
  }
  return true;
}

/** Solve L L^T x = b in place given the Cholesky factor L. */
export function choleskySolve(L: Float64Array, n: number, b: Float64Array): void {
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * b[k];
    b[i] = s / L[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * b[k];
    b[i] = s / L[i * n + i];
  }
}

/**
 * Solve the symmetric positive definite system A x = b. A is overwritten with
 * its factor and b with x. Returns false if A is not SPD.
 */
export function solveSPD(A: Float64Array, n: number, b: Float64Array): boolean {
  if (!cholesky(A, n)) return false;
  choleskySolve(A, n, b);
  return true;
}

/** out = A^-1 for SPD A (A is destroyed). */
export function invertSPD(A: Float64Array, n: number, out: Float64Array, col: Float64Array = new Float64Array(n)): boolean {
  if (!cholesky(A, n)) return false;
  for (let j = 0; j < n; j++) {
    col.fill(0);
    col[j] = 1;
    choleskySolve(A, n, col);
    for (let i = 0; i < n; i++) out[i * n + j] = col[i];
  }
  return true;
}

/** Gaussian elimination with partial pivoting. A (n x n) and b are destroyed; b holds x. */
export function solveLinear(A: Float64Array, n: number, b: Float64Array): boolean {
  for (let c = 0; c < n; c++) {
    let p = c;
    let best = Math.abs(A[c * n + c]);
    for (let r = c + 1; r < n; r++) {
      const v = Math.abs(A[r * n + c]);
      if (v > best) { best = v; p = r; }
    }
    if (best < 1e-300) return false;
    if (p !== c) {
      for (let k = 0; k < n; k++) {
        const t = A[c * n + k]; A[c * n + k] = A[p * n + k]; A[p * n + k] = t;
      }
      const t = b[c]; b[c] = b[p]; b[p] = t;
    }
    const inv = 1 / A[c * n + c];
    for (let r = c + 1; r < n; r++) {
      const f = A[r * n + c] * inv;
      if (f === 0) continue;
      for (let k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
      b[r] -= f * b[c];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= A[r * n + k] * b[k];
    b[r] = s / A[r * n + r];
  }
  return true;
}

/**
 * Cyclic Jacobi eigen-decomposition of a symmetric matrix. A is destroyed.
 * Eigenvalues are returned ascending; eigenvector i is column i of V.
 */
export function eigenSym(A: Float64Array, n: number, values: Float64Array, V: Float64Array): void {
  V.fill(0);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const app = A[p * n + p], aqq = A[q * n + q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  for (let i = 0; i < n; i++) values[i] = A[i * n + i];
  // Selection sort ascending, permuting eigenvector columns.
  for (let i = 0; i < n; i++) {
    let m = i;
    for (let j = i + 1; j < n; j++) if (values[j] < values[m]) m = j;
    if (m !== i) {
      const t = values[i]; values[i] = values[m]; values[m] = t;
      for (let k = 0; k < n; k++) {
        const v = V[k * n + i]; V[k * n + i] = V[k * n + m]; V[k * n + m] = v;
      }
    }
  }
}
