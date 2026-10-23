/** k-th smallest of a[0..n) via in-place quickselect (a is reordered). */
export function quickselect(a: Float32Array | Float64Array, n: number, k: number): number {
  let lo = 0, hi = n - 1;
  while (lo < hi) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) {
        const t = a[i]; a[i] = a[j]; a[j] = t;
        i++; j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else break;
  }
  return a[k];
}

export function median(a: Float32Array | Float64Array, n: number): number {
  return n === 0 ? 0 : quickselect(a, n, n >> 1);
}
