import { hamming } from './orb';
import { Rng } from '../math/rng';

// Flattened tree in an Int32Array.
//   internal node: [k, (medoidIndex, childOffset) * k]
//   leaf:          [-m, featureIndex * m]
// The root is at offset 0.

export interface KmTreeOptions {
  branching: number;
  leafSize: number;
  iterations: number;
  seed: number;
}

export const DEFAULT_KMTREE: KmTreeOptions = { branching: 8, leafSize: 16, iterations: 3, seed: 1234 };

export function buildKmTree(desc: Uint32Array, count: number, o: KmTreeOptions = DEFAULT_KMTREE): Int32Array {
  const rng = new Rng(o.seed);
  const out: number[] = [];
  const all = new Int32Array(count);
  for (let i = 0; i < count; i++) all[i] = i;

  const build = (idx: Int32Array): number => {
    const at = out.length;
    if (idx.length <= o.leafSize || idx.length <= o.branching) {
      out.push(-idx.length);
      for (const i of idx) out.push(i);
      return at;
    }
    const k = o.branching;
    // k-means++ style seeding on Hamming distance.
    const medoids: number[] = [idx[rng.int(idx.length)]];
    const dmin = new Float64Array(idx.length).fill(Infinity);
    while (medoids.length < k) {
      const last = medoids[medoids.length - 1];
      let total = 0;
      for (let i = 0; i < idx.length; i++) {
        const dd = hamming(desc, 8 * idx[i], desc, 8 * last);
        if (dd < dmin[i]) dmin[i] = dd;
        total += dmin[i] * dmin[i];
      }
      if (total === 0) break;
      let r = rng.next() * total, pick = idx.length - 1;
      for (let i = 0; i < idx.length; i++) {
        r -= dmin[i] * dmin[i];
        if (r <= 0) { pick = i; break; }
      }
      medoids.push(idx[pick]);
    }
    const assign = new Int32Array(idx.length);
    for (let it = 0; it <= o.iterations; it++) {
      for (let i = 0; i < idx.length; i++) {
        let best = Infinity, bj = 0;
        for (let j = 0; j < medoids.length; j++) {
          const dd = hamming(desc, 8 * idx[i], desc, 8 * medoids[j]);
          if (dd < best) { best = dd; bj = j; }
        }
        assign[i] = bj;
      }
      if (it === o.iterations) break;
      // Update each medoid to the (sampled) member minimizing total distance.
      for (let j = 0; j < medoids.length; j++) {
        const members: number[] = [];
        for (let i = 0; i < idx.length; i++) if (assign[i] === j) members.push(idx[i]);
        if (members.length === 0) continue;
        const cands = members.length > 48 ? Array.from({ length: 48 }, () => members[rng.int(members.length)]) : members;
        const ref = members.length > 128 ? Array.from({ length: 128 }, () => members[rng.int(members.length)]) : members;
        let best = Infinity, bm = medoids[j];
        for (const c of cands) {
          let s = 0;
          for (const m of ref) s += hamming(desc, 8 * c, desc, 8 * m);
          if (s < best) { best = s; bm = c; }
        }
        medoids[j] = bm;
      }
    }
    const groups: number[][] = medoids.map(() => []);
    for (let i = 0; i < idx.length; i++) groups[assign[i]].push(idx[i]);
    const nonEmpty = medoids.map((m, j) => ({ m, g: groups[j] })).filter((e) => e.g.length > 0);
    if (nonEmpty.length <= 1) {
      // Degenerate (identical descriptors): make a leaf.
      out.push(-idx.length);
      for (const i of idx) out.push(i);
      return at;
    }
    out.push(nonEmpty.length);
    const slots = out.length;
    for (const e of nonEmpty) out.push(e.m, 0);
    nonEmpty.forEach((e, j) => {
      const child = build(Int32Array.from(e.g));
      out[slots + 2 * j + 1] = child;
    });
    return at;
  };

  if (count > 0) build(all);
  else out.push(0);
  return Int32Array.from(out);
}

/**
 * Approximate nearest-neighbour search with best-bin-first backtracking.
 * Reports the two best distances; `exclude(i)` lets callers treat
 * near-duplicate features (same location, different scale) as one.
 */
export class KmTreeSearcher {
  best1 = -1;
  dist1 = 0;
  dist2 = 0;
  private heapNode: Int32Array;
  private heapKey: Int32Array;
  private heapSize = 0;

  constructor(readonly maxChecks = 96, heapCapacity = 4096) {
    this.heapNode = new Int32Array(heapCapacity);
    this.heapKey = new Int32Array(heapCapacity);
  }

  /**
   * kp = [x, y, scale, angle] per target feature; second-best candidates
   * within `dupRadius` base pixels of the best are ignored for the ratio test,
   * and so are those for which `covisible(best, other)` is false (video
   * targets: the same content seen at another time, somewhere else).
   */
  search(
    tree: Int32Array, tdesc: Uint32Array, kp: Float32Array,
    q: Uint32Array, qo: number, dupRadius = 6,
    covisible?: (a: number, b: number) => boolean,
  ): number {
    this.best1 = -1;
    this.dist1 = 257;
    this.dist2 = 257;
    this.heapSize = 0;
    let checks = 0;
    let node = 0;
    const r2 = dupRadius * dupRadius;
    const rival = (f: number, b: number): boolean => {
      const dx = kp[4 * f] - kp[4 * b], dy = kp[4 * f + 1] - kp[4 * b + 1];
      return dx * dx + dy * dy > r2 && (!covisible || covisible(f, b));
    };
    for (;;) {
      // Descend greedily, pushing siblings into the heap.
      while (tree[node] > 0) {
        const k = tree[node];
        let bestD = 1e9, bestChild = -1;
        for (let j = 0; j < k; j++) {
          const m = tree[node + 1 + 2 * j];
          const dd = hamming(tdesc, 8 * m, q, qo);
          const child = tree[node + 2 + 2 * j];
          if (dd < bestD) {
            if (bestChild >= 0) this.push(bestChild, bestD);
            bestD = dd;
            bestChild = child;
          } else {
            this.push(child, dd);
          }
        }
        node = bestChild;
      }
      const m = -tree[node];
      for (let j = 0; j < m; j++) {
        const f = tree[node + 1 + j];
        const dd = hamming(tdesc, 8 * f, q, qo);
        checks++;
        if (dd < this.dist1) {
          if (this.best1 >= 0 && rival(f, this.best1)) this.dist2 = this.dist1;
          this.dist1 = dd;
          this.best1 = f;
        } else if (dd < this.dist2 && rival(f, this.best1)) {
          this.dist2 = dd;
        }
      }
      if (checks >= this.maxChecks || this.heapSize === 0) break;
      node = this.pop();
    }
    return this.best1;
  }

  private push(node: number, key: number): void {
    if (this.heapSize >= this.heapNode.length) return;
    let i = this.heapSize++;
    const hn = this.heapNode, hk = this.heapKey;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (hk[p] <= key) break;
      hn[i] = hn[p]; hk[i] = hk[p];
      i = p;
    }
    hn[i] = node; hk[i] = key;
  }

  private pop(): number {
    const hn = this.heapNode, hk = this.heapKey;
    const top = hn[0];
    const n = --this.heapSize;
    const ln = hn[n], lk = hk[n];
    let i = 0;
    for (;;) {
      let c = 2 * i + 1;
      if (c >= n) break;
      if (c + 1 < n && hk[c + 1] < hk[c]) c++;
      if (hk[c] >= lk) break;
      hn[i] = hn[c]; hk[i] = hk[c];
      i = c;
    }
    hn[i] = ln; hk[i] = lk;
    return top;
  }
}
