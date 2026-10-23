import type { BenchCase } from '../harness';
import { invertSPD } from '../../src/math/linalg';
import { so3Exp, so3Log, mat3, vec3 } from '../../src/math/rot';

export function mathCases(): BenchCase[] {
  return [
    {
      name: 'math: so3 exp/log x1000',
      budgetMs: 0.2,
      setup: () => {
        const R = mat3();
        const w = vec3();
        return () => {
          for (let i = 0; i < 1000; i++) {
            so3Exp(0.1 + i * 1e-4, -0.2, 0.3, R);
            so3Log(R, w);
          }
        };
      },
    },
    {
      name: 'math: 19x19 SPD inverse',
      budgetMs: 0.05,
      setup: () => {
        const n = 19;
        const base = new Float64Array(n * n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) base[i * n + j] = i === j ? n + 1 : 1 / (1 + i + j);
        const A = new Float64Array(n * n);
        const out = new Float64Array(n * n);
        const col = new Float64Array(n);
        return () => {
          A.set(base);
          invertSPD(A, n, out, col);
        };
      },
    },
  ];
}
