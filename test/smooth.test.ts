import { describe, expect, it } from 'vitest';
import { PoseSmoother } from '../src/fusion/smooth';
import { mat3, so3Exp, vec3, rotationAngle } from '../src/math/rot';

/** Target-in-camera pose for a camera yawing at w rad/s, target 30 cm ahead. */
function truth(time: number, w = 0.5) {
  const R = so3Exp(0, -w * time, 0, mat3());
  const t = vec3();
  // t = R_cam^T (p_t - p_c) with the target at (0, 0, 0.3) in the first camera frame.
  const Rc = so3Exp(0, w * time, 0, mat3());
  t[0] = Rc[0] * 0 + Rc[3] * 0 + Rc[6] * 0.3;
  t[1] = Rc[1] * 0 + Rc[4] * 0 + Rc[7] * 0.3;
  t[2] = Rc[2] * 0 + Rc[5] * 0 + Rc[8] * 0.3;
  return { R, t };
}

describe('pose smoother', () => {
  it('follows camera motion without lag and eases a sudden correction', () => {
    const sm = new PoseSmoother();
    const R = mat3(), t = vec3();
    const jumpErr: number[] = [];
    for (let k = 0; k < 30; k++) {
      const time = k / 30;
      const g = truth(time);
      // The filter's pose jumps 5 mm sideways at frame 10 (e.g. a re-anchor) and stays.
      const off = k >= 10 ? 0.005 : 0;
      R.set(g.R); t.set(g.t); t[0] += off;
      const prev = sm.previousTime;
      let Rp: Float64Array | null = null, tp: Float64Array | null = null;
      if (Number.isFinite(prev)) {
        const gp = truth(prev);
        Rp = gp.R; tp = gp.t; tp[0] += k >= 10 ? 0.005 : 0; // same filter state at the previous frame
      }
      sm.update(time, R, t, Rp, tp);
      if (k < 10) {
        expect(Math.abs(t[0] - g.t[0])).toBeLessThan(1e-9);
        expect(rotationAngle(R, g.R)).toBeLessThan(1e-9);
      } else jumpErr.push(t[0] - (g.t[0] + off));
    }
    // Not applied at once, converges within a few frames, monotonically.
    expect(Math.abs(jumpErr[0])).toBeGreaterThan(0.002);
    expect(Math.abs(jumpErr[10])).toBeLessThan(2e-4);
    for (let i = 1; i < jumpErr.length; i++) expect(Math.abs(jumpErr[i])).toBeLessThanOrEqual(Math.abs(jumpErr[i - 1]) + 1e-12);
  });
});
