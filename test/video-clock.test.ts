import { describe, expect, it } from 'vitest';
import { VideoClock } from '../src/fusion/video-clock';
import { Rng } from '../src/math/rng';

const D = 6, FPS = 30;

/** Circular distance between video times. */
const dtv = (a: number, b: number) => {
  const d = Math.abs(a - b) % D;
  return Math.min(d, D - d);
};

/**
 * Probe scores a tracker would produce: 20 probes, NCC 0.9 against the frame
 * on screen, 0.3 against others, mixed when a 10 ms exposure spans a frame change.
 */
function frameScore(f: number, tv: number, probes = 20, exposure = 0.01): number {
  const a = tv - exposure / 2, b = tv + exposure / 2;
  const N = D * FPS;
  let best = 0;
  for (const k of [-1, 0, 1]) {
    const fa = (((f % N) + N) % N) / FPS + k * D, fb = fa + 1 / FPS;
    best = Math.max(best, Math.max(0, Math.min(b, fb) - Math.max(a, fa)) / exposure);
  }
  return probes * (0.3 + 0.6 * best);
}

/** Per-frame log-likelihood from candidate scores, as the tracker's probes produce it. */
function toLlr(fr: ArrayLike<number>, sc: ArrayLike<number>, n: number, probes = 20): Float64Array {
  const llr = new Float64Array(D * FPS);
  for (let k = 0; k < n; k++) {
    const v = Math.max(-1.2, Math.min(1.2, 3 * (sc[k] / probes - 0.6)));
    llr[((fr[k] % (D * FPS)) + D * FPS) % (D * FPS)] += probes * v;
  }
  return llr;
}

function probe(c: VideoClock, t: number, truth: (t: number) => number, rng: Rng, probes = 20): void {
  const fr = new Int32Array(64), sc = new Float64Array(64);
  const n = c.candidates(t, fr);
  for (let k = 0; k < n; k++) sc[k] = frameScore(fr[k], truth(t), probes) + rng.gauss() * 0.2 * Math.sqrt(probes);
  c.addFrameLikelihood(t, toLlr(fr, sc, n, probes));
}

describe('video clock', () => {
  it('starts uniform and narrows to an acquisition interval', () => {
    const c = new VideoClock(D, FPS);
    expect(c.window(0).locked).toBe(false);
    c.addRanges(10, [2, 3]);
    const w = c.window(10);
    expect(w.tv).toBeGreaterThan(2);
    expect(w.tv).toBeLessThan(3);
    expect(w.lo).toBeLessThan(2.05);
    expect(w.hi).toBeGreaterThan(2.95);
    expect(w.std).toBeGreaterThan(0.2);
    expect(w.mass).toBeGreaterThan(0.9);
    // The interval moves with playback.
    const w2 = c.window(10.5);
    expect(w2.tv).toBeCloseTo(w.tv + 0.5, 6);
  });

  it('locks to within a third of a frame from probe scores and follows playback', () => {
    const b0 = 1.234;
    const truth = (t: number) => ((t - b0) % D + D) % D;
    const c = new VideoClock(D, FPS);
    const rng = new Rng(3);
    c.addRanges(5, [truth(5) - 0.4, truth(5) + 0.3]);
    // Camera frames at 29.5 fps: their phase against the video frames drifts.
    let t = 5, lockedAt = NaN;
    for (let k = 0; k < 60; k++, t += 1 / 29.5) {
      probe(c, t, truth, rng);
      if (c.window(t).locked && Number.isNaN(lockedAt)) lockedAt = k;
    }
    expect(lockedAt).toBeLessThan(10);
    const w = c.window(t);
    expect(w.locked).toBe(true);
    expect(dtv(w.tv, truth(t))).toBeLessThan(1 / (3 * FPS));
    expect(dtv(c.offset(), b0)).toBeLessThan(1 / (3 * FPS));
  });

  it('picks the right one of two hypotheses once probes disagree with one', () => {
    const b0 = 0.5;
    const truth = (t: number) => ((t - b0) % D + D) % D;
    const c = new VideoClock(D, FPS);
    c.addRanges(3, [truth(3) - 0.2, truth(3) + 0.2, truth(3) + 2.8, truth(3) + 3.2]);
    const rng = new Rng(9);
    let t = 3;
    for (let k = 0; k < 30; k++, t += 1 / 29.5) {
      // Score every frame so both hypotheses get tested.
      const fr = new Int32Array(D * FPS), sc = new Float64Array(D * FPS);
      for (let f = 0; f < D * FPS; f++) { fr[f] = f; sc[f] = frameScore(f, truth(t)) + rng.gauss(); }
      c.addFrameLikelihood(t, toLlr(fr, sc, D * FPS));
    }
    expect(dtv(c.window(t).tv, truth(t))).toBeLessThan(2 / FPS);
  });

  it('re-locks after a pause at the loop seam and learns the pause', () => {
    const gap = 0.08;
    // Playback with a pause at each seam: period D + gap.
    const truth = (t: number) => {
      const ph = ((t - 0.3) % (D + gap) + (D + gap)) % (D + gap);
      return Math.min(ph, D - 1e-6);
    };
    const c = new VideoClock(D, FPS);
    const rng = new Rng(5);
    c.addRanges(1, [truth(1) - 0.3, truth(1) + 0.3]);
    let t = 1;
    const relock: number[] = [];
    let seamAt = NaN;
    for (let k = 0; k < Math.round(14 * 30); k++, t += 1 / 29.5) {
      const before = truth(t - 1 / 29.5);
      if (truth(t) < before - 1) seamAt = t;
      if (truth(t) >= D - 1e-5) continue; // paused on the last frame: nothing new to see
      probe(c, t, truth, rng);
      const w = c.window(t);
      if (!Number.isNaN(seamAt) && w.locked && dtv(w.tv, truth(t)) < 1 / FPS) {
        relock.push(t - seamAt);
        seamAt = NaN;
      }
    }
    expect(relock.length).toBe(2);
    expect(relock[0]).toBeLessThan(0.5);
    expect(relock[1]).toBeLessThan(0.2);
    expect(c.loopGap).toBeGreaterThan(gap - 0.02);
    expect(c.loopGap).toBeLessThan(gap + 0.03);
    expect(dtv(c.window(t).tv, truth(t))).toBeLessThan(1 / (2 * FPS));
  });

  it('starts over when the camera clock jumps or stalls for long', () => {
    const c = new VideoClock(D, FPS);
    for (let k = 0; k < 3; k++) c.addRanges(10, [2, 2.2]);
    expect(c.window(10).hi - c.window(10).lo).toBeLessThan(0.3);
    c.advance(10.5);
    expect(c.window(10.5).hi - c.window(10.5).lo).toBeLessThan(0.3);
    c.advance(4000); // a new time base
    expect(c.window(4000).mass).toBeGreaterThan(0.99);
    expect(c.window(4000).hi - c.window(4000).lo).toBeGreaterThan(D * 0.9);
    c.addRanges(4000, [1, 1.5]);
    c.advance(3990); // backwards
    expect(c.window(3990).hi - c.window(3990).lo).toBeGreaterThan(D * 0.9);
  });
});
