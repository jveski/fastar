import { describe, expect, it } from 'vitest';
import { DisplayPacer } from '../src/shell/pacer';

const FRAME = 1000 / 30;
const VSYNC = 1000 / 60;

interface Run {
  /** Display time of each frame shown, by frame index. */
  shownAt: Map<number, number>;
  pacer: DisplayPacer<number>;
}

/**
 * Camera frames every 1/30 s, reaching the page `arrive` ms after capture;
 * frame i's tracking result (if any) `latency(i)` ms after capture; the
 * display polls the pacer every 1/60 s.
 */
function run(o: { frames: number; arrive?: number; latency?: (i: number) => number | null; active?: (t: number) => boolean; maxDelay?: number }): Run {
  const pacer = new DisplayPacer<number>(o.maxDelay !== undefined ? { maxDelay: o.maxDelay } : {});
  pacer.capacity = 12;
  const arrive = o.arrive ?? 20;
  const events: Array<{ t: number; kind: 'frame' | 'result' | 'vsync'; i: number }> = [];
  for (let i = 0; i < o.frames; i++) {
    events.push({ t: i * FRAME + arrive, kind: 'frame', i });
    const l = o.latency?.(i);
    if (l !== null && l !== undefined) events.push({ t: i * FRAME + l, kind: 'result', i });
  }
  const end = o.frames * FRAME + 400;
  for (let t = 0; t < end; t += VSYNC) events.push({ t, kind: 'vsync', i: -1 });
  // Same time: frames and results before the display poll.
  const rank = { frame: 0, result: 1, vsync: 2 };
  events.sort((a, b) => a.t - b.t || rank[a.kind] - rank[b.kind]);
  const shownAt = new Map<number, number>();
  for (const e of events) {
    pacer.active = o.active ? o.active(e.t) : true;
    if (e.kind === 'frame') pacer.push({ captureTime: e.i * FRAME, arrival: e.t, payload: e.i });
    else if (e.kind === 'result') pacer.result(e.i * FRAME, e.t);
    else {
      const f = pacer.next(e.t);
      if (f) shownAt.set(f.payload, e.t);
    }
  }
  return { shownAt, pacer };
}

/** Gaps between consecutive shown frames from `from` on. */
function gaps(r: Run, from: number): number[] {
  const idx = [...r.shownAt.keys()].filter((i) => i >= from).sort((a, b) => a - b);
  const out: number[] = [];
  for (let k = 1; k < idx.length; k++) out.push(r.shownAt.get(idx[k])! - r.shownAt.get(idx[k - 1])!);
  return out;
}

describe('display pacing (syncDisplay)', () => {
  it('with nothing in view, shows each frame as it arrives', () => {
    const r = run({ frames: 60, active: () => false, latency: () => 50 });
    expect(r.shownAt.size).toBe(60);
    for (const [i, t] of r.shownAt) expect(t - (i * FRAME + 20)).toBeLessThan(VSYNC + 1e-6);
    expect(r.pacer.extraDelay).toBeLessThan(1);
  });

  it('settles a steady delay that most results arrive within, and shows every frame at the camera rate', () => {
    const r = run({ frames: 120, latency: (i) => 45 + (i % 5) * 3 });
    expect(r.shownAt.size).toBe(120);
    expect(r.pacer.dropped).toBe(0);
    // After the delay has settled: two display frames per camera frame, every time.
    for (const g of gaps(r, 40)) expect(Math.abs(g - FRAME)).toBeLessThan(VSYNC + 1e-6);
    // Shown after its result: the overlay uses the pose measured on the frame.
    let onTime = 0, n = 0;
    for (const [i, t] of r.shownAt) {
      if (i < 40) continue;
      n++;
      if (t >= i * FRAME + 45 + (i % 5) * 3) onTime++;
    }
    expect(onTime / n).toBeGreaterThan(0.85);
    expect(r.pacer.extraDelay).toBeGreaterThan(30);
    expect(r.pacer.extraDelay).toBeLessThan(45);
  });

  it('on a device tracking only some frames, at uneven times, the picture still moves at the camera rate', () => {
    // Every 2nd or 3rd frame is tracked (whichever arrives while the tracker is free), taking 50-80 ms.
    const tracked = (i: number) => i % 5 === 0 || i % 5 === 2;
    const r = run({ frames: 150, latency: (i) => (tracked(i) ? 50 + ((i * 7) % 31) : null) });
    expect(r.shownAt.size).toBe(150);
    const g = gaps(r, 50);
    for (const x of g) expect(Math.abs(x - FRAME)).toBeLessThan(VSYNC + 1e-6);
  });

  it('never waits longer than maxDelay past arrival: late frames are shown on time instead', () => {
    const r = run({ frames: 120, latency: () => 400, maxDelay: 60 });
    expect(r.shownAt.size).toBe(120);
    for (const [i, t] of r.shownAt) expect(t - (i * FRAME + 20)).toBeLessThanOrEqual(60 + VSYNC + 1e-6);
    expect(r.pacer.extraDelay).toBeCloseTo(60, 0);
    expect(r.pacer.late / r.pacer.shown).toBeGreaterThan(0.9);
    for (const g of gaps(r, 40)) expect(Math.abs(g - FRAME)).toBeLessThan(VSYNC + 1e-6);
  });

  it('eases the delay in when a target is found, a few ms per frame, without skipping frames', () => {
    const foundAt = 40 * FRAME;
    const r = run({ frames: 120, latency: () => 90, active: (t) => t >= foundAt });
    expect(r.shownAt.size).toBe(120);
    expect(r.pacer.dropped).toBe(0);
    // Each frame is shown no more than one display frame later than the camera rate allows.
    for (const g of gaps(r, 0)) {
      expect(g).toBeGreaterThan(0);
      expect(g).toBeLessThanOrEqual(FRAME + VSYNC + 1e-6);
    }
    expect(r.pacer.extraDelay).toBeCloseTo(72, 0);
  });

  it('drops the oldest frames beyond its capacity', () => {
    const p = new DisplayPacer<number>();
    p.capacity = 3;
    p.active = true;
    p.result(-1000, 0);
    for (let i = 0; i < 5; i++) p.push({ captureTime: i * FRAME, arrival: i * FRAME + 20, payload: i });
    expect(p.length).toBe(3);
    expect(p.dropped).toBe(2);
  });
});
