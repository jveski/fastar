import { describe, expect, it } from 'vitest';
import { Governor, Ladder, LEVELS, deviceStartLevel } from '../src/shell/governor';

const HOLD = { hold: 5000, maxHold: 60000, relapse: 20000 };

/**
 * Drive a governor for `ms` from `t0`: display frames every `vsync` ms
 * (every `slowEvery`-th one late by a frame), `busy` of the time handling
 * frames, and a tracking load.
 */
function drive(g: Governor, t0: number, ms: number, o: { vsync?: number; busy?: number; slowEvery?: number; stride?: number; overload?: number }): number {
  const vsync = o.vsync ?? 1000 / 60;
  g.setTrackingLoad({ stride: o.stride ?? 1, overload: o.overload ?? 0, interval: 1000 / 30, acquireBusy: 0.1 });
  let t = t0, k = 0;
  while (t < t0 + ms) {
    const d = o.slowEvery && ++k % o.slowEvery === 0 ? 2 * vsync : vsync;
    t += d;
    g.animationFrame(t);
    g.addMain((o.busy ?? 0.1) * d);
    g.update(t);
  }
  return t;
}

describe('quality ladder', () => {
  it('steps down after two bad windows, and back up only after a calm hold', () => {
    const l = new Ladder(3, HOLD);
    expect(l.step(0, true, false)).toBe(false);
    expect(l.step(500, true, false)).toBe(true);
    expect(l.level).toBe(1);
    // Neither bad nor calm: no change, and the calm clock does not run.
    expect(l.step(1000, false, false)).toBe(false);
    expect(l.step(1500, false, true)).toBe(false);
    expect(l.step(6000, false, true)).toBe(false);
    expect(l.step(6500, false, true)).toBe(true);
    expect(l.level).toBe(0);
  });

  it('waits twice as long to raise quality after it turned out too soon, up to a cap', () => {
    const l = new Ladder(3, HOLD);
    l.step(0, true, false); l.step(500, true, false);
    l.step(1000, false, true); l.step(6000, false, true); // back to 0 at 6000
    expect(l.level).toBe(0);
    // Trouble again within the relapse window: the next hold is 10 s.
    l.step(7000, true, false); l.step(7500, true, false);
    expect(l.level).toBe(1);
    l.step(8000, false, true);
    expect(l.step(13000, false, true)).toBe(false);
    expect(l.step(18000, false, true)).toBe(true);
  });

  it('never goes past its range', () => {
    const l = new Ladder(1, HOLD);
    for (let t = 0; t < 5000; t += 500) l.step(t, true, false);
    expect(l.level).toBe(1);
  });
});

describe('performance governor', () => {
  it('leaves a device that keeps up at full quality', () => {
    const g = new Governor({ devicePixelRatio: 3 });
    drive(g, 0, 20000, {});
    const s = g.state;
    expect(s.level).toBe(0);
    expect(s.bottleneck).toBe('none');
    expect(s.pixelRatio).toBe(3);
    expect(s.effects).toBe('full');
    expect(g.acquireDuty).toBe(LEVELS[0].acquireDuty);
  });

  it('a page that cannot keep up: suggests cheaper rendering and takes acquisition down too', () => {
    const g = new Governor({ devicePixelRatio: 3 });
    let t = drive(g, 0, 3000, {});
    t = drive(g, t, 3600, { busy: 0.7, slowEvery: 2 });
    const s = g.state;
    expect(s.bottleneck).toBe('render');
    expect(s.level).toBe(3);
    expect(s.pixelRatio).toBe(1);
    expect(s.effects).toBe('minimal');
    expect(s.mainBusy).toBeGreaterThan(0.5);
    expect(g.acquireDuty).toBe(LEVELS[3].acquireDuty);
    // Once it copes: back up a step at a time, after the hold.
    t = drive(g, t, 4000, {});
    expect(g.state.level).toBe(3);
    t = drive(g, t, 2000, {});
    expect(g.state.level).toBe(2);
    drive(g, t, 11000, {});
    expect(g.state.level).toBe(0);
    expect(g.state.pixelRatio).toBe(3);
  });

  it('tracking that cannot keep up: throttles acquisition, but leaves the rendering hints alone', () => {
    const g = new Governor({ devicePixelRatio: 2 });
    let t = drive(g, 0, 3000, {});
    t = drive(g, t, 1200, { stride: 2 });
    const s = g.state;
    expect(s.bottleneck).toBe('tracking');
    expect(s.level).toBeGreaterThan(0);
    expect(s.pixelRatio).toBe(2);
    expect(s.effects).toBe('full');
    expect(g.acquireDuty).toBeLessThan(LEVELS[0].acquireDuty);
    expect(s.stride).toBe(2);
  });

  it('takes a display capped at 30 Hz (iOS Low Power Mode) as normal, not as late frames', () => {
    const g = new Governor();
    drive(g, 0, 10000, { vsync: 1000 / 30 });
    expect(g.state.jank).toBe(0);
    expect(g.state.level).toBe(0);
  });

  it('counts a late display frame once, not the short gap that follows it', () => {
    // One frame in 6 is 12 ms late, the next one 12 ms early (back on the vsync grid).
    const g = new Governor();
    let t = 0;
    for (let k = 0; t < 10000; k++) {
      t += k % 6 === 0 ? 28 : k % 6 === 1 ? 5 : 1000 / 60;
      g.animationFrame(t);
      g.addMain(1);
      g.update(t);
    }
    expect(g.state.jank).toBeGreaterThan(0.1);
    expect(g.state.jank).toBeLessThan(0.25);
    expect(g.state.level).toBe(0);
  });

  it('notices when display frames start coming late (sustained, it reads as a 30 Hz display after a while)', () => {
    const g = new Governor();
    let t = drive(g, 0, 3000, {});
    t = drive(g, t, 3000, { slowEvery: 1 });
    expect(g.state.jank).toBeGreaterThan(0.9);
    expect(g.state.bottleneck).toBe('render');
  });

  it('ignores warm-up and pauses (a hidden page)', () => {
    const g = new Governor();
    let t = drive(g, 0, 1500, { busy: 0.9, slowEvery: 2 });
    expect(g.state.level).toBe(0);
    t = drive(g, t, 3000, {});
    // A 10 s gap between display frames, then normal frames.
    t = drive(g, t + 10000, 1000, {});
    expect(g.state.level).toBe(0);
  });

  it('starts a step down on devices that say they have little memory', () => {
    expect(deviceStartLevel({ deviceMemory: 1 })).toBe(1);
    expect(deviceStartLevel({ deviceMemory: 8 })).toBe(0);
    expect(deviceStartLevel({})).toBe(0);
  });
});
