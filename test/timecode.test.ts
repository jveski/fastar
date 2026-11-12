import { describe, expect, it } from 'vitest';
import { PositionCode, isPrimitive, timecodeParams } from '../src/timecode/code';
import { LevelFollower, TimecodeEncoder } from '../src/timecode/encoder';
import { TimecodeDecoder, fitEstimate } from '../src/timecode/decoder';
import { TimecodeCore, type FromTimecode } from '../src/worker/timecode-core';
import { LOOPBACK, LoopbackDetector, loopbackSignal } from '../src/timecode/loopback';
import { makeMusic, receive, type ChannelOptions } from '../src/sim/audio';
import { Rng } from '../src/math/rng';

const FS = 48000;

/** Power at frequency f (Hz) in x, Hann-windowed, averaged over 4096-sample blocks. */
function power(x: Float32Array, f: number, fs = FS): number {
  const N = 4096, w = (2 * Math.PI * f) / fs, c = 2 * Math.cos(w);
  let sum = 0, blocks = 0;
  for (let o = 0; o + N <= x.length; o += N, blocks++) {
    let s1 = 0, s2 = 0;
    for (let i = 0; i < N; i++) {
      const s0 = x[o + i] * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1))) + c * s1 - s2;
      s2 = s1; s1 = s0;
    }
    sum += s1 * s1 + s2 * s2 - c * s1 * s2;
  }
  return sum / blocks;
}

/** Music with timecode (from set time `start`) as played by the PA. */
function playback(seconds: number, o: { start: number; level?: number; key?: string; music?: number; gainAt?: (t: number) => number }): Float32Array {
  const music = makeMusic(FS, seconds, 3), tc = new Float32Array(music.length);
  const enc = new TimecodeEncoder({ sampleRate: FS, start: o.start, level: o.level ?? -40, key: o.key });
  if (o.gainAt) {
    const g = new Float32Array(music.length), a = 10 ** ((o.level ?? -40) / 20);
    for (let i = 0; i < g.length; i++) g[i] = a * o.gainAt(i / FS);
    enc.render(tc, g);
  } else {
    enc.render(tc);
  }
  for (let i = 0; i < music.length; i++) music[i] = (o.music ?? 0.3) * music[i] + tc[i];
  return music;
}

/** Feed a recording to a decoder in worklet-sized blocks; calls back every `every` seconds. */
function decode(dec: TimecodeDecoder, x: Float32Array, every: number, cb: (t: number) => void, block = 2048): void {
  let next = every;
  for (let i = 0; i < x.length; i += block) {
    dec.push(x.subarray(i, Math.min(i + block, x.length)), i);
    while (dec.time >= next) { cb(next); next += every; }
  }
}

const ROOM: Omit<ChannelOptions, 'rate'> = {
  ppm: -45, distance: 25, sway: { amplitude: 0.3, frequency: 0.8 }, echoes: [[0.014, 0.5], [0.033, 0.3]], noise: 0.002, clip: 0.3,
};

describe('timecode position code', () => {
  it('tests primitivity', () => {
    expect(isPrimitive(0b10011, 4)).toBe(true); // x^4 + x + 1
    expect(isPrimitive(0b11111, 4)).toBe(false); // x^4 + x^3 + x^2 + x + 1: order 5
    expect(isPrimitive(0b10101, 4)).toBe(false); // (x^2 + x + 1)^2
  });

  it('is a maximal-length sequence whose windows each locate one position', () => {
    for (const key of ['fastar', 'my-set']) {
      const p = timecodeParams({ key, degree: 12 });
      const code = new PositionCode(p.degree, p.poly);
      const seen = new Uint8Array(p.length + 1);
      for (let i = 0; i < p.length; i++) {
        let w = 0;
        for (let t = 0; t < p.degree; t++) w |= code.bit(i + t) << t;
        expect(seen[w]).toBe(0);
        seen[w] = 1;
        if (i % 97 === 0) expect(code.indexOf(w)).toBe(i);
      }
      expect(code.indexOf(0)).toBe(-1);
    }
  });

  it('differs between keys, and checks its options', () => {
    expect(timecodeParams({ key: 'a' }).poly).not.toBe(timecodeParams({ key: 'b' }).poly);
    expect(timecodeParams().period).toBeCloseTo((2 ** 18 - 1) * 0.1, 6);
    expect(() => timecodeParams({ band: [20000, 19000] })).toThrow(/band/);
    expect(() => timecodeParams({ degree: 30 })).toThrow(/degree/);
  });
});

describe('timecode generator', () => {
  it('stays in its band at a constant level', () => {
    const x = new Float32Array(FS * 2);
    new TimecodeEncoder({ sampleRate: FS, level: -12, fadeIn: 0 }).render(x);
    let peak = 0;
    for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
    expect(20 * Math.log10(peak)).toBeCloseTo(-12, 1);
    // Constant envelope: RMS over each 10 ms is that of a sine.
    for (let o = 0; o + 480 <= x.length; o += 480) {
      let s = 0;
      for (let i = 0; i < 480; i++) s += x[o + i] ** 2;
      expect(Math.sqrt(s / 480) / peak).toBeCloseTo(Math.SQRT1_2, 1);
    }
    let inBand = 0;
    for (let f = 19000; f <= 20000; f += 50) inBand = Math.max(inBand, power(x, f));
    let below = 0;
    for (let f = 500; f <= 17500; f += 250) below = Math.max(below, power(x, f));
    expect(10 * Math.log10(below / inBand)).toBeLessThan(-70);
  });

  it('continues seamlessly across render calls and from any start', () => {
    const a = new Float32Array(9000), b = new Float32Array(9000);
    const e1 = new TimecodeEncoder({ sampleRate: FS, start: 12.34 });
    e1.render(a);
    const e2 = new TimecodeEncoder({ sampleRate: FS, start: 12.34 });
    e2.render(b.subarray(0, 4000));
    e2.render(b.subarray(4000));
    expect(b).toEqual(a);
    expect(e1.time).toBeCloseTo(12.34 + 9000 / FS, 9);
  });

  it('follows the level of the music, and fades out in quiet passages', () => {
    const fol = new LevelFollower(FS, { relative: -24, max: -18, gate: -50 });
    const music = makeMusic(FS, 6, 4), g = new Float32Array(music.length);
    for (let i = 0; i < music.length; i++) music[i] *= i < 3 * FS ? 0.3 : 0.0005;
    fol.process(music, g);
    const db = (i: number) => 20 * Math.log10(g[i]);
    // Loud: a fixed amount under the music's highs, below the cap.
    const loud = db(2.5 * FS);
    expect(loud).toBeLessThanOrEqual(-18 + 1e-6);
    expect(loud).toBeGreaterThan(-60);
    // Quiet: faded right out within a second.
    expect(g[4 * FS]).toBeLessThan(1e-5);
    // And no sudden steps.
    let step = 0;
    for (let i = 1; i < g.length; i++) step = Math.max(step, Math.abs(g[i] - g[i - 1]));
    expect(step).toBeLessThan(1e-3 * Math.max(...g.subarray(0, FS * 3)));
  });
});

describe('timecode listener', () => {
  it('locks onto timecode under music across a room, to a millisecond or two', () => {
    const tx = playback(31, { start: 3723.4 });
    for (const rate of [48000, 44100]) {
      const rx = receive(tx, FS, 30, { ...ROOM, rate });
      const dec = new TimecodeDecoder({ sampleRate: rate });
      let lockedAt = NaN;
      const errs: number[] = [];
      decode(dec, rx.samples, 0.5, (t) => {
        const e = dec.estimate(t);
        if (e.locked && Number.isNaN(lockedAt)) lockedAt = t;
        if (e.locked) errs.push(Math.abs(e.time - (3723.4 + rx.emittedAt(t))));
      });
      expect(lockedAt).toBeLessThan(8);
      expect(Math.max(...errs)).toBeLessThan(0.004);
      expect(dec.stats.errors).toBe(0);
      // Clock rate against the phone's (45 ppm slow).
      expect((dec.fit.rate - 1) * 1e6).toBeGreaterThan(30);
      expect((dec.fit.rate - 1) * 1e6).toBeLessThan(60);
    }
  });

  it('holds the time through a breakdown with no timecode, and through dropped audio', () => {
    // The timecode fades out from 20 s to 60 s, as the level follower does in a breakdown.
    const tx = playback(81, { start: 100, gainAt: (t) => (t > 20 && t < 60 ? 0 : 1) });
    const rx = receive(tx, FS, 80, { ...ROOM, rate: FS });
    // The worklet's audio also skips half a second at 70 s.
    const cut = 70 * FS;
    rx.samples.fill(0, cut, cut + FS / 2);
    const dec = new TimecodeDecoder({ sampleRate: FS });
    let worst = 0, unlocked = 0;
    let next = 10;
    for (let i = 0; i < rx.samples.length; i += 2048) {
      if (i >= cut && i < cut + FS / 2) continue; // dropped blocks: the next push says where it is
      dec.push(rx.samples.subarray(i, Math.min(i + 2048, rx.samples.length)), i);
      while (dec.time >= next) {
        const e = dec.estimate(next);
        if (!e.locked) unlocked++;
        else worst = Math.max(worst, Math.abs(e.time - (100 + rx.emittedAt(next))));
        next += 0.5;
      }
    }
    expect(unlocked).toBe(0);
    expect(worst).toBeLessThan(0.005);
  });

  it('relocks when the timecode restarts at another time', () => {
    const a = playback(31, { start: 500 }), b = playback(31, { start: 9000 });
    const tx = new Float32Array(a.length * 2);
    tx.set(a.subarray(0, 15 * FS));
    tx.set(b.subarray(0, 16 * FS), 15 * FS);
    const rx = receive(tx, FS, 30, { ...ROOM, rate: FS });
    const truth = (t: number) => { const te = rx.emittedAt(t); return te < 15 ? 500 + te : 9000 + te - 15; };
    const dec = new TimecodeDecoder({ sampleRate: FS });
    let relocked = NaN;
    decode(dec, rx.samples, 0.25, (t) => {
      const e = dec.estimate(t);
      if (t > 15 && e.locked && Math.abs(e.time - truth(t)) < 0.005 && Number.isNaN(relocked)) relocked = t;
    });
    expect(relocked - 15).toBeLessThan(9);
    expect(Math.abs(dec.estimate(30).time - truth(30))).toBeLessThan(0.004);
  });

  it('never locks onto music, noise, or another event\'s timecode', () => {
    const rng = new Rng(2), music = makeMusic(FS, 120, 9);
    for (let i = 0; i < music.length; i++) music[i] = Math.max(-0.9, Math.min(0.9, 0.9 * music[i] + 0.02 * rng.gauss()));
    const other = playback(30, { start: 0, key: 'another-event' });
    for (const x of [music, other]) {
      const dec = new TimecodeDecoder({ sampleRate: FS });
      decode(dec, x, 1, (t) => expect(dec.estimate(t).locked).toBe(false));
      expect(dec.stats.codeLocked).toBe(false);
    }
  });

  it('estimates from a fit at any audio time', () => {
    const f = { codeLocked: true, points: 50, at: 10, time: 110, rate: 1.0001, varTime: 1e-6, varRate: 1e-10, last: 10 };
    const e = fitEstimate(f, 20);
    expect(e.time).toBeCloseTo(120.001, 9);
    expect(e.std).toBeCloseTo(Math.sqrt(1e-6 + 100 * 1e-10), 9);
    expect(e.locked).toBe(true);
    expect(fitEstimate(f, 2000).locked).toBe(false); // too long without hearing it
    expect(fitEstimate({ ...f, codeLocked: false }, 20).locked).toBe(false);
  });
});

describe('timecode worker', () => {
  it('decodes audio from its port and reports the fit', async () => {
    const posted: FromTimecode[] = [];
    const core = new TimecodeCore((m) => posted.push(m));
    const ch = new MessageChannel();
    core.init({ type: 'init-timecode', options: {}, sampleRate: FS, port: ch.port2 });
    expect(posted[0].type).toBe('timecode-ready');
    const tx = playback(12, { start: 42 });
    // As the worklet sends it: frames counted from when the audio context started.
    for (let i = 0; i < tx.length; i += 2048) core.audio({ frame: 96000 + i, samples: tx.slice(i, i + 2048) });
    const last = posted.filter((m) => m.type === 'timecode').pop();
    expect(last?.type).toBe('timecode');
    if (last?.type !== 'timecode') return;
    expect(last.stats.codeLocked).toBe(true);
    const t = last.audioTime;
    expect(fitEstimate(last.fit, t).time).toBeCloseTo(42 + t - 2, 3);
    ch.port1.close();
    ch.port2.close();
  });
});

describe('timecode loopback calibration', () => {
  /**
   * What the microphone hears when the burst rendered at audio time `start`
   * comes back `delay` seconds later from the phone's own speaker, over loud
   * music (and its echo off a hand).
   */
  function heard(start: number, delay: number | null, seconds: number, music = 0.3): Float32Array {
    const tx = new Float32Array(seconds * FS), burst = loopbackSignal(FS);
    if (delay !== null) tx.set(burst, Math.round(start * FS));
    // A speaker a few cm away: the distance sets the delay (343 m/s), gain 10/distance undone.
    const d = (delay ?? 0) * 343;
    const rx = receive(tx, FS, seconds, { rate: FS, distance: d, echoes: [[0.0008, 0.4]], noise: 0.003, seed: 4 }).samples;
    const m = makeMusic(FS, seconds, 12);
    for (let i = 0; i < rx.length; i++) rx[i] = rx[i] * 0.2 * (d / 10) + music * m[i];
    return rx;
  }

  function detect(x: Float32Array, start: number, from = 0): number | null | undefined {
    const det = new LoopbackDetector(FS, start);
    let r: number | null | undefined;
    for (let i = 0; i < x.length && r === undefined; i += 2048) r = det.push(x.subarray(i, Math.min(i + 2048, x.length)), from + i);
    return r;
  }

  it('plays six chirps in the band, at no two equal spacings', () => {
    const s = loopbackSignal(FS);
    expect(s.length).toBeGreaterThan(0.95 * FS);
    const diffs = new Set<number>();
    for (const a of LOOPBACK.offsets) for (const b of LOOPBACK.offsets) if (a < b) diffs.add(Math.round((b - a) * 1000));
    expect(diffs.size).toBe((LOOPBACK.offsets.length * (LOOPBACK.offsets.length - 1)) / 2);
    expect(power(s, 5000)).toBeGreaterThan(1e4 * power(s, 15000));
  });

  it('measures the microphone delay under loud music, to a fraction of a millisecond', () => {
    for (const delay of [0.0213, 0.3124, 1.4567]) {
      const r = detect(heard(1.5, delay, 7), 1.5);
      expect(r).not.toBeNull();
      expect(Math.abs(r! - delay)).toBeLessThan(0.0003);
    }
  });

  it('works on any audio clock origin, and reports nothing when the burst is not heard', () => {
    // Frames counted from 10 s: the burst at audio time 11.5 s.
    const x = heard(1.5, 0.25, 7);
    expect(Math.abs(detect(x, 11.5, 10 * FS)! - 0.25)).toBeLessThan(0.0003);
    expect(detect(heard(1.5, null, 7, 0.9), 1.5)).toBeNull();
  });

  it('runs in the worker on the audio the decoder hears', () => {
    const posted: FromTimecode[] = [];
    const core = new TimecodeCore((m) => posted.push(m));
    const ch = new MessageChannel();
    core.init({ type: 'init-timecode', options: {}, sampleRate: FS, port: ch.port2 });
    core.handle({ type: 'calibrate', start: 2.5 });
    const x = heard(0.5, 0.18, 6);
    for (let i = 0; i < x.length; i += 2048) core.audio({ frame: 2 * FS + i, samples: x.slice(i, i + 2048) });
    const r = posted.find((m) => m.type === 'timecode-latency');
    expect(r?.type === 'timecode-latency' && Math.abs(r.latency! - 0.18)).toBeLessThan(0.0003);
    ch.port1.close();
    ch.port2.close();
  });
});
