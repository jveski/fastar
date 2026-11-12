import { describe, expect, it } from 'vitest';
import { deflateSync } from 'node:zlib';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { decodePng } from '../cli/png';
import { getPoster } from './fixtures';
import { decodeTargets } from '../src/target/format';
import { makeMotionClip } from '../src/sim/motion';
import { makeMusic } from '../src/sim/audio';
import { WavReader, WavWriter } from '../cli/wav';

function crc32(b: Uint8Array): number {
  let c = ~0;
  for (const x of b) {
    c ^= x;
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c >>> 0;
}

/** Encode RGB(A)/grey 8-bit PNG cycling through all five filter types. */
function encodePng(w: number, h: number, channels: number, px: Uint8Array): Uint8Array {
  const stride = w * channels;
  const raw = new Uint8Array((stride + 1) * h);
  for (let y = 0; y < h; y++) {
    const f = y % 5;
    raw[y * (stride + 1)] = f;
    for (let i = 0; i < stride; i++) {
      const x = px[y * stride + i];
      const a = i >= channels ? px[y * stride + i - channels] : 0;
      const b = y > 0 ? px[(y - 1) * stride + i] : 0;
      const c = i >= channels && y > 0 ? px[(y - 1) * stride + i - channels] : 0;
      const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
      const pred = [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][f];
      raw[y * (stride + 1) + 1 + i] = (x - pred) & 255;
    }
  }
  const chunk = (type: string, data: Uint8Array) => {
    const out = new Uint8Array(12 + data.length);
    const dv = new DataView(out.buffer);
    dv.setUint32(0, data.length);
    for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
    out.set(data, 8);
    dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
    return out;
  };
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, w); dv.setUint32(4, h);
  ihdr[8] = 8; ihdr[9] = channels === 1 ? 0 : channels === 3 ? 2 : 6;
  const parts = [Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', new Uint8Array(0))];
  const total = parts.reduce((s, p) => s + p.length, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

describe('PNG decoder', () => {
  it('decodes greyscale with every filter type exactly', () => {
    const img = getPoster();
    const dec = decodePng(encodePng(img.width, img.height, 1, img.data));
    expect(dec.width).toBe(img.width);
    expect(dec.data).toEqual(img.data);
  });

  it('decodes RGB to luma', () => {
    const w = 7, h = 5, px = new Uint8Array(w * h * 3);
    for (let i = 0; i < px.length; i++) px[i] = (i * 37) & 255;
    const dec = decodePng(encodePng(w, h, 3, px));
    for (let i = 0; i < w * h; i++) {
      expect(dec.data[i]).toBe((px[3 * i] * 77 + px[3 * i + 1] * 150 + px[3 * i + 2] * 29 + 128) >> 8);
    }
  });
});

describe('fastar-compile CLI', () => {
  it('compiles a PNG into a loadable targets file', () => {
    const dir = mkdtempSync(join(tmpdir(), 'fastar-'));
    const img = getPoster();
    writeFileSync(join(dir, 'poster.png'), encodePng(img.width, img.height, 1, img.data));
    const out = execFileSync('npx', ['tsx', 'cli/compile.ts', '-o', join(dir, 't.bin'), '--report', join(dir, 'r.json'), join(dir, 'poster.png:0.2')], {
      encoding: 'utf8',
    });
    expect(out).toMatch(/poster: 640x480px, 0.2 m wide/);
    const buf = readFileSync(join(dir, 't.bin'));
    const [t] = decodeTargets(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    expect(t.meta.name).toBe('poster');
    expect(t.meta.physicalWidth).toBe(0.2);
    expect(JSON.parse(readFileSync(join(dir, 'r.json'), 'utf8'))[0].features).toBeGreaterThan(1000);
  });
});

describe('fastar-compile CLI, video', () => {
  const clip = makeMotionClip({ width: 320, height: 180, duration: 2 });
  const load = (path: string) => {
    const buf = readFileSync(path);
    return decodeTargets(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  };

  it('compiles a directory of frames, alongside an image', () => {
    const dir = mkdtempSync(join(tmpdir(), 'fastar-'));
    mkdirSync(join(dir, 'clip'));
    for (let i = 0; i < clip.frameCount; i++) {
      const f = clip.frame(i);
      writeFileSync(join(dir, 'clip', `f${i + 1}.png`), encodePng(f.width, f.height, 1, f.data));
    }
    const img = getPoster();
    writeFileSync(join(dir, 'poster.png'), encodePng(img.width, img.height, 1, img.data));
    const out = execFileSync('npx', ['tsx', 'cli/compile.ts', '--fps', '30', '-o', join(dir, 't.bin'), join(dir, 'poster.png:0.2'), join(dir, 'clip:0.4')], {
      encoding: 'utf8',
    });
    expect(out).toMatch(/clip: 320x180px video, 2.0 s at 30 fps, 0.4 m wide/);
    const [a, b] = load(join(dir, 't.bin'));
    expect(a.video).toBeUndefined();
    expect(b.meta.video!.frameCount).toBe(60);
    expect(b.meta.video!.changes).toEqual(clip.cuts);
  });

  const hasFfmpeg = spawnSync('ffmpeg', ['-version']).status === 0;
  it.skipIf(!hasFfmpeg)('compiles a video file through ffmpeg', () => {
    const dir = mkdtempSync(join(tmpdir(), 'fastar-'));
    const raw = new Uint8Array(clip.width * clip.height * clip.frameCount);
    for (let i = 0; i < clip.frameCount; i++) raw.set(clip.frame(i).data, i * clip.width * clip.height);
    writeFileSync(join(dir, 'clip.gray'), raw);
    execFileSync('ffmpeg', [
      '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', `${clip.width}x${clip.height}`, '-r', '30', '-i', join(dir, 'clip.gray'),
      '-c:v', 'mpeg4', '-q:v', '2', '-pix_fmt', 'yuv420p', join(dir, 'clip.mp4'),
    ]);
    const out = execFileSync('npx', ['tsx', 'cli/compile.ts', '-o', join(dir, 't.bin'), join(dir, 'clip.mp4:0.4')], { encoding: 'utf8' });
    expect(out).toMatch(/clip: 320x180px video, 2.0 s at 30 fps/);
    const [t] = load(join(dir, 't.bin'));
    expect(t.meta.video!.frameCount).toBe(60);
    expect(t.meta.video!.changes).toEqual(clip.cuts);
    expect(t.meta.video!.report.minPoints).toBeGreaterThan(20);
  });
});

describe('fastar-timecode CLI', () => {
  const run = (args: string[], input?: Buffer) => spawnSync('npx', ['tsx', 'cli/timecode.ts', ...args], { encoding: input ? 'buffer' : 'utf8', input, maxBuffer: 1 << 26 });
  const report = (args: string[]) => {
    const r = run([...args, '--json']);
    return { status: r.status, json: JSON.parse(r.stdout as string) };
  };

  it('generates timecode that decodes to its set time', () => {
    const dir = mkdtempSync(join(tmpdir(), 'fastar-'));
    const wav = join(dir, 'tc.wav');
    const out = run(['generate', '-o', wav, '-d', '20', '--start', '1:00:00', '--key', 'club']);
    expect(out.status).toBe(0);
    expect(out.stdout).toMatch(/0:00:20.000 of timecode from set time 1:00:00.000, 48000 Hz, key "club"/);
    const r = new WavReader(wav);
    expect(r.info).toEqual({ sampleRate: 48000, channels: 1, format: 'pcm16', frames: 20 * 48000 });
    r.close();
    const ok = report(['decode', wav, '--key', 'club']);
    expect(ok.status).toBe(0);
    expect(ok.json.found).toBe(true);
    expect(ok.json.lockedAtEnd).toBe(true);
    expect(Math.abs(ok.json.startTime - 3600)).toBeLessThan(0.001);
    expect(ok.json.lockedAfter).toBeLessThan(8);
    // Another event's listeners hear nothing.
    const other = report(['decode', wav]);
    expect(other.status).toBe(2);
    expect(other.json.found).toBe(false);
  });

  it('mixes timecode into music, following its level', () => {
    const dir = mkdtempSync(join(tmpdir(), 'fastar-'));
    const music = makeMusic(48000, 20, 5);
    const w = new WavWriter(join(dir, 'set.wav'), 48000, 2, 'pcm24');
    const l = music.map((v) => 0.3 * v), r = music.map((v) => -0.3 * v);
    w.write([l, r]);
    w.close();
    const out = run(['mix', '-o', join(dir, 'out.wav'), join(dir, 'set.wav'), '--start', '90']);
    expect(out.status).toBe(0);
    expect(out.stdout).toMatch(/timecode on 100% of the time, at -[0-9.]+ dBFS on average/);
    const mixed = new WavReader(join(dir, 'out.wav'));
    expect(mixed.info).toEqual({ sampleRate: 48000, channels: 2, format: 'pcm24', frames: 20 * 48000 });
    const ch = mixed.read(48000 * 20)!;
    mixed.close();
    // The music is untouched below the timecode's band: what was added is small.
    let peak = 0;
    for (let i = 0; i < l.length; i++) peak = Math.max(peak, Math.abs(ch[0][i] - l[i]));
    expect(20 * Math.log10(peak)).toBeLessThan(-18 + 0.1);
    const d = report(['decode', join(dir, 'out.wav')]);
    expect(Math.abs(d.json.startTime - 90)).toBeLessThan(0.001);
  });

  it('mixes a live raw stream', () => {
    const n = 48000 * 2, x = new Float32Array(2 * n);
    const music = makeMusic(48000, 2, 6);
    for (let i = 0; i < n; i++) x[2 * i] = x[2 * i + 1] = 0.3 * music[i];
    const r = run(['mix', '--raw', '--rate', '48000', '--channels', '2'], Buffer.from(x.buffer));
    expect(r.status).toBe(0);
    const y = new Float32Array(new Uint8Array(r.stdout as Buffer).slice().buffer);
    expect(y.length).toBe(x.length);
    let diff = 0;
    for (let i = 0; i < y.length; i++) diff = Math.max(diff, Math.abs(y[i] - x[i]));
    expect(diff).toBeGreaterThan(1e-4);
    expect(diff).toBeLessThan(0.13);
  });
});
