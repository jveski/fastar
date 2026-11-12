// Minimal WAV reading and writing (PCM 16/24/32-bit and 32-bit float),
// streamed in blocks so hours of audio never sit in memory.
import { closeSync, fstatSync, openSync, readSync, writeSync } from 'node:fs';

export type SampleFormat = 'pcm16' | 'pcm24' | 'pcm32' | 'float32';

export interface WavInfo {
  sampleRate: number;
  channels: number;
  format: SampleFormat;
  /** Sample frames in the file. */
  frames: number;
}

const BYTES: Record<SampleFormat, number> = { pcm16: 2, pcm24: 3, pcm32: 4, float32: 4 };

export class WavReader {
  readonly info: WavInfo;
  private readonly fd: number;
  private readonly dataStart: number;
  private pos = 0;

  constructor(path: string) {
    this.fd = openSync(path, 'r');
    const size = fstatSync(this.fd).size;
    const head = Buffer.alloc(12);
    readSync(this.fd, head, 0, 12, 0);
    if (head.toString('latin1', 0, 4) !== 'RIFF' || head.toString('latin1', 8, 12) !== 'WAVE') throw new Error(`${path} is not a WAV file`);
    let off = 12, fmt: { tag: number; channels: number; rate: number; bits: number } | null = null;
    let dataStart = -1, dataLen = 0;
    const ck = Buffer.alloc(8);
    while (off + 8 <= size) {
      readSync(this.fd, ck, 0, 8, off);
      const id = ck.toString('latin1', 0, 4), len = ck.readUInt32LE(4);
      if (id === 'fmt ') {
        const b = Buffer.alloc(Math.min(len, 40));
        readSync(this.fd, b, 0, b.length, off + 8);
        let tag = b.readUInt16LE(0);
        if (tag === 0xfffe && b.length >= 26) tag = b.readUInt16LE(24); // WAVE_FORMAT_EXTENSIBLE: the sub-format
        fmt = { tag, channels: b.readUInt16LE(2), rate: b.readUInt32LE(4), bits: b.readUInt16LE(14) };
      } else if (id === 'data') {
        dataStart = off + 8;
        // Files written while streaming may say 0 or 0xffffffff: read to the end.
        dataLen = len === 0 || len === 0xffffffff || dataStart + len > size ? size - dataStart : len;
        break;
      }
      off += 8 + len + (len & 1);
    }
    if (!fmt || dataStart < 0) throw new Error(`${path}: no fmt or data chunk`);
    const format: SampleFormat | null =
      fmt.tag === 3 && fmt.bits === 32 ? 'float32'
        : fmt.tag === 1 && fmt.bits === 16 ? 'pcm16'
          : fmt.tag === 1 && fmt.bits === 24 ? 'pcm24'
            : fmt.tag === 1 && fmt.bits === 32 ? 'pcm32' : null;
    if (!format) throw new Error(`${path}: unsupported WAV encoding (format ${fmt.tag}, ${fmt.bits} bits)`);
    this.dataStart = dataStart;
    this.info = { sampleRate: fmt.rate, channels: fmt.channels, format, frames: Math.floor(dataLen / (BYTES[format] * fmt.channels)) };
  }

  /** The next frames (at most `frames`), one array per channel; null at the end. */
  read(frames: number): Float32Array[] | null {
    const { channels, format } = this.info, bps = BYTES[format];
    const n = Math.min(frames, this.info.frames - this.pos);
    if (n <= 0) return null;
    const buf = Buffer.alloc(n * channels * bps);
    readSync(this.fd, buf, 0, buf.length, this.dataStart + this.pos * channels * bps);
    this.pos += n;
    return deinterleave(buf, n, channels, format);
  }

  close(): void {
    closeSync(this.fd);
  }
}

export class WavWriter {
  private readonly fd: number;
  private frames = 0;
  /** Samples clipped to full scale (integer formats). */
  clipped = 0;

  constructor(path: string, readonly sampleRate: number, readonly channels: number, readonly format: SampleFormat = 'pcm16') {
    this.fd = openSync(path, 'w');
    writeSync(this.fd, this.header());
  }

  write(data: Float32Array[]): void {
    const n = data[0].length;
    const buf = interleave(data, n, this.format);
    this.clipped += buf.clipped;
    writeSync(this.fd, buf.bytes);
    this.frames += n;
  }

  close(): void {
    writeSync(this.fd, this.header(), 0, 44, 0);
    closeSync(this.fd);
  }

  private header(): Buffer {
    const bps = BYTES[this.format], data = this.frames * this.channels * bps;
    if (data > 0xffffffff - 36) throw new Error('WAV files are limited to 4 GB; use a shorter duration or fewer bits');
    const h = Buffer.alloc(44);
    h.write('RIFF', 0, 'latin1');
    h.writeUInt32LE(36 + data, 4);
    h.write('WAVEfmt ', 8, 'latin1');
    h.writeUInt32LE(16, 16);
    h.writeUInt16LE(this.format === 'float32' ? 3 : 1, 20);
    h.writeUInt16LE(this.channels, 22);
    h.writeUInt32LE(this.sampleRate, 24);
    h.writeUInt32LE(this.sampleRate * this.channels * bps, 28);
    h.writeUInt16LE(this.channels * bps, 32);
    h.writeUInt16LE(bps * 8, 34);
    h.write('data', 36, 'latin1');
    h.writeUInt32LE(data, 40);
    return h;
  }
}

/** Interleaved samples (any format) to one array per channel. */
export function deinterleave(buf: Uint8Array, n: number, channels: number, format: SampleFormat): Float32Array[] {
  const out = Array.from({ length: channels }, () => new Float32Array(n));
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength), bps = BYTES[format];
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < channels; c++) {
      const o = (i * channels + c) * bps;
      out[c][i] =
        format === 'pcm16' ? dv.getInt16(o, true) / 32768
          : format === 'pcm24' ? (((dv.getUint8(o + 2) << 24) | (dv.getUint8(o + 1) << 16) | (dv.getUint8(o) << 8)) >> 8) / 8388608
            : format === 'pcm32' ? dv.getInt32(o, true) / 2147483648
              : dv.getFloat32(o, true);
    }
  }
  return out;
}

/** One array per channel to interleaved bytes. */
export function interleave(data: Float32Array[], n: number, format: SampleFormat): { bytes: Uint8Array; clipped: number } {
  const channels = data.length, bps = BYTES[format];
  const bytes = new Uint8Array(n * channels * bps), dv = new DataView(bytes.buffer);
  let clipped = 0;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < channels; c++) {
      const o = (i * channels + c) * bps;
      let v = data[c][i];
      if (format === 'float32') { dv.setFloat32(o, v, true); continue; }
      if (v > 1 || v < -1) { clipped++; v = v > 0 ? 1 : -1; }
      if (format === 'pcm16') dv.setInt16(o, Math.max(-32768, Math.min(32767, Math.round(v * 32768))), true);
      else if (format === 'pcm32') dv.setInt32(o, Math.max(-2147483648, Math.min(2147483647, Math.round(v * 2147483648))), true);
      else {
        const s = Math.max(-8388608, Math.min(8388607, Math.round(v * 8388608)));
        dv.setUint8(o, s & 255); dv.setUint8(o + 1, (s >> 8) & 255); dv.setUint8(o + 2, (s >> 16) & 255);
      }
    }
  }
  return { bytes, clipped };
}
