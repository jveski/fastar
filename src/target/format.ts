// Versioned binary target format, loaded zero-copy as typed-array views.
//
//   header:     magic 'FSTR' u32, version u32, targetCount u32, reserved u32
//   directory:  (offset u32, length u32) * targetCount
//   target:     sections of (tag u32, byteLength u32, payload padded to 8)
//
// Section tags: META (UTF-8 JSON), KPTS (f32 [x,y,scale,angle]), DESC (u32 x8),
// TREE (i32), IMG0..IMG2 (u8 greyscale), TPTS (f32 [x,y]).

export const MAGIC = 0x52545346; // 'FSTR' little-endian
export const FORMAT_VERSION = 1;

export interface TargetImageInfo {
  width: number;
  height: number;
  /** Base pixels per image pixel. */
  scale: number;
}

export interface QualityReport {
  features: number;
  featuresPerLevel: number[];
  /** Fraction of a 8x8 grid over the target with at least one feature. */
  coverage: number;
  /** Fraction of descriptors with a near-identical twin elsewhere on the target. */
  selfSimilarity: number;
  trackingPoints: number;
  warnings: string[];
}

export interface TargetMeta {
  name: string;
  /** Base image size in pixels; all target coordinates use base pixels. */
  width: number;
  height: number;
  /** Physical width in metres. */
  physicalWidth: number;
  levelScales: number[];
  images: TargetImageInfo[];
  quality: QualityReport;
}

export interface TargetData {
  meta: TargetMeta;
  kp: Float32Array;
  desc: Uint32Array;
  tree: Int32Array;
  images: Uint8Array[];
  trackPoints: Float32Array;
}

const tag = (s: string) => s.charCodeAt(0) | (s.charCodeAt(1) << 8) | (s.charCodeAt(2) << 16) | (s.charCodeAt(3) << 24);
const pad8 = (n: number) => (n + 7) & ~7;

export function encodeTargets(targets: TargetData[]): ArrayBuffer {
  const blocks = targets.map(encodeTarget);
  const headerSize = pad8(16 + 8 * targets.length);
  const total = headerSize + blocks.reduce((s, b) => s + b.byteLength, 0);
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  dv.setUint32(0, MAGIC, true);
  dv.setUint32(4, FORMAT_VERSION, true);
  dv.setUint32(8, targets.length, true);
  let off = headerSize;
  blocks.forEach((b, i) => {
    dv.setUint32(16 + 8 * i, off, true);
    dv.setUint32(20 + 8 * i, b.byteLength, true);
    new Uint8Array(buf, off, b.byteLength).set(b);
    off += b.byteLength;
  });
  return buf;
}

function encodeTarget(t: TargetData): Uint8Array {
  const sections: Array<[string, Uint8Array]> = [
    ['META', new TextEncoder().encode(JSON.stringify(t.meta))],
    ['KPTS', bytes(t.kp)],
    ['DESC', bytes(t.desc)],
    ['TREE', bytes(t.tree)],
    ...t.images.map((im, i): [string, Uint8Array] => [`IMG${i}`, im]),
    ['TPTS', bytes(t.trackPoints)],
  ];
  const size = sections.reduce((s, [, d]) => s + 8 + pad8(d.byteLength), 0);
  const out = new Uint8Array(size);
  const dv = new DataView(out.buffer);
  let o = 0;
  for (const [name, data] of sections) {
    dv.setUint32(o, tag(name), true);
    dv.setUint32(o + 4, data.byteLength, true);
    out.set(data, o + 8);
    o += 8 + pad8(data.byteLength);
  }
  return out;
}

function bytes(a: ArrayBufferView): Uint8Array {
  return new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
}

/** Parse a targets file. Arrays are views into `buf` (no copies). */
export function decodeTargets(buf: ArrayBuffer): TargetData[] {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC) throw new Error('fastar: not a targets file');
  const version = dv.getUint32(4, true);
  if (version !== FORMAT_VERSION) throw new Error(`fastar: unsupported targets version ${version}`);
  const count = dv.getUint32(8, true);
  const out: TargetData[] = [];
  for (let i = 0; i < count; i++) {
    const start = dv.getUint32(16 + 8 * i, true);
    const len = dv.getUint32(20 + 8 * i, true);
    const sec = new Map<number, [number, number]>();
    for (let o = start; o < start + len;) {
      const tg = dv.getUint32(o, true), n = dv.getUint32(o + 4, true);
      sec.set(tg, [o + 8, n]);
      o += 8 + pad8(n);
    }
    const get = (name: string): [number, number] => {
      const s = sec.get(tag(name));
      if (!s) throw new Error(`fastar: missing section ${name}`);
      return s;
    };
    const [mo, mn] = get('META');
    const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, mo, mn))) as TargetMeta;
    const [ko, kn] = get('KPTS');
    const [d0, dn] = get('DESC');
    const [to, tn] = get('TREE');
    const [po, pn] = get('TPTS');
    const images = meta.images.map((_, k) => {
      const [io, inn] = get(`IMG${k}`);
      return new Uint8Array(buf, io, inn);
    });
    out.push({
      meta,
      kp: new Float32Array(buf, ko, kn / 4),
      desc: new Uint32Array(buf, d0, dn / 4),
      tree: new Int32Array(buf, to, tn / 4),
      images,
      trackPoints: new Float32Array(buf, po, pn / 4),
    });
  }
  return out;
}
