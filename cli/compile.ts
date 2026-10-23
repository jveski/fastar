// fastar-compile: build a targets file from images.
//
//   fastar-compile -o targets.bin poster.png:0.42 cover.jpg:0.21
//
// Each image takes its physical width in metres after a colon (or --width).
import { readFileSync, writeFileSync } from 'node:fs';
import { basename, extname } from 'node:path';
import { decode as decodeJpeg } from 'jpeg-js';
import { decodePng } from './png';
import { compileTargets } from '../src/target/compiler';
import { rgbaToGray, type GrayImage } from '../src/image/gray';

const HELP = `Usage: fastar-compile [options] <image[:widthMetres]>...

Options:
  -o, --out <file>      output file (default: targets.bin)
  -w, --width <m>       default physical width in metres
      --max-dim <px>    base image long side (default 1024)
      --report <file>   write the quality report as JSON
  -h, --help            show this help

Images: PNG, JPEG or binary PGM.`;

function loadImage(path: string): GrayImage {
  const buf = readFileSync(path);
  const ext = extname(path).toLowerCase();
  if (ext === '.png') return decodePng(buf);
  if (ext === '.pgm') return decodePgm(buf);
  if (ext === '.jpg' || ext === '.jpeg') {
    const r = decodeJpeg(buf, { useTArray: true, formatAsRGBA: true });
    return rgbaToGray(r.data, r.width, r.height);
  }
  throw new Error(`unsupported image type: ${ext}`);
}

function decodePgm(buf: Uint8Array): GrayImage {
  const text = new TextDecoder('latin1').decode(buf.subarray(0, 64));
  const m = /^P5\s+(?:#.*\s+)*(\d+)\s+(\d+)\s+(\d+)\s/.exec(text);
  if (!m) throw new Error('only binary PGM (P5) is supported');
  const [hdr, w, h] = [m[0].length, +m[1], +m[2]];
  return { width: w, height: h, data: new Uint8Array(buf.subarray(hdr, hdr + w * h)) };
}

async function main(argv: string[]) {
  let out = 'targets.bin', width: number | undefined, maxDim: number | undefined, report: string | undefined;
  const inputs: Array<{ path: string; width?: number }> = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '-h' || a === '--help') { console.log(HELP); return; }
    else if (a === '-o' || a === '--out') out = argv[++i];
    else if (a === '-w' || a === '--width') width = parseFloat(argv[++i]);
    else if (a === '--max-dim') maxDim = parseInt(argv[++i], 10);
    else if (a === '--report') report = argv[++i];
    else if (a.startsWith('-')) throw new Error(`unknown option ${a}`);
    else {
      const m = /^(.*?)(?::([0-9.]+))?$/.exec(a)!;
      inputs.push({ path: m[1], width: m[2] ? parseFloat(m[2]) : undefined });
    }
  }
  if (inputs.length === 0) { console.log(HELP); process.exitCode = 1; return; }
  const list = [];
  for (const inp of inputs) {
    const w = inp.width ?? width;
    if (!w) throw new Error(`no physical width for ${inp.path}; use ${inp.path}:<metres> or --width`);
    list.push({ image: loadImage(inp.path), physicalWidth: w, maxDim, name: basename(inp.path, extname(inp.path)) });
  }
  const t0 = Date.now();
  const { buffer, targets } = compileTargets(list);
  writeFileSync(out, new Uint8Array(buffer));
  targets.forEach((t, i) => {
    const q = t.meta.quality;
    console.log(
      `[${i}] ${t.meta.name}: ${t.meta.width}x${t.meta.height}px, ${t.meta.physicalWidth} m wide, ` +
        `${q.features} features, ${(q.coverage * 100).toFixed(0)}% coverage, ` +
        `${(q.selfSimilarity * 100).toFixed(0)}% self-similar, ${q.trackingPoints} tracking points`,
    );
    for (const w of q.warnings) console.warn(`    warning: ${w}`);
  });
  console.log(`wrote ${out} (${(buffer.byteLength / 1024).toFixed(0)} KB) in ${Date.now() - t0} ms`);
  if (report) writeFileSync(report, JSON.stringify(targets.map((t) => ({ name: t.meta.name, ...t.meta.quality })), null, 2));
}

main(process.argv.slice(2)).catch((e: Error) => {
  console.error(`fastar-compile: ${e.message}`);
  process.exit(1);
});
