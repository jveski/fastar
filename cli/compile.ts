// fastar-compile: build a targets file from images and videos.
//
//   fastar-compile -o targets.bin poster.png:0.42 cover.jpg:0.21 loop.mp4:1.2
//
// Each input takes its physical width in metres after a colon (or --width).
// Videos are decoded with ffmpeg (which must be on the PATH); a directory of
// numbered frames works too, with --fps.
import { readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { spawn, spawnSync } from 'node:child_process';
import { basename, extname, join } from 'node:path';
import { decode as decodeJpeg } from 'jpeg-js';
import { decodePng } from './png';
import { compileTarget } from '../src/target/compiler';
import { VideoTargetBuilder } from '../src/target/video-compiler';
import { encodeTargets, type TargetData } from '../src/target/format';
import { rgbaToGray, type GrayImage } from '../src/image/gray';

const HELP = `Usage: fastar-compile [options] <input[:widthMetres]>...

Inputs: images (PNG, JPEG, binary PGM), videos (anything ffmpeg reads:
.mp4, .mov, .webm, ...) or a directory of numbered image frames.

Options:
  -o, --out <file>          output file (default: targets.bin)
  -w, --width <m>           default physical width in metres
      --max-dim <px>        analysed long side (default 1024)
      --report <file>       write the quality report as JSON
  -h, --help                show this help

Video options:
      --fps <n>             frame rate (resamples videos; required for frame directories)
      --no-loop             the video does not loop
      --keyframes <s>       seconds between keyframes (default 0.5)
      --points <n>          tracking points to keep on screen (default 160)`;

const VIDEO_EXT = new Set(['.mp4', '.m4v', '.mov', '.webm', '.mkv', '.avi', '.gif', '.mpg', '.mpeg', '.ogv']);
const IMAGE_EXT = new Set(['.png', '.jpg', '.jpeg', '.pgm']);

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

interface VideoOpts {
  fps?: number;
  loop: boolean;
  keyframes?: number;
  points?: number;
  maxDim?: number;
}

/** Analysed frame size for a source size, as the builder would choose it. */
function analysisSize(w: number, h: number, maxDim: number): [number, number] {
  const s = Math.max(w, h) > maxDim ? maxDim / Math.max(w, h) : 1;
  return [Math.round(w * s), Math.round(h * s)];
}

function parseRate(r: string | undefined): number {
  if (!r) return NaN;
  const [a, b] = r.split('/').map(Number);
  return b ? a / b : a;
}

async function compileVideoFile(path: string, name: string, widthM: number, o: VideoOpts): Promise<TargetData> {
  const probe = spawnSync('ffprobe', [
    '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate', '-of', 'json', path,
  ], { encoding: 'utf8' });
  if (probe.error) throw new Error(`videos need ffmpeg/ffprobe on the PATH (${(probe.error as Error).message}); or pass a directory of frames with --fps`);
  if (probe.status !== 0) throw new Error(`ffprobe failed on ${path}: ${probe.stderr.trim()}`);
  const st = (JSON.parse(probe.stdout).streams ?? [])[0];
  if (!st) throw new Error(`no video stream in ${path}`);
  const srcFps = parseRate(st.avg_frame_rate) || parseRate(st.r_frame_rate);
  const fps = o.fps ?? srcFps;
  if (!(fps > 0)) throw new Error(`cannot tell the frame rate of ${path}; pass --fps`);
  const [W, H] = analysisSize(st.width, st.height, o.maxDim ?? 1024);
  const filters = [...(o.fps ? [`fps=${o.fps}`] : []), `scale=${W}:${H}:flags=area`, 'format=gray'];
  const builder = new VideoTargetBuilder({
    name, physicalWidth: widthM, fps, loop: o.loop, maxDim: o.maxDim, keyframeInterval: o.keyframes, pointsPerFrame: o.points,
  });
  const ff = spawn('ffmpeg', ['-v', 'error', '-i', path, '-vf', filters.join(','), '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const frame = new Uint8Array(W * H);
  let fill = 0, err = '';
  ff.stderr.on('data', (d: Buffer) => { err += d.toString(); });
  await new Promise<void>((resolve, reject) => {
    ff.on('error', (e) => reject(new Error(`could not run ffmpeg: ${e.message}`)));
    ff.stdout.on('data', (chunk: Buffer) => {
      let o2 = 0;
      while (o2 < chunk.length) {
        const n = Math.min(chunk.length - o2, frame.length - fill);
        frame.set(chunk.subarray(o2, o2 + n), fill);
        fill += n; o2 += n;
        if (fill === frame.length) {
          builder.addFrame({ width: W, height: H, data: frame });
          fill = 0;
        }
      }
    });
    ff.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg failed on ${path}: ${err.trim()}`))));
  });
  return builder.finish();
}

function compileFrameDir(dir: string, name: string, widthM: number, o: VideoOpts): TargetData {
  if (!o.fps) throw new Error(`frame directory ${dir} needs --fps`);
  const files = readdirSync(dir)
    .filter((f) => IMAGE_EXT.has(extname(f).toLowerCase()))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  if (files.length < 2) throw new Error(`need at least two frames in ${dir}`);
  const builder = new VideoTargetBuilder({
    name, physicalWidth: widthM, fps: o.fps, loop: o.loop, maxDim: o.maxDim, keyframeInterval: o.keyframes, pointsPerFrame: o.points,
  });
  for (const f of files) builder.addFrame(loadImage(join(dir, f)));
  return builder.finish();
}

async function main(argv: string[]) {
  let out = 'targets.bin', width: number | undefined, maxDim: number | undefined, report: string | undefined;
  const vo: VideoOpts = { loop: true };
  const inputs: Array<{ path: string; width?: number }> = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '-h' || a === '--help') { console.log(HELP); return; }
    else if (a === '-o' || a === '--out') out = argv[++i];
    else if (a === '-w' || a === '--width') width = parseFloat(argv[++i]);
    else if (a === '--max-dim') maxDim = parseInt(argv[++i], 10);
    else if (a === '--report') report = argv[++i];
    else if (a === '--fps') vo.fps = parseFloat(argv[++i]);
    else if (a === '--no-loop') vo.loop = false;
    else if (a === '--keyframes') vo.keyframes = parseFloat(argv[++i]);
    else if (a === '--points') vo.points = parseInt(argv[++i], 10);
    else if (a.startsWith('-')) throw new Error(`unknown option ${a}`);
    else {
      const m = /^(.*?)(?::([0-9.]+))?$/.exec(a)!;
      inputs.push({ path: m[1], width: m[2] ? parseFloat(m[2]) : undefined });
    }
  }
  if (inputs.length === 0) { console.log(HELP); process.exitCode = 1; return; }
  vo.maxDim = maxDim;
  const t0 = Date.now();
  const targets: TargetData[] = [];
  for (const inp of inputs) {
    const w = inp.width ?? width;
    if (!w) throw new Error(`no physical width for ${inp.path}; use ${inp.path}:<metres> or --width`);
    const ext = extname(inp.path).toLowerCase();
    const name = basename(inp.path, ext);
    if (statSync(inp.path).isDirectory()) targets.push(compileFrameDir(inp.path, basename(inp.path), w, vo));
    else if (VIDEO_EXT.has(ext)) targets.push(await compileVideoFile(inp.path, name, w, vo));
    else targets.push(compileTarget(loadImage(inp.path), { physicalWidth: w, maxDim, name }));
  }
  const buffer = encodeTargets(targets);
  writeFileSync(out, new Uint8Array(buffer));
  targets.forEach((t, i) => {
    const q = t.meta.quality, v = t.meta.video;
    if (v) {
      console.log(
        `[${i}] ${t.meta.name}: ${t.meta.width}x${t.meta.height}px video, ${v.duration.toFixed(1)} s at ${+v.fps.toFixed(3)} fps, ` +
          `${t.meta.physicalWidth} m wide, ${q.features} features over ${v.keyframes.length} keyframes, ` +
          `${v.report.points} tracking points (${v.report.movingPoints} moving, at least ${v.report.minPoints} on screen), ` +
          `${v.changes.length} cut${v.changes.length === 1 ? '' : 's'}`,
      );
    } else {
      console.log(
        `[${i}] ${t.meta.name}: ${t.meta.width}x${t.meta.height}px, ${t.meta.physicalWidth} m wide, ` +
          `${q.features} features, ${(q.coverage * 100).toFixed(0)}% coverage, ` +
          `${(q.selfSimilarity * 100).toFixed(0)}% self-similar, ${q.trackingPoints} tracking points`,
      );
    }
    for (const w of q.warnings) console.warn(`    warning: ${w}`);
  });
  console.log(`wrote ${out} (${(buffer.byteLength / 1024).toFixed(0)} KB) in ${Date.now() - t0} ms`);
  if (report) {
    writeFileSync(report, JSON.stringify(targets.map((t) => ({ name: t.meta.name, ...t.meta.quality, ...(t.meta.video ? { video: t.meta.video.report } : {}) })), null, 2));
  }
}

main(process.argv.slice(2)).catch((e: Error) => {
  console.error(`fastar-compile: ${e.message}`);
  process.exit(1);
});
