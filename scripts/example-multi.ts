// Generates the multi-target demo in site/demo/multi: four different
// looping "marbled ink" clips (clips/<name>.mp4) to show side by side on one
// screen, one targets.bin holding all four as video targets, and per clip
// the content its overlay loads when the clip is found (clips/<name>.json:
// the swirl at the clip's centre, plus the overlay's shape and colour).
//   npx tsx scripts/example-multi.ts
import { mkdirSync, writeFileSync } from 'node:fs';
import { execFileSync, spawn } from 'node:child_process';
import { makeShowcaseClip } from '../src/sim/showcase';

const OUT = 'site/demo/multi';
// Each clip is shown about 14 cm wide (a quarter of a laptop screen); change to your tiles' width.
const WIDTH_M = 0.14;
// Fewer points on screen at a time keep each target, and targets.bin, small.
const POINTS = 60;
const CLIPS = [
  { name: 'amber', seed: 11, shape: 'cube', color: '#ffb02e' },
  { name: 'cobalt', seed: 12, shape: 'torusKnot', color: '#3d7bff' },
  { name: 'jade', seed: 13, shape: 'icosahedron', color: '#2fd17c' },
  { name: 'coral', seed: 14, shape: 'cone', color: '#ff5f6d' },
];

mkdirSync(`${OUT}/clips`, { recursive: true });
const round = (_: string, v: unknown) => (typeof v === 'number' ? Math.round(v * 1e6) / 1e6 : v);

for (const c of CLIPS) {
  const clip = makeShowcaseClip({ width: 640, fps: 30, seed: c.seed });
  const ff = spawn('ffmpeg', [
    '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${clip.width}x${clip.height}`, '-r', String(clip.fps), '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '27', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', `${OUT}/clips/${c.name}.mp4`,
  ], { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise<void>((resolve, reject) => {
    ff.on('error', (e) => reject(new Error(`could not run ffmpeg: ${e.message}`)));
    ff.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exited with ${code}`))));
  });
  for (let i = 0; i < clip.frameCount; i++) {
    if (!ff.stdin.write(Buffer.from(clip.frameRgba(i).buffer))) await new Promise((r) => ff.stdin.once('drain', r));
  }
  ff.stdin.end();
  await done;
  // Waves are [amp, cycles, phase]: value(t) = sum of amp * sin(2 pi cycles t / loop + phase).
  writeFileSync(`${OUT}/clips/${c.name}.json`, JSON.stringify({ ...clip.scene, shape: c.shape, color: c.color }, round, 2) + '\n');
  console.log(`${c.name}: ${clip.frameCount} frames`);
}

execFileSync('npx', [
  'tsx', 'cli/compile.ts', '--points', String(POINTS), '-o', `${OUT}/targets.bin`,
  ...CLIPS.map((c) => `${OUT}/clips/${c.name}.mp4:${WIDTH_M}`),
], { stdio: 'inherit' });
