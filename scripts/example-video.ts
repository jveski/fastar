// Generates the video example in examples/video: a looping generative
// "marbled ink" animation (clip.mp4, H.264 via ffmpeg), its targets.bin
// (compiled from the encoded file, so the target matches what a screen
// shows), and scene.json describing the swirl at the screen centre for the
// AR overlay.
//   npx tsx scripts/example-video.ts
import { mkdirSync, writeFileSync } from 'node:fs';
import { execFileSync, spawn } from 'node:child_process';
import { makeShowcaseClip } from '../src/sim/showcase';

const OUT = 'examples/video';
const WIDTH_M = 0.3; // shown about 30 cm wide (a laptop screen); change to your screen's width
// The picture changes all the time, so every tracking point keeps needing new
// snapshots: fewer points on screen at a time keep targets.bin small.
const POINTS = 120;

mkdirSync(OUT, { recursive: true });
const clip = makeShowcaseClip({ width: 960, fps: 30 });

const ff = spawn('ffmpeg', [
  '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${clip.width}x${clip.height}`, '-r', String(clip.fps), '-i', '-',
  '-c:v', 'libx264', '-preset', 'slow', '-crf', '26', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', `${OUT}/clip.mp4`,
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
const round = (_: string, v: unknown) => (typeof v === 'number' ? Math.round(v * 1e6) / 1e6 : v);
writeFileSync(`${OUT}/scene.json`, JSON.stringify(clip.scene, round, 2) + '\n');
execFileSync('npx', ['tsx', 'cli/compile.ts', '--points', String(POINTS), '-o', `${OUT}/targets.bin`, `${OUT}/clip.mp4:${WIDTH_M}`], { stdio: 'inherit' });
