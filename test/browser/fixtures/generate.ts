// Generate the committed browser-test fixtures:
//   targets-image.bin / targets-video.bin   compiled targets
//   <scenario>.mp4                          rendered camera video
//   <scenario>.session.json                 SessionLog for start({ replay })
//   <scenario>.truth.json                   per-frame ground truth
//   target.png                              the image target, for debugging
//
// Run with: npm run fixtures:browser
// Requires ffmpeg on PATH (only when regenerating; the output is committed).
import { spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { intrinsicsFromFov } from '../../../src/vision/camera';
import { makePoster } from '../../../src/sim/poster';
import { makeMotionClip } from '../../../src/sim/motion';
import { compileTarget } from '../../../src/target/compiler';
import { compileVideoTarget as compileVideo } from '../../../src/target/video-compiler';
import { encodeTargets } from '../../../src/target/format';
import { createSceneRenderer, relativePose, scenarios, simulateImu, type Scenario } from '../../../src/sim/scene';
import { videoTimeAt } from '../../../src/sim/harness';
import type { SessionLog } from '../../../src/shell/recorder';
import type { TargetData } from '../../../src/target/format';
import type { GrayImage } from '../../../src/image/gray';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, 'data');
const W = 640, H = 480, FPS = 30;
const K = intrinsicsFromFov(W, H, 63);
const POSTER_W_M = 0.2;
const TD = 0.02;
const IMU_HZ = 60;

interface Fixture {
  name: string;
  scn: Scenario;
  poster: GrayImage;
  posterWidthM: number;
  target: TargetData;
  targetFile: string;
  video?: { clip: ReturnType<typeof makeMotionClip>; start: number };
}

function renderVideo(out: string, frames: Uint8Array[]): void {
  const input = Buffer.concat(frames.map((f) => Buffer.from(f)));
  const r = spawnSync('ffmpeg', [
    '-hide_banner', '-loglevel', 'error', '-y',
    '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', `${W}x${H}`, '-r', String(FPS), '-i', '-',
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart', out,
  ], { input, maxBuffer: 1 << 30 });
  if (r.status !== 0) throw new Error(`ffmpeg failed for ${out}: ${r.stderr?.toString()}`);
}

function makeSession(f: Fixture, frames: GrayImage[], truth: Array<Record<string, unknown>>): void {
  renderVideo(join(OUT, `${f.name}.mp4`), frames.map((x) => x.data));

  const imu = simulateImu(f.scn, { rate: IMU_HZ, gyroNoise: 0.004, accelNoise: 0.05, seed: 5 })
    .map((s) => [s.t * 1000 + TD * 1000, s.gyro[0], s.gyro[1], s.gyro[2], s.accel[0], s.accel[1], s.accel[2]]);

  const log: SessionLog = {
    version: 1,
    device: {
      userAgent: `fastar-browser-fixture/${f.name}`, videoWidth: W, videoHeight: H, width: W, height: H,
      intrinsics: K, sensors: 'devicemotion', path: 'copy', td: TD, tdCalibrated: false,
    },
    startTime: 0,
    frames: frames.map((_, k) => ({
      t: (k * 1000) / FPS,
      mediaTime: k / FPS,
      presentedFrames: k,
      expectedDisplayTime: ((k + 1) * 1000) / FPS,
    })),
    imu,
  };
  writeFileSync(join(OUT, `${f.name}.session.json`), JSON.stringify(log));

  writeFileSync(join(OUT, `${f.name}.truth.json`), JSON.stringify({
    name: f.name,
    fps: FPS,
    width: W,
    height: H,
    targetWidth: f.target.meta.width,
    targetHeight: f.target.meta.height,
    posterWidthM: f.posterWidthM,
    intrinsics: K,
    targetFile: f.targetFile,
    duration: frames.length / FPS,
    video: f.video ? { duration: f.video.clip.duration, start: f.video.start } : null,
    frames: truth,
  }));
}

mkdirSync(OUT, { recursive: true });

const poster = makePoster(640, 480, 11);
const imageTarget = compileTarget(poster, { physicalWidth: POSTER_W_M, name: 'browser-image' });
writeFileSync(join(OUT, 'targets-image.bin'), Buffer.from(encodeTargets([imageTarget])));

// Preserve the poster as an 8-bit PNG via ffmpeg for manual inspection.
{
  const png = spawnSync('ffmpeg', [
    '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
    '-s', `${poster.width}x${poster.height}`, '-i', '-', '-frames:v', '1', join(OUT, 'target.png'),
  ], { input: Buffer.from(poster.data), maxBuffer: 1 << 28 });
  if (png.status !== 0) throw new Error(`ffmpeg png failed: ${png.stderr?.toString()}`);
}

// The video clip is smaller than the camera frame: it only needs to be sharp
// enough to compile and track when shown at 0.5 m wide.
const clip = makeMotionClip({ width: 480, height: 270, fps: FPS, duration: 3 });
const videoTarget = compileVideo(Array.from({ length: clip.frameCount }, (_, i) => clip.frame(i)), { physicalWidth: 0.5, fps: FPS });
writeFileSync(join(OUT, 'targets-video.bin'), Buffer.from(encodeTargets([videoTarget])));

const fixtures: Fixture[] = [
  { name: 'static', scn: scenarios.static(0.3, 3), poster, posterWidthM: POSTER_W_M, target: imageTarget, targetFile: 'targets-image.bin' },
  { name: 'sweep', scn: scenarios.slowSweep(0.35, 4), poster, posterWidthM: POSTER_W_M, target: imageTarget, targetFile: 'targets-image.bin' },
  { name: 'occlusion', scn: scenarios.occlusion(0.35, 4, 1.5, 0.6), poster, posterWidthM: POSTER_W_M, target: imageTarget, targetFile: 'targets-image.bin' },
  {
    name: 'video', scn: scenarios.static(0.6, 3), poster: clip.frame(0), posterWidthM: 0.5,
    target: videoTarget, targetFile: 'targets-video.bin', video: { clip, start: 0 },
  },
];

for (const f of fixtures) {
  const renderer = createSceneRenderer(f.scn, {
    K, poster: f.poster, posterWidthM: f.posterWidthM,
    textureAt: f.video ? (t) => f.video!.clip.frame(Math.floor(videoTimeAt({ clip: f.video!.clip, start: f.video!.start }, t) * FPS + 1e-6)) : undefined,
  });
  const n = Math.round(f.scn.duration * FPS);
  const frames: GrayImage[] = [];
  const truth: Array<Record<string, unknown>> = [];
  for (let k = 0; k < n; k++) {
    const t = k / FPS;
    frames.push(renderer.render(t, f.scn.camera, { noise: 2, blurPasses: 1, readout: 0.008, exposure: 0.006, exposureSamples: 2 }));
    const rel = relativePose(f.scn.camera(t), f.scn.target(t));
    truth.push({
      t,
      mediaTime: k / FPS,
      R: Array.from(rel.R),
      tv: Array.from(rel.t),
      visible: !(f.scn.occluded?.(t) ?? false),
      videoTime: f.video ? videoTimeAt({ clip: f.video.clip, start: f.video.start }, t) : null,
    });
  }
  makeSession(f, frames, truth);
  console.log(`${f.name}: ${n} frames`);
}

console.log(`wrote fixtures to ${OUT}`);
