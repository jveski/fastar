// Browser-side test harness. It loads the built /fastar.js and drives the
// public API with a synthetic camera stream, a recorded session, or a real
// getUserMedia camera. Results are exposed as JSON for Playwright/Appium.
import type { FrameEvent, ImageTracker, SessionLog } from '../../../src/index';
import type { FrameSample, HarnessResult, PoseSample, StatsSample, Truth } from '../types';

const LIB_URL = '/fastar.js';
// Fixtures are rendered with the default 63° horizontal FOV. Real phones get a
// lookup-table value from their user agent; pin it so mobile emulation is
// compared against the same ground truth.
const HFOV = 63;
async function loadFastar(): Promise<typeof import('../../../src/index')> {
  return (await import(LIB_URL)) as typeof import('../../../src/index');
}

declare global {
  interface Window {
    __fastarTest: HarnessResult;
    __fastarStart: () => Promise<void>;
  }
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const targetVideo = $<HTMLVideoElement>('target');
const sourceVideo = $<HTMLVideoElement>('source');
const synthetic = $<HTMLCanvasElement>('synthetic');
const startButton = $<HTMLButtonElement>('start');
const status = $<HTMLElement>('status');

const p = new URLSearchParams(location.search);
const mode = p.get('mode') ?? 'replay';
const scenario = p.get('scenario') ?? 'static';
const sync = p.get('sync') === '1';
const forceCopy = p.get('copy') === '1';
const noRvfc = p.get('norvfc') === '1';
const noWebgl2 = p.get('no-webgl2') === '1';
const denyMotion = p.get('deny') === '1';
const useImu = p.get('useImu') !== '0';
const runOverride = Number(p.get('runMs') ?? 0);
const targetFile = p.get('target') ?? 'targets-image.bin';
const useTruth = p.get('truth') !== '0';

const result: HarnessResult = {
  done: false, error: null, mode, scenario, startedAt: 0, endedAt: 0,
  frames: [], events: { found: [], lost: [] }, stats: null,
  diagnostics: {
    userAgent: navigator.userAgent,
    devicePixelRatio: devicePixelRatio || 1,
    screen: [screen.width, screen.height],
    requestVideoFrameCallback: typeof HTMLVideoElement.prototype.requestVideoFrameCallback === 'function',
    mediaStreamTrackProcessor: typeof (globalThis as { MediaStreamTrackProcessor?: unknown }).MediaStreamTrackProcessor === 'function',
    webgl2: !!document.createElement('canvas').getContext('webgl2'),
    deviceMotionEvent: 'DeviceMotionEvent' in window,
    needsMotionPermission: !!(window as unknown as { DeviceMotionEvent?: { requestPermission?: unknown } }).DeviceMotionEvent?.requestPermission,
    secureContext: window.isSecureContext,
  },
  replay: null,
};
window.__fastarTest = result;

function log(msg: string): void {
  status.textContent = msg;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function setDone(err: unknown = null): void {
  result.error = err ? String((err as Error)?.message ?? err) : null;
  result.endedAt = performance.now();
  result.done = true;
  log(result.error ?? 'done');
}

function noteSizes(ar: ImageTracker): void {
  const p = ar.processingSize;
  result.diagnostics.processing = [p.width, p.height];
  result.diagnostics.videoSize = [targetVideo.videoWidth, targetVideo.videoHeight];
}

function snapshot(ar: ImageTracker): StatsSample {
  const s = ar.stats;
  return {
    path: s.path, sensors: s.sensors, motionPermission: s.motionPermission,
    imuSamples: s.imuSamples, fps: s.fps, latency: s.latency, td: s.td, tdCalibrated: s.tdCalibrated,
    targets: s.targets.map((t) => ({
      index: t.index, status: t.status, confidence: t.confidence, inliers: t.inliers, rms: t.rms,
    })),
  };
}

function attach(ar: ImageTracker, frames: FrameSample[], events: HarnessResult['events']): void {
  ar.on('found', () => events.found.push(performance.now()));
  ar.on('lost', () => events.lost.push(performance.now()));
  ar.onFrame((e: FrameEvent) => {
    const poses: PoseSample[] = [];
    for (const [index, pose] of e.poses) {
      poses.push({
        index,
        R: Array.from(pose.R),
        t: Array.from(pose.t),
        confidence: pose.confidence,
        videoTime: pose.videoTime ?? null,
        locked: !!pose.videoTimeLocked,
      });
    }
    frames.push({
      captureTime: e.presentedFrame.captureTime,
      mediaTime: e.presentedFrame.mediaTime,
      presentedFrames: e.presentedFrame.presentedFrames,
      hasImage: !!e.image,
      poses,
      view: [targetVideo.clientWidth, targetVideo.clientHeight],
      projection: Array.from(ar.projectionMatrix),
    });
  });
}

async function fetchJson<T>(name: string): Promise<T> {
  const res = await fetch(`/fixtures/${name}`);
  if (!res.ok) throw new Error(`fixture ${name}: HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

function fixture(name: string): string {
  return `/fixtures/${name}`;
}

async function playSource(): Promise<void> {
  sourceVideo.src = fixture(`${scenario}.mp4`);
  await new Promise<void>((resolve, reject) => {
    sourceVideo.onloadeddata = () => resolve();
    sourceVideo.onerror = () => reject(new Error('source video failed to load'));
  });
  await sourceVideo.play();
}

/** Draw the fixture video into a canvas and expose it as a camera-like stream. */
async function openSyntheticStream(): Promise<MediaStream> {
  await playSource();
  const ctx = synthetic.getContext('2d')!;
  let stopped = false;
  const draw = () => {
    if (stopped) return;
    if (sourceVideo.readyState >= 2) ctx.drawImage(sourceVideo, 0, 0, synthetic.width, synthetic.height);
    requestAnimationFrame(draw);
  };
  draw();
  const stream = synthetic.captureStream(30);
  const origStop = stream.getTracks()[0].stop.bind(stream.getTracks()[0]);
  stream.getTracks()[0].stop = () => { stopped = true; origStop(); };
  return stream;
}

function cameraToDevice(cx: number, cy: number, cz: number): [number, number, number] {
  // Environment-facing inverse of deviceToCamera(), at the current screen angle.
  const a = ((screen.orientation?.angle ?? 0) * Math.PI) / 180;
  const c = Math.cos(a), s = Math.sin(a);
  const sx = cx, sy = -cy, sz = -cz;
  return [c * sx + s * sy, -s * sx + c * sy, sz];
}

/** Replay the fixture's IMU as devicemotion events, exercising MotionSensors. */
function dispatchImu(log: SessionLog): () => void {
  let i = 0;
  const t0 = performance.now();
  const timer = setInterval(() => {
    const now = performance.now() - t0;
    while (i < log.imu.length && log.imu[i][0] <= now) {
      const [, gx, gy, gz, ax, ay, az] = log.imu[i++];
      const [dx, dy, dz] = cameraToDevice(gx, gy, gz);
      const [dax, day, daz] = cameraToDevice(ax, ay, az);
      const e = new Event('devicemotion') as Event & Record<string, unknown>;
      Object.defineProperties(e, {
        rotationRate: { value: { alpha: (dx * 180) / Math.PI, beta: (dy * 180) / Math.PI, gamma: (dz * 180) / Math.PI } },
        accelerationIncludingGravity: { value: { x: dax, y: day, z: daz } },
        interval: { value: 1000 / 60 },
      });
      window.dispatchEvent(e);
    }
  }, 5);
  return () => clearInterval(timer);
}

function sessionMs(log: SessionLog): number {
  const first = log.frames[0]?.t ?? 0;
  const last = log.frames[log.frames.length - 1]?.t ?? first;
  return Math.max(1000, last - first + 200);
}

async function startReplay(log: SessionLog, ms: number): Promise<void> {
  const { ImageTracker } = await loadFastar();
  const ar = await ImageTracker.create({ targets: fixture(targetFile) });
  const frames: FrameSample[] = [];
  const events: HarnessResult['events'] = { found: [], lost: [] };
  attach(ar, frames, events);
  await ar.start({
    video: targetVideo, replay: { log, video: fixture(`${scenario}.mp4`) },
    syncDisplay: sync, forceCopyPath: forceCopy, useImu, hfov: HFOV,
  });
  noteSizes(ar);
  await sleep(ms);
  result.frames = frames;
  result.events = events;
  result.stats = snapshot(ar);
  ar.stop();
}

async function startLive(): Promise<void> {
  const { ImageTracker } = await loadFastar();
  const ar = await ImageTracker.create({ targets: fixture(targetFile) });
  const frames: FrameSample[] = [];
  const events: HarnessResult['events'] = { found: [], lost: [] };
  attach(ar, frames, events);

  let clearImu = () => {};
  if (mode === 'camera') {
    await ar.start({ video: targetVideo, requestMotionPermission: true, useImu, forceCopyPath: forceCopy, syncDisplay: sync, hfov: HFOV });
  } else {
    const stream = await openSyntheticStream();
    const log = await fetchJson<SessionLog>(`${scenario}.session.json`);
    await ar.start({
      video: targetVideo, stream, syncDisplay: sync, forceCopyPath: forceCopy, useImu,
      // The synthetic sensor test injects events itself; asking Chrome/Safari
      // for permission in an automated session would otherwise deny them.
      requestMotionPermission: mode === 'sensors' && !denyMotion ? false : true,
      hfov: HFOV,
    });
    if (mode === 'sensors') clearImu = dispatchImu(log);
  }
  noteSizes(ar);
  const truth = useTruth ? await fetchJson<Truth>(`${scenario}.truth.json`) : null;
  const ms = runOverride || (truth ? (mode === 'camera' ? truth.duration * 1000 : truth.duration * 1000 + 500) : 4000);
  await sleep(ms);
  clearImu();
  result.frames = frames;
  result.events = events;
  result.stats = snapshot(ar);
  ar.stop();
}

async function startRecord(): Promise<void> {
  const { ImageTracker } = await loadFastar();
  const ar = await ImageTracker.create({ targets: fixture(targetFile) });
  const frames: FrameSample[] = [];
  const events: HarnessResult['events'] = { found: [], lost: [] };
  attach(ar, frames, events);
  const stream = await openSyntheticStream();
  await ar.start({ video: targetVideo, stream, syncDisplay: sync, useImu: false, hfov: HFOV });
  noteSizes(ar);
  const truth = useTruth ? await fetchJson<Truth>(`${scenario}.truth.json`) : null;
  await sleep(1500);
  ar.recorder.start();
  await sleep(Math.min(2000, (truth?.duration ?? 3) * 1000));
  const session = await ar.recorder.stop();
  result.frames = frames;
  result.events = events;
  result.stats = snapshot(ar);
  ar.stop();
  if (!session.video) throw new Error('MediaRecorder produced no video');

  const blobUrl = URL.createObjectURL(session.video);
  const ar2 = await ImageTracker.create({ targets: fixture(targetFile) });
  const frames2: FrameSample[] = [];
  const events2: HarnessResult['events'] = { found: [], lost: [] };
  attach(ar2, frames2, events2);
  await ar2.start({ video: targetVideo, replay: { log: session.log, video: blobUrl }, useImu: true });
  await sleep(1800);
  result.replay = { frames: frames2, stats: snapshot(ar2), error: null };
  ar2.stop();
  setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
}

async function run(): Promise<void> {
  result.startedAt = performance.now();
  try {
    localStorage.clear();
    const truth = useTruth ? await fetchJson<Truth>(`${scenario}.truth.json`) : null;
    if (mode === 'record') {
      await startRecord();
    } else if (mode === 'replay') {
      const log = await fetchJson<SessionLog>(`${scenario}.session.json`);
      if (!useImu) log.imu = [];
      const ms = runOverride || (truth ? Math.max(1000, truth.duration * 1000 - 150) : sessionMs(log));
      await startReplay(log, ms);
    } else {
      await startLive();
    }
    setDone();
  } catch (e) {
    setDone(e);
  }
}

// Environment patches must happen before fastar.js snapshots prototypes.
if (noRvfc) {
  delete (HTMLVideoElement.prototype as unknown as Record<string, unknown>).requestVideoFrameCallback;
}
if (noWebgl2) {
  const patch = (proto: { getContext: (id: string, ...a: unknown[]) => unknown }) => {
    const orig = proto.getContext;
    proto.getContext = function (id: string, ...a: unknown[]) {
      return id === 'webgl2' ? null : orig.call(this, id, ...a);
    };
  };
  patch(HTMLCanvasElement.prototype as unknown as { getContext: (id: string, ...a: unknown[]) => unknown });
  if (typeof OffscreenCanvas !== 'undefined') {
    patch(OffscreenCanvas.prototype as unknown as { getContext: (id: string, ...a: unknown[]) => unknown });
  }
}
if (denyMotion) {
  const D = (window as unknown as { DeviceMotionEvent?: { requestPermission?: () => Promise<string> } }).DeviceMotionEvent;
  if (D && typeof D.requestPermission === 'function') D.requestPermission = async () => 'denied';
}

window.__fastarStart = run;
startButton.onclick = () => {
  startButton.remove();
  void run();
};
if (p.get('autostart') === '1') {
  startButton.click();
} else {
  log('ready');
}
