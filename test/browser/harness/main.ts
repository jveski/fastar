// Browser-side test harness. It loads the built /fastar.js and drives the
// public API with a synthetic camera stream, a recorded session, or a real
// getUserMedia camera. Results are exposed as JSON for Playwright/Appium.
import type { CreateOptions, FrameEvent, ImageTracker, SessionLog } from '../../../src/index';
import { SURFACE_GRID, type FrameSample, type HarnessResult, type PoseSample, type StatsSample, type Truth } from '../types';

const LIB_URL = '/fastar.js';
// Fixtures are rendered with the default 63° horizontal FOV. Real phones get a
// lookup-table value from their user agent; pin it so mobile emulation is
// compared against the same ground truth.
const HFOV = 63;
// All replay/live runs are shorter than this; only a stuck media element
// should ever hit it.
const RUN_TIMEOUT = 45_000;
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
const noOffscreen = p.get('no-offscreen') === '1';
const denyMotion = p.get('deny') === '1';
const OffscreenCtor = (globalThis as { OffscreenCanvas?: typeof OffscreenCanvas }).OffscreenCanvas;
const useImu = p.get('useImu') !== '0';
const runOverride = Number(p.get('runMs') ?? 0);
const targetFile = p.get('target') ?? 'targets-image.bin';
const deformable = p.get('deformable') === '1';
const useTruth = p.get('truth') !== '0';
// Tracking time per frame (ms), to exercise the budget.
const trackMs = Number(p.get('trackMs') ?? 0) || undefined;
// Slow devices can play the clip slower to give acquisition more wall time;
// truth is keyed by mediaTime, so the comparison is unchanged.
const playbackRate = Number(p.get('rate') ?? 1);
// A slow device's tracker: the worker spends about this long (ms) more on each frame, varying +/-30%.
const slowWorker = Number(p.get('slowWorker') ?? 0);

/** Create options shared by every mode. */
function createOptions(): Omit<CreateOptions, 'targets'> {
  return { deformable, budget: trackMs ? { trackMs } : undefined, workerUrl: slowWorker > 0 ? slowWorkerUrl(slowWorker) : undefined };
}

/** The real worker script, with a busy wait before each frame it is sent. */
function slowWorkerUrl(ms: number): string {
  const src = `importScripts(${JSON.stringify(new URL('/worker.js', location.href).href)});
const handle = self.onmessage;
self.onmessage = (e) => {
  if (e.data && e.data.type === 'frame') { const end = performance.now() + ${ms} * (0.7 + 0.6 * Math.random()); while (performance.now() < end); }
  handle.call(self, e);
};`;
  return URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
}

const result: HarnessResult = {
  done: false, error: null, mode, scenario, startedAt: 0, endedAt: 0,
  frames: [], events: { found: [], lost: [] }, stats: null,
  logs: [], probe: null,
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

// Worker failures only reach console.error, so keep them in the result.
for (const level of ['error', 'warn'] as const) {
  const orig = console[level].bind(console);
  console[level] = (...args: unknown[]) => {
    result.logs.push(`${level}: ${args.map((a) => String(a)).join(' ')}`.slice(0, 1000));
    if (result.logs.length > 50) result.logs.shift();
    orig(...args);
  };
}
window.addEventListener('error', (e) => result.logs.push(`window error: ${e.message}`));
window.addEventListener('unhandledrejection', (e) => result.logs.push(`rejection: ${String(e.reason)}`));

function log(msg: string): void {
  status.textContent = msg;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function setDone(err: unknown = null): void {
  // The watchdog may fire while a stuck await never settles; keep its error
  // even if that await eventually resumes.
  if (result.done) return;
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

/** Sample the displayed video at the end: a black image means decode/GPU trouble. */
function probeVideo(): void {
  const c = document.createElement('canvas');
  c.width = 64;
  c.height = 48;
  let meanLuma = -1;
  let offscreenLuma = -1;
  let error: string | null = null;
  const mean = (ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D): number => {
    ctx.drawImage(targetVideo, 0, 0, 64, 48);
    const d = ctx.getImageData(0, 0, 64, 48).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    return sum / (64 * 48);
  };
  try {
    meanLuma = mean(c.getContext('2d')!);
    if (OffscreenCtor) {
      const off = new OffscreenCtor(64, 48);
      offscreenLuma = mean(off.getContext('2d')!);
    }
  } catch (e) {
    error = String((e as Error)?.message ?? e);
  }
  result.probe = {
    videoTime: targetVideo.currentTime,
    paused: targetVideo.paused,
    readyState: targetVideo.readyState,
    videoWidth: targetVideo.videoWidth,
    videoHeight: targetVideo.videoHeight,
    meanLuma,
    offscreenLuma,
    error,
  };
}

function snapshot(ar: ImageTracker): StatsSample {
  const s = ar.stats;
  return {
    path: s.path, sensors: s.sensors, motionPermission: s.motionPermission,
    imuSamples: s.imuSamples, fps: s.fps, latency: s.latency,
    trackerReady: s.trackerReady, acquireReady: s.acquireReady, td: s.td, tdCalibrated: s.tdCalibrated, budgetMs: s.budgetMs,
    targets: s.targets.map((t) => ({
      index: t.index, status: t.status, confidence: t.confidence, inliers: t.inliers, rms: t.rms,
      priority: t.priority, points: t.points, every: t.every,
    })),
    performance: { ...ar.performance },
  };
}

function attach(ar: ImageTracker, frames: FrameSample[], events: HarnessResult['events']): void {
  ar.on('found', (i) => { events.found.push(performance.now()); (events.foundIndex ??= []).push(i); });
  ar.on('lost', () => events.lost.push(performance.now()));
  ar.on('near', (i) => (events.near ??= []).push(i));
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
        surface: pose.surface ? SURFACE_GRID.flatMap(([u, v]) => Array.from(pose.surface!.sample(u, v).position)) : undefined,
      });
    }
    frames.push({
      captureTime: e.presentedFrame.captureTime,
      mediaTime: e.presentedFrame.mediaTime,
      presentedFrames: e.presentedFrame.presentedFrames,
      hasImage: !!e.image,
      shownAt: performance.now(),
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
  const ar = await ImageTracker.create({ targets: fixture(targetFile), ...createOptions() });
  const frames: FrameSample[] = [];
  const events: HarnessResult['events'] = { found: [], lost: [] };
  attach(ar, frames, events);
  // Set the rate before playback starts: on a slow device the worker handshake
  // can take seconds, and a clip that ran at 1x during startup may be over
  // before acquisition gets a chance.
  if (playbackRate > 0 && playbackRate !== 1) targetVideo.playbackRate = playbackRate;
  await ar.start({
    video: targetVideo, replay: { log, video: fixture(`${scenario}.mp4`) },
    syncDisplay: sync, forceCopyPath: forceCopy, useImu, hfov: HFOV,
  });
  if (playbackRate > 0 && playbackRate !== 1) targetVideo.playbackRate = playbackRate;
  noteSizes(ar);
  await sleep(ms);
  probeVideo();
  result.frames = frames;
  result.events = events;
  result.stats = snapshot(ar);
  ar.stop();
}

async function startLive(): Promise<void> {
  const { ImageTracker } = await loadFastar();
  const ar = await ImageTracker.create({ targets: fixture(targetFile), ...createOptions() });
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
  probeVideo();
  result.frames = frames;
  result.events = events;
  result.stats = snapshot(ar);
  ar.stop();
}

async function startRecord(): Promise<void> {
  const { ImageTracker } = await loadFastar();
  const ar = await ImageTracker.create({ targets: fixture(targetFile), ...createOptions() });
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
  const ar2 = await ImageTracker.create({ targets: fixture(targetFile), ...createOptions() });
  const frames2: FrameSample[] = [];
  const events2: HarnessResult['events'] = { found: [], lost: [] };
  attach(ar2, frames2, events2);
  await ar2.start({ video: targetVideo, replay: { log: session.log, video: blobUrl }, useImu: true });
  await sleep(1800);
  result.replay = { frames: frames2, stats: snapshot(ar2), error: null };
  ar2.stop();
  setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
}

async function workerSmokeTest(): Promise<string> {
  try {
    const url = URL.createObjectURL(new Blob(['self.onmessage = () => self.postMessage("ok")'], { type: 'text/javascript' }));
    const w = new Worker(url);
    const out = await new Promise<string>((resolve) => {
      const timer = setTimeout(() => resolve('timeout'), 3000);
      w.onmessage = () => { clearTimeout(timer); resolve('ok'); };
      w.onerror = (e) => { clearTimeout(timer); resolve(`error: ${e.message}`); };
      w.postMessage(0);
    });
    w.terminate();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    return out;
  } catch (e) {
    return `throw: ${String((e as Error)?.message ?? e)}`;
  }
}

async function run(): Promise<void> {
  result.startedAt = performance.now();
  // A browser that never starts playback (seen for canvas captureStream in
  // Mobile Safari) would otherwise leave the runner polling forever.
  const watchdog = setTimeout(() => {
    setDone(new Error(`${mode} harness timed out after ${RUN_TIMEOUT} ms`));
  }, RUN_TIMEOUT);
  try {
    localStorage.clear();
    result.diagnostics.worker = await workerSmokeTest();
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
  } finally {
    clearTimeout(watchdog);
  }
}

// Environment patches must happen before fastar.js snapshots prototypes.
if (noRvfc) {
  delete (HTMLVideoElement.prototype as unknown as Record<string, unknown>).requestVideoFrameCallback;
}
if (noOffscreen) {
  // Chrome 113 on the Android emulator can return black from drawImage(video)
  // on an OffscreenCanvas; use the HTMLCanvasElement path instead.
  try {
    Object.defineProperty(globalThis, 'OffscreenCanvas', { value: undefined, configurable: true });
  } catch {
    // Leave the native constructor in place if the global is not configurable.
  }
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
