// Public main-thread API.
import workerSource from 'virtual:worker-source';
import { decodeTargets, type TargetMeta } from '../target/format';
import { glProjection, intrinsicsFromFocal, lookupHfov, intrinsicsFromFov, poseToGl, type FitMode, type Intrinsics } from '../vision/camera';
import { ImuBuffer } from '../fusion/imu';
import { PosePredictor } from '../fusion/predict';
import { RegularClock } from '../fusion/clock';
import { PoseSmoother, type SmoothingOptions } from '../fusion/smooth';
import { MotionSensors, type Facing, type ImuReading, type MotionPermission, type SensorBackend } from './sensors';
import { LumaGrabber, canUseTrackProcessor, createTrackProcessor, openCamera, processingSize } from './capture';
import type { FromTracker, ResultMessage, ToTracker, AcquireInit } from '../worker/protocol';
import type { EskfOptions } from '../fusion/eskf';
import type { TrackerOptions } from '../track/tracker';
import type { FrameTargetInfo } from '../track/pipeline';
import { Recorder, type SessionLog } from './recorder';

export interface CreateOptions {
  /** URL of a compiled targets file, or its bytes. */
  targets: string | URL | ArrayBuffer;
  /** Max simultaneously tracked targets. */
  maxTrack?: number;
  /** Load the worker from a URL instead of the inlined source (strict CSP). */
  workerUrl?: string | URL;
  /** Confidence needed to show a target, and below which it is hidden again. */
  showThreshold?: number;
  hideThreshold?: number;
  filter?: Partial<EskfOptions>;
  tracker?: Partial<TrackerOptions>;
  /**
   * Ease sudden pose corrections over a few frames (camera motion is never
   * delayed, and small corrections pass through). false turns it off.
   */
  smoothing?: Partial<SmoothingOptions> | false;
}

export interface StartOptions {
  video: HTMLVideoElement;
  /** Ask for motion-sensor permission (iOS); call start() from a tap. */
  requestMotionPermission?: boolean;
  facingMode?: Facing;
  /** Use an existing stream instead of opening the camera. */
  stream?: MediaStream;
  /** Set false to run vision-only. */
  useImu?: boolean;
  /** Horizontal field of view override (degrees, long side). */
  hfov?: number;
  /** Force the main-thread copy path even where MediaStreamTrackProcessor exists. */
  forceCopyPath?: boolean;
  /** Replay a recorded session (video + log) instead of the live camera and sensors. */
  replay?: { log: SessionLog; video: Blob | string };
  /**
   * Hold each camera frame until it has been tracked, then emit it with a copy
   * of the frame (`FrameEvent.image`) to draw as the background. The overlay
   * then uses the pose measured on that very frame rather than a prediction,
   * which removes most visible jitter, at the cost of the tracking time in
   * display latency. Draw `image` instead of showing the <video> element.
   */
  syncDisplay?: boolean;
}

export interface Pose {
  /** Column-major 4x4 model-view matrix (GL camera, target y up, z out of the target). */
  matrix: Float32Array;
  /** OpenCV-convention target-in-camera rotation (row-major) and translation (m). */
  R: Float64Array;
  t: Float64Array;
  confidence: number;
}

export interface PresentedFrame {
  /** Capture time of the presented frame (performance.now() ms). */
  captureTime: number;
  expectedDisplayTime: number;
  mediaTime: number;
  presentedFrames: number;
  width: number;
  height: number;
}

export interface FrameEvent {
  presentedFrame: PresentedFrame;
  poses: Map<number, Pose>;
  /** With syncDisplay: this frame's pixels, to draw as the background. Reused, so draw it now. */
  image?: HTMLCanvasElement | OffscreenCanvas;
}

/** A presented camera frame waiting for its tracking result (syncDisplay). */
interface QueuedFrame {
  captureTime: number;
  meta: VideoFrameCallbackMetadata;
  image: HTMLCanvasElement | OffscreenCanvas;
  /** A tracking result for this frame is expected. */
  expect: boolean;
  queuedAt: number;
}

// syncDisplay: frames held at once, and the longest a frame waits for its result (ms).
const DISPLAY_QUEUE = 4;
const DISPLAY_MAX_DELAY = 120;
// Capture-to-display latency (ms) assumed when rVFC gives no captureTime: 1.5 frames at 30 fps plus 10 ms.
const ASSUMED_LATENCY = 60;

export interface TrackerStats {
  path: 'track-processor' | 'copy' | 'none';
  sensors: SensorBackend;
  /** Result of the motion permission request (why sensors are 'none'). */
  motionPermission: MotionPermission;
  /** Motion samples received so far. */
  imuSamples: number;
  /** Processed frames per second. */
  fps: number;
  /** Capture to result latency (ms). */
  latency: number;
  timings: ResultMessage['timings'] | null;
  td: number;
  tdCalibrated: boolean;
  targets: FrameTargetInfo[];
}

type Listener = (index: number) => void;

const STORAGE = 'fastar:';
// Bump when a sensor-mapping or timestamp change makes stored IMU calibrations invalid.
const IMU_CAL_KEY = `${STORAGE}imu3:`;

/** Prefix an error with the start() step that failed, keeping its name. */
function stepError(step: string, e: unknown): Error {
  const err = e instanceof Error ? e : new Error(String(e));
  const out = new Error(`fastar: ${step} failed: ${err.message}`, { cause: err });
  out.name = err.name;
  return out;
}

export class ImageTracker {
  readonly targets: TargetMeta[];
  intrinsics: Intrinsics | null = null;
  readonly stats: TrackerStats = {
    path: 'none', sensors: 'none', motionPermission: 'not-requested', imuSamples: 0, fps: 0, latency: 0, timings: null, td: 0, tdCalibrated: false, targets: [],
  };
  readonly recorder: Recorder;
  /** Called with every worker result (benchmark / debugging). */
  onResult: ((r: ResultMessage) => void) | null = null;

  private readonly buffer: ArrayBuffer;
  private readonly o: Required<Pick<CreateOptions, 'maxTrack' | 'showThreshold' | 'hideThreshold'>> & CreateOptions;
  private tracker: Worker | null = null;
  private acquirer: Worker | null = null;
  private video: HTMLVideoElement | null = null;
  private stream: MediaStream | null = null;
  private ownsStream = false;
  private sensors: MotionSensors | null = null;
  private grabber: LumaGrabber | null = null;
  private readonly imu = new ImuBuffer(512);
  private readonly axisSigns = Float64Array.of(1, 1, 1);
  private readonly predictors = new Map<number, PosePredictor>();
  private readonly smoothers = new Map<number, PoseSmoother>();
  private readonly Rp = new Float64Array(9);
  private readonly tp = new Float64Array(3);
  private readonly confidence = new Map<number, number>();
  private readonly shown = new Set<number>();
  private readonly listeners = { found: new Set<Listener>(), lost: new Set<Listener>() };
  private readonly frameListeners = new Set<(e: FrameEvent) => void>();
  private readonly pool: ArrayBuffer[] = [];
  private busy = false;
  private running = false;
  private rvfcHandle = 0;
  private frameCount = 0;
  private fpsWindow: number[] = [];
  private width = 0;
  private height = 0;
  private fit: FitMode = 'cover';
  private imuBatch: number[] = [];
  private readonly proj = new Float32Array(16);
  /** Replay: recorded time + shift = playback clock. */
  private replayShift = NaN;
  private replayLog: SessionLog | null = null;
  private replayTimer = 0;
  private readonly R = new Float64Array(9);
  private readonly t = new Float64Array(3);
  /** Regularizes display-quantized frame times (WebKit). */
  private frameClock: RegularClock | null = null;
  // syncDisplay state.
  private syncDisplay = false;
  private readonly displayQueue: QueuedFrame[] = [];
  private readonly displayImages: Array<HTMLCanvasElement | OffscreenCanvas> = [];
  private displayImageNext = 0;
  /** Capture time of the newest tracked frame (ms). */
  private lastResultTime = -Infinity;

  private constructor(buffer: ArrayBuffer, o: CreateOptions) {
    this.buffer = buffer;
    this.o = { maxTrack: 1, showThreshold: 0.5, hideThreshold: 0.2, ...o };
    this.targets = decodeTargets(buffer.slice(0)).map((t) => t.meta);
    this.recorder = new Recorder(this);
  }

  static async create(o: CreateOptions): Promise<ImageTracker> {
    let buf: ArrayBuffer;
    if (o.targets instanceof ArrayBuffer) buf = o.targets;
    else {
      const res = await fetch(o.targets);
      if (!res.ok) throw new Error(`fastar: failed to load targets (${res.status})`);
      buf = await res.arrayBuffer();
    }
    return new ImageTracker(buf, o);
  }

  /** Start camera, sensors and tracking. Call from a user gesture on iOS. */
  async start(o: StartOptions): Promise<void> {
    if (this.running) return;
    // Must be requested synchronously inside the gesture, before any await.
    const permission: Promise<MotionPermission> = !MotionSensors.needsPermission()
      ? Promise.resolve('not-required')
      : o.requestMotionPermission !== false && o.useImu !== false
        ? MotionSensors.requestPermissionState()
        : Promise.resolve('not-requested');
    // iOS shows one permission prompt at a time: opening the camera while the
    // motion prompt is up leaves getUserMedia hanging (black video), so let
    // the motion prompt settle first. Neither the camera nor muted inline
    // playback needs the gesture.
    const motionState = (this.stats.motionPermission = await permission);
    const motionAllowed = motionState === 'granted' || motionState === 'not-required' || motionState === 'not-requested';
    if (!motionAllowed) console.warn(`fastar: motion sensors unavailable (${motionState}); running vision-only`);
    const facing = o.facingMode ?? 'environment';
    const video = (this.video = o.video);
    this.syncDisplay = !!o.syncDisplay;
    video.muted = true;
    video.playsInline = true;
    video.setAttribute('playsinline', '');
    if (o.replay) {
      this.replayLog = o.replay.log;
      this.ownsStream = false;
      video.srcObject = null;
      video.src = typeof o.replay.video === 'string' ? o.replay.video : URL.createObjectURL(o.replay.video);
    } else {
      this.ownsStream = !o.stream;
      this.stream = o.stream ?? (await openCamera(facing).catch((e) => { throw stepError('camera access', e); }));
      video.srcObject = this.stream;
    }
    await video.play().catch((e) => { throw stepError('video playback', e); });
    if (!video.videoWidth) await new Promise((r) => video.addEventListener('loadedmetadata', r, { once: true }));
    const { width, height } = processingSize(video.videoWidth, video.videoHeight);
    this.width = width;
    this.height = height;
    this.fit = (getComputedStyle(video).objectFit as FitMode) || 'contain';
    this.intrinsics = o.replay?.log.device.intrinsics && o.replay.log.device.width === width
      ? o.replay.log.device.intrinsics
      : this.loadIntrinsics(width, height, o.hfov);

    // Sensors first: whether we have an IMU decides the filter mode.
    let useImu = false;
    if (o.replay) {
      useImu = o.useImu !== false && o.replay.log.imu.length > 0;
      this.stats.sensors = useImu ? 'devicemotion' : 'none';
      if (useImu) this.startReplayImu(o.replay.log);
    } else if (o.useImu !== false && motionAllowed) {
      this.sensors = new MotionSensors((r) => this.onImu(r), { facing });
      this.stats.sensors = await this.sensors.start();
      useImu = this.stats.sensors !== 'none';
    }

    this.spawnWorkers(useImu, width, height);
    const track = this.stream?.getVideoTracks()[0];
    if (!o.forceCopyPath && !o.replay && canUseTrackProcessor() && track) {
      try {
        const readable = createTrackProcessor(track.clone());
        this.post({ type: 'stream', readable }, [readable as unknown as Transferable]);
        this.stats.path = 'track-processor';
      } catch {
        this.stats.path = 'copy';
      }
    } else {
      this.stats.path = 'copy';
    }
    if (this.stats.path === 'copy') {
      this.grabber = new LumaGrabber(width, height);
      for (let i = 0; i < 2; i++) this.pool.push(new ArrayBuffer(width * height));
    }
    this.running = true;
    this.scheduleFrame();
  }

  stop(): void {
    this.running = false;
    if (this.video && this.rvfcHandle && 'cancelVideoFrameCallback' in this.video) this.video.cancelVideoFrameCallback(this.rvfcHandle);
    this.sensors?.stop();
    clearInterval(this.replayTimer);
    this.tracker?.terminate();
    this.acquirer?.terminate();
    this.tracker = this.acquirer = null;
    if (this.ownsStream) this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.predictors.clear();
    this.smoothers.clear();
    this.shown.clear();
    this.displayQueue.length = 0;
    this.lastResultTime = -Infinity;
    this.frameClock = null;
  }

  /** CSS object-fit of the video, which the projection (and a drawn background) follows. */
  get fitMode(): FitMode {
    return this.fit;
  }

  on(event: 'found' | 'lost', cb: Listener): () => void {
    this.listeners[event].add(cb);
    return () => this.listeners[event].delete(cb);
  }

  onFrame(cb: (e: FrameEvent) => void): () => void {
    this.frameListeners.add(cb);
    return () => this.frameListeners.delete(cb);
  }

  /** Projection for the video element's displayed box (object-fit aware). */
  get projectionMatrix(): Float32Array {
    const v = this.video;
    return this.getProjectionMatrix(v?.clientWidth || this.width, v?.clientHeight || this.height);
  }

  getProjectionMatrix(viewW: number, viewH: number, near = 0.01, far = 100, fit: FitMode = this.fit): Float32Array {
    if (!this.intrinsics) return this.proj;
    return glProjection(this.intrinsics, viewW, viewH, near, far, fit, this.proj);
  }

  /** Low level: target pose for a frame captured at `captureTime` (ms). */
  poseAt(index: number, captureTime: number, R: Float64Array, t: Float64Array): boolean {
    const p = this.predictors.get(index);
    return !!p && p.poseAt(captureTime / 1000, this.stats.sensors !== 'none' ? this.imu : null, R, t);
  }

  get processingSize(): { width: number; height: number } {
    return { width: this.width, height: this.height };
  }

  /** The live camera stream (for recording). */
  get mediaStream(): MediaStream | null {
    return this.stream;
  }

  // --- internals ---

  private spawnWorkers(useImu: boolean, width: number, height: number): void {
    const make = () => {
      if (this.o.workerUrl) return new Worker(this.o.workerUrl);
      const url = URL.createObjectURL(new Blob([workerSource], { type: 'text/javascript' }));
      const w = new Worker(url);
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      return w;
    };
    this.tracker = make();
    this.acquirer = make();
    const ch = new MessageChannel();
    const K = this.intrinsics!;
    const acqInit: AcquireInit = { type: 'init-acquire', targets: this.buffer.slice(0), width, height, K, port: ch.port2 };
    this.acquirer.postMessage(acqInit, [acqInit.targets, ch.port2]);
    // Replays calibrate from scratch: stored values belong to this device, not the recording's.
    const cal = this.replayLog ? null : this.loadCalibration();
    if (cal) this.axisSigns.set(cal.axisSigns);
    const init: ToTracker = {
      type: 'init', targets: this.buffer.slice(0), width, height, K, maxTrack: this.o.maxTrack, useImu,
      td: cal?.td, tdCalibrated: !!cal, axisSigns: cal?.axisSigns, mainTimeOrigin: performance.timeOrigin,
      acquirePort: ch.port1, eskf: this.o.filter, tracker: this.o.tracker, calibrateFocal: !this.replayLog && this.storedFocal() === null,
    };
    this.tracker.postMessage(init, [init.targets, ch.port1]);
    this.tracker.onmessage = (e) => this.onWorker(e.data as FromTracker);
    this.tracker.onerror = (e) => console.error('fastar worker error', e.message);
  }

  private post(m: ToTracker, transfer: Transferable[] = []): void {
    this.tracker?.postMessage(m, transfer);
  }

  private onImu(r: ImuReading): void {
    const s = this.axisSigns;
    this.imu.push(r.t / 1000, r.gx * s[0], r.gy * s[1], r.gz * s[2], r.ax * s[0], r.ay * s[1], r.az * s[2]);
    this.recorder.onImu(r);
    this.stats.imuSamples++;
    // Batch a few samples per message; the worker applies its own axis signs.
    this.imuBatch.push(r.t, r.gx, r.gy, r.gz, r.ax, r.ay, r.az);
    if (this.imuBatch.length >= 14) this.flushImu();
  }

  private flushImu(): void {
    if (!this.imuBatch.length) return;
    const a = Float64Array.from(this.imuBatch);
    this.imuBatch.length = 0;
    this.post({ type: 'imu', samples: a }, [a.buffer]);
  }

  private onWorker(m: FromTracker): void {
    switch (m.type) {
      case 'result': return this.onResultMsg(m);
      case 'calibration':
        if (!this.replayLog) this.saveCalibration(m.td, m.axisSigns);
        if (m.axisSigns.some((v) => v < 0)) {
          this.imu.scaleAxes(m.axisSigns);
          for (let k = 0; k < 3; k++) this.axisSigns[k] *= m.axisSigns[k];
        }
        return;
      case 'focal':
        this.saveFocal(m.f);
        return;
      case 'error':
        console.warn('fastar:', m.message);
        return;
    }
  }

  private onResultMsg(m: ResultMessage): void {
    if (m.buffer) this.pool.push(m.buffer);
    // Frame results only: a timing check would misfire where performance.now()
    // is coarse (WebKit rounds to 1 ms, so a fast frame reports total 0).
    if (!m.acquisition) {
      this.busy = false;
      const now = performance.now();
      this.fpsWindow.push(now);
      while (this.fpsWindow.length && now - this.fpsWindow[0] > 1000) this.fpsWindow.shift();
      this.stats.fps = this.fpsWindow.length;
      this.stats.latency = m.doneAt - m.t;
      this.stats.timings = m.timings;
      this.stats.targets = m.targets;
      if (m.t > this.lastResultTime) this.lastResultTime = m.t;
    }
    this.stats.td = m.td;
    this.stats.tdCalibrated = m.tdCalibrated;
    for (const s of m.snapshots) {
      let p = this.predictors.get(s.index);
      if (!p) this.predictors.set(s.index, (p = new PosePredictor()));
      p.ingest(s.snap);
    }
    for (const r of m.targets) this.confidence.set(r.index, r.confidence);
    for (const i of m.lost) {
      this.predictors.delete(i);
      this.confidence.set(i, 0);
    }
    // Show/hide hysteresis on filter confidence.
    for (const [i, c] of this.confidence) {
      if (!this.shown.has(i) && c >= this.o.showThreshold && this.predictors.has(i)) {
        this.shown.add(i);
        this.listeners.found.forEach((cb) => cb(i));
      } else if (this.shown.has(i) && (c < this.o.hideThreshold || !this.predictors.has(i))) {
        this.shown.delete(i);
        this.smoothers.delete(i);
        this.listeners.lost.forEach((cb) => cb(i));
      }
    }
    this.onResult?.(m);
    if (this.syncDisplay) this.flushDisplay();
  }

  private scheduleFrame(): void {
    const v = this.video;
    if (!this.running || !v) return;
    if (typeof (v as { requestVideoFrameCallback?: unknown }).requestVideoFrameCallback === 'function') {
      this.rvfcHandle = v.requestVideoFrameCallback((now, meta) => this.onVideoFrame(now, meta));
    } else {
      // Fallback: rAF without exact frame identity.
      requestAnimationFrame((now) => this.onVideoFrame(now, {
        presentationTime: now, expectedDisplayTime: now, width: v.videoWidth, height: v.videoHeight,
        mediaTime: v.currentTime, presentedFrames: ++this.frameCount, processingDuration: 0,
      } as VideoFrameCallbackMetadata));
    }
  }

  private onVideoFrame(_now: number, meta: VideoFrameCallbackMetadata): void {
    if (!this.running) return;
    let captureTime = this.captureTimeOf(meta);
    if (this.replayLog) {
      // Recorded media time 0 is the first recorded frame; map onto the playback clock.
      if (Number.isNaN(this.replayShift)) this.replayShift = performance.now() - (this.replayLog.startTime + meta.mediaTime * 1000);
      captureTime = this.replayLog.startTime + meta.mediaTime * 1000 + this.replayShift;
    }
    if (meta.captureTime !== undefined && this.stats.path === 'track-processor' && meta.presentedFrames % 10 === 0) {
      this.post({ type: 'clock', mediaTime: meta.mediaTime, captureTime: meta.captureTime });
    }
    let sent = false;
    if (this.stats.path === 'copy' && !this.busy && this.grabber && this.pool.length) {
      const buf = this.pool.pop()!;
      if (this.grabber.grab(this.video!, new Uint8Array(buf))) {
        this.busy = sent = true;
        // The filter applies a frame only once gyro data covers its capture time.
        this.flushImu();
        this.post({ type: 'frame', id: meta.presentedFrames, t: captureTime, luma: buf }, [buf]);
      } else {
        this.pool.push(buf);
      }
    }
    this.recorder.onVideoFrame(captureTime, meta, this.video!);
    if (this.syncDisplay) {
      this.queueDisplay(captureTime, meta, this.stats.path === 'track-processor' || sent);
    } else {
      // Pose for exactly this presented frame (no prediction to "now").
      this.emitFrame(captureTime, meta);
    }
    this.scheduleFrame();
  }

  /**
   * Capture time (ms) of a presented frame: rVFC captureTime where it is a
   * real capture time (Chrome). WebKit reports captureTime equal to
   * expectedDisplayTime (and mediaTime 0 for camera streams), i.e. a display
   * time quantized to the refresh, so there the frames are put back on their
   * regular grid instead (the camera-to-IMU offset absorbs the constant).
   * Without captureTime, the display time minus a typical latency is used.
   */
  private captureTimeOf(meta: VideoFrameCallbackMetadata): number {
    const cap = meta.captureTime;
    if (cap !== undefined && Math.abs(cap - meta.expectedDisplayTime) > 0.5) return cap;
    this.frameClock ??= new RegularClock(1000 / 30, 0.6);
    return this.frameClock.map(cap ?? meta.expectedDisplayTime - ASSUMED_LATENCY);
  }

  private emitFrame(captureTime: number, meta: VideoFrameCallbackMetadata, image?: HTMLCanvasElement | OffscreenCanvas): void {
    const poses = new Map<number, Pose>();
    for (const i of this.shown) {
      if (!this.poseAt(i, captureTime, this.R, this.t)) continue;
      if (this.o.smoothing !== false) {
        let sm = this.smoothers.get(i);
        if (!sm) this.smoothers.set(i, (sm = new PoseSmoother(this.o.smoothing)));
        // Same filter state at the previous displayed frame: the difference is pure camera motion.
        const tp = sm.previousTime;
        const ok = Number.isFinite(tp) && this.poseAt(i, tp * 1000, this.Rp, this.tp);
        sm.update(captureTime / 1000, this.R, this.t, ok ? this.Rp : null, ok ? this.tp : null);
      }
      poses.set(i, {
        matrix: poseToGl(this.R, this.t),
        R: this.R.slice(), t: this.t.slice(),
        confidence: this.confidence.get(i) ?? 0,
      });
    }
    const ev: FrameEvent = {
      presentedFrame: {
        captureTime, expectedDisplayTime: meta.expectedDisplayTime, mediaTime: meta.mediaTime,
        presentedFrames: meta.presentedFrames, width: meta.width, height: meta.height,
      },
      poses,
      image,
    };
    this.frameListeners.forEach((cb) => cb(ev));
  }

  /** syncDisplay: copy the presented frame and hold it until it is tracked. */
  private queueDisplay(captureTime: number, meta: VideoFrameCallbackMetadata, expect: boolean): void {
    const v = this.video!;
    // Full queue: show the oldest now rather than drop frames silently.
    while (this.displayQueue.length >= DISPLAY_QUEUE - 1) {
      const e = this.displayQueue.shift()!;
      this.emitFrame(e.captureTime, e.meta, e.image);
    }
    const image = this.nextDisplayImage(v.videoWidth, v.videoHeight);
    const ctx = image.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
    if (!ctx) return this.emitFrame(captureTime, meta);
    ctx.drawImage(v, 0, 0, image.width, image.height);
    this.displayQueue.push({ captureTime, meta, image, expect, queuedAt: performance.now() });
    this.flushDisplay();
  }

  private nextDisplayImage(w: number, h: number): HTMLCanvasElement | OffscreenCanvas {
    const k = this.displayImageNext;
    this.displayImageNext = (k + 1) % DISPLAY_QUEUE;
    let c = this.displayImages[k];
    if (!c) {
      c = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
      this.displayImages[k] = c;
    }
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    return c;
  }

  /**
   * Show the newest queued frame that is ready: tracked (a result at or after
   * its capture time), not sent for tracking with nothing earlier pending, or
   * waiting too long. Older frames are skipped; only one can be on screen.
   */
  private flushDisplay(): void {
    const q = this.displayQueue;
    const now = performance.now();
    let pick = -1;
    for (let i = 0; i < q.length; i++) {
      const e = q[i];
      if (e.captureTime <= this.lastResultTime + 4 || now - e.queuedAt > DISPLAY_MAX_DELAY || !e.expect) pick = i;
      else break;
    }
    if (pick < 0) return;
    const e = q[pick];
    q.splice(0, pick + 1);
    this.emitFrame(e.captureTime, e.meta, e.image);
  }

  /** Feed recorded IMU samples on the playback clock. */
  private startReplayImu(log: SessionLog): void {
    let i = 0;
    this.replayTimer = setInterval(() => {
      if (Number.isNaN(this.replayShift)) return;
      const now = performance.now();
      while (i < log.imu.length && log.imu[i][0] + this.replayShift <= now) {
        const s = log.imu[i++];
        this.onImu({ t: s[0] + this.replayShift, gx: s[1], gy: s[2], gz: s[3], ax: s[4], ay: s[5], az: s[6] });
      }
    }, 5) as unknown as number;
  }

  // --- persisted per-device calibration ---

  private deviceKey(): string {
    return `${navigator.userAgent}|${this.video?.videoWidth}x${this.video?.videoHeight}`;
  }

  private loadIntrinsics(w: number, h: number, hfov?: number): Intrinsics {
    if (hfov) return intrinsicsFromFov(w, h, hfov);
    const f = this.storedFocal();
    if (f) return intrinsicsFromFocal(w, h, f * Math.max(w, h));
    return intrinsicsFromFov(w, h, lookupHfov(navigator.userAgent));
  }

  private storedFocal(): number | null {
    try {
      const v = localStorage.getItem(`${STORAGE}focal:${this.deviceKey()}`);
      return v ? parseFloat(v) : null;
    } catch {
      return null;
    }
  }

  private saveFocal(fPx: number): void {
    const ratio = fPx / Math.max(this.width, this.height);
    try {
      localStorage.setItem(`${STORAGE}focal:${this.deviceKey()}`, String(ratio));
    } catch { /* storage unavailable */ }
    this.intrinsics = intrinsicsFromFocal(this.width, this.height, fPx);
    this.post({ type: 'intrinsics', K: this.intrinsics });
  }

  private loadCalibration(): { td: number; axisSigns: [number, number, number] } | null {
    try {
      const v = localStorage.getItem(`${IMU_CAL_KEY}${navigator.userAgent}`);
      return v ? JSON.parse(v) : null;
    } catch {
      return null;
    }
  }

  private saveCalibration(td: number, axisSigns: [number, number, number]): void {
    try {
      const prev = this.loadCalibration();
      const signs = prev ? prev.axisSigns.map((s, k) => s * axisSigns[k]) : axisSigns;
      localStorage.setItem(`${IMU_CAL_KEY}${navigator.userAgent}`, JSON.stringify({ td, axisSigns: signs }));
    } catch { /* storage unavailable */ }
  }
}
