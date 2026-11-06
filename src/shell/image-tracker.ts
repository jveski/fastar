// Public main-thread API.
import workerSource from 'virtual:worker-source';
import { decodeTargets, type TargetMeta } from '../target/format';
import { glProjection, intrinsicsFromFocal, lookupHfov, intrinsicsFromFov, poseToGl, type FitMode, type Intrinsics } from '../vision/camera';
import { ImuBuffer } from '../fusion/imu';
import { PosePredictor } from '../fusion/predict';
import { RegularClock } from '../fusion/clock';
import { videoTimeFromOffset } from '../fusion/video-clock';
import { PoseSmoother, type SmoothingOptions } from '../fusion/smooth';
import { MotionSensors, type Facing, type ImuReading, type MotionPermission, type SensorBackend } from './sensors';
import { LumaGrabber, canUseTrackProcessor, createTrackProcessor, openCamera, processingSize } from './capture';
import type { FromTracker, ResultMessage, ToTracker, AcquireInit } from '../worker/protocol';
import type { EskfOptions } from '../fusion/eskf';
import type { TrackerOptions } from '../track/tracker';
import type { ClockState, FrameTargetInfo } from '../track/pipeline';
import { Recorder, type SessionLog } from './recorder';
import { Surface, SurfaceFeed } from './surface';
import { AccelSigns } from '../fusion/gravity';
import { ContentManager, type ContentOptions } from './content';
import { DisplayPacer, type PacedFrame } from './pacer';
import { Governor, deviceStartLevel, type PerformanceState } from './governor';

export interface CreateOptions {
  /** URL of a compiled targets file, or its bytes. */
  targets: string | URL | ArrayBuffer;
  /** Most targets tracked at once (default: all of them). */
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
  /**
   * Track targets that bend, like printed paper: each pose then carries the
   * target's shape (Pose.surface). true for every image target, or choose
   * targets and options. Video targets are always tracked as rigid.
   */
  deformable?: boolean | DeformableOptions;
  /**
   * Bound the time spent per camera frame. Every target in view is tracked,
   * but the less prominent ones (smaller, partly out of view, off centre)
   * with fewer points, or on fewer frames, when time is short.
   */
  budget?: TrackingBudget;
}

export interface TrackingBudget {
  /** Tracking time per frame (ms; default: `share` of the frame interval). */
  trackMs?: number;
  /** Share of the frame interval spent tracking (default 0.6). */
  share?: number;
  /** Time per pass looking for targets not yet tracked (ms, default 20; at least one target is always tried). */
  acquireMs?: number;
}

export interface DeformableOptions {
  /** Indices of the targets that bend (default: every image target). */
  targets?: number[];
  /**
   * Mesh vertices along each target's long side (default 9; cells stay
   * roughly square). More follows tighter bends but needs more texture.
   */
  grid?: number;
  /** Share of a small shape change eased over to the next frame (0 to 1, default 0.5; 0 = off). */
  smoothing?: number;
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
   * Show each camera frame with the pose measured on it rather than a
   * prediction, which removes most visible jitter: frames are emitted with
   * a copy of their pixels (`FrameEvent.image`) to draw as the background,
   * a steady delay after capture so that most have been tracked by then.
   * The delay follows how long tracking takes on this device, at most
   * `maxDelay` ms (default 80); a frame not tracked by then is shown on time
   * with the predicted pose. With no target in view, frames are shown as
   * they arrive. Draw `image` instead of showing the <video> element.
   */
  syncDisplay?: boolean | SyncDisplayOptions;
}

export interface SyncDisplayOptions {
  /** Most delay (ms) added to the display to wait for tracking results (default 80). */
  maxDelay?: number;
}

export interface Pose {
  /** Column-major 4x4 model-view matrix (GL camera, target y up, z out of the target). */
  matrix: Float32Array;
  /** OpenCV-convention target-in-camera rotation (row-major) and translation (m). */
  R: Float64Array;
  t: Float64Array;
  confidence: number;
  /** Video targets: where the video is (s, in [0, duration)) in this frame. */
  videoTime?: number;
  /** Standard deviation of videoTime (s). */
  videoTimeStd?: number;
  /** videoTime is good to a fraction of a video frame. */
  videoTimeLocked?: boolean;
  /** Bending targets: the target's shape in this frame, in the coordinates of `matrix`. */
  surface?: Surface;
  /**
   * Unit vector pointing down (along gravity) in the coordinates of
   * `matrix`, when motion sensors are available.
   */
  gravity?: Float32Array;
}

export interface VideoTime {
  /** Seconds into the clip, in [0, duration). */
  time: number;
  std: number;
  locked: boolean;
  duration: number;
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

/** A camera frame waiting to be shown (syncDisplay). */
interface DisplayFrame {
  meta: VideoFrameCallbackMetadata;
  image: HTMLCanvasElement | OffscreenCanvas;
}

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
  /** Both workers finished their init handshake. */
  trackerReady: boolean;
  acquireReady: boolean;
  td: number;
  tdCalibrated: boolean;
  /** Tracked targets, with their priority, points and update rate under the budget. */
  targets: FrameTargetInfo[];
  /** Tracking time allowed per frame (ms). */
  budgetMs: number;
}

type Listener = (index: number) => void;
type TargetEvent = 'found' | 'lost' | 'near';

/** A target nearly found is reported at most this often (ms). */
const NEAR_INTERVAL = 1000;

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
    path: 'none', sensors: 'none', motionPermission: 'not-requested', imuSamples: 0, fps: 0, latency: 0, timings: null,
    trackerReady: false, acquireReady: false, td: 0, tdCalibrated: false, targets: [], budgetMs: 0,
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
  private readonly clocks = new Map<number, ClockState>();
  private readonly shown = new Set<number>();
  private readonly listeners: Record<TargetEvent, Set<Listener>> = { found: new Set(), lost: new Set(), near: new Set() };
  private readonly nearAt = new Map<number, number>();
  private readonly priorities = new Map<number, number>();
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
  private pacer: DisplayPacer<DisplayFrame> | null = null;
  private readonly displayImages: Array<HTMLCanvasElement | OffscreenCanvas> = [];
  private displayImageNext = 0;
  // Load and throttling.
  private governor = new Governor();
  private readonly perfListeners = new Set<(p: PerformanceState) => void>();
  private rafHandle = 0;
  /** Camera frame interval (ms) from capture times, and what was last sent to the worker. */
  private camInterval = NaN;
  private sentInterval = NaN;
  private prevCapture = NaN;
  private prevPresented = -1;
  /** Copy path: camera frames per tracked frame, and the last frame sent. */
  private stride = 1;
  private lastSent = -Infinity;
  private acquireDuty = NaN;
  /** Targets tracked as bending surfaces, and their shapes. */
  private readonly deformable: number[];
  private readonly surfaces = new Map<number, SurfaceFeed>();
  private readonly up = new Float64Array(3);
  /** The accelerometer's sign convention, from its own readings (for Pose.gravity). */
  private readonly accelSigns = new AccelSigns();

  private constructor(buffer: ArrayBuffer, o: CreateOptions) {
    this.buffer = buffer;
    this.targets = decodeTargets(buffer.slice(0)).map((t) => t.meta);
    this.o = { maxTrack: Math.max(1, this.targets.length), showThreshold: 0.5, hideThreshold: 0.2, ...o };
    const d = o.deformable;
    const chosen = !d ? [] : d === true || !d.targets ? this.targets.map((_, i) => i) : d.targets;
    this.deformable = chosen.filter((i) => this.targets[i] && !this.targets[i].video);
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
    const sync = o.syncDisplay;
    this.pacer = sync
      ? new DisplayPacer<DisplayFrame>(typeof sync === 'object' && sync.maxDelay !== undefined ? { maxDelay: Math.max(0, sync.maxDelay) } : {})
      : null;
    this.governor = new Governor({
      devicePixelRatio: typeof devicePixelRatio === 'number' ? devicePixelRatio : 1,
      startLevel: deviceStartLevel(),
    });
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
    this.post({ type: 'throttle', acquireDuty: (this.acquireDuty = this.governor.acquireDuty) });
    const settings = this.stream?.getVideoTracks()[0]?.getSettings?.();
    if (settings?.frameRate && settings.frameRate > 0) this.setCameraInterval(1000 / settings.frameRate);
    this.scheduleFrame();
    if (typeof requestAnimationFrame === 'function') this.rafHandle = requestAnimationFrame(this.onAnimationFrame);
  }

  stop(): void {
    this.running = false;
    if (this.video && this.rvfcHandle && 'cancelVideoFrameCallback' in this.video) this.video.cancelVideoFrameCallback(this.rvfcHandle);
    if (this.rafHandle && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(this.rafHandle);
    this.rafHandle = 0;
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
    this.nearAt.clear();
    this.clocks.clear();
    this.surfaces.clear();
    this.accelSigns.reset();
    this.pacer?.clear();
    this.frameClock = null;
    this.camInterval = this.sentInterval = this.prevCapture = NaN;
    this.prevPresented = -1;
    this.stride = 1;
    this.lastSent = -Infinity;
  }

  /** CSS object-fit of the video, which the projection (and a drawn background) follows. */
  get fitMode(): FitMode {
    return this.fit;
  }

  /**
   * 'found' / 'lost': a target came into view or left it. 'near': a target
   * not yet tracked nearly matched (far away, blurred, coming into view);
   * it is likely found soon, so its content can start loading.
   * 'performance': the throttling level changed (see `performance`).
   */
  on(event: TargetEvent, cb: Listener): () => void;
  on(event: 'performance', cb: (p: PerformanceState) => void): () => void;
  on(event: TargetEvent | 'performance', cb: Listener | ((p: PerformanceState) => void)): () => void {
    if (event === 'performance') {
      const f = cb as (p: PerformanceState) => void;
      this.perfListeners.add(f);
      return () => this.perfListeners.delete(f);
    }
    const f = cb as Listener;
    this.listeners[event].add(f);
    return () => this.listeners[event].delete(f);
  }

  /**
   * How loaded this device is, and hints for the app's rendering: when it
   * cannot keep up, `level` rises and `pixelRatio` / `effects` suggest
   * cheaper rendering (apply them on the 'performance' event). Tracking
   * throttles itself; `bottleneck` says whether rendering or tracking is
   * what holds the frame rate back.
   */
  get performance(): PerformanceState {
    return this.governor.state;
  }

  /** Target `index` is in view (between its 'found' and 'lost'). */
  isShown(index: number): boolean {
    return this.shown.has(index);
  }

  /** Index of the target compiled with this name, or -1. */
  targetIndex(name: string): number {
    return this.targets.findIndex((t) => t.name === name);
  }

  /**
   * Weight target `index` in the tracking budget (default 1): 2 counts it
   * as twice as prominent as it looks, 0 keeps it at the fewest points.
   */
  setPriority(index: number, weight: number): void {
    this.priorities.set(index, weight);
    this.post({ type: 'priority', index, weight });
  }

  /**
   * Load each target's content when it is found (or nearly found), and
   * unload content that has been out of view a while when room is needed.
   * See ContentOptions.
   */
  content<T>(o: ContentOptions<T>): ContentManager<T> {
    return new ContentManager(this, o);
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

  /**
   * Video targets: where the video is in a frame captured at `captureTime`
   * (ms, e.g. FrameEvent.presentedFrame.captureTime), or null before the
   * video has been seen. Keeps running while the target is out of view.
   */
  videoTimeAt(index: number, captureTime: number): VideoTime | null {
    const c = this.clocks.get(index);
    if (!c) return null;
    return { time: videoTimeFromOffset(c.offset, c.duration, captureTime / 1000), std: c.std, locked: c.locked, duration: c.duration };
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
    // Worker errors are otherwise invisible: the tracker recovers, but no
    // acquisition ever completes and the UI simply never locks.
    this.acquirer.onerror = (e) => console.error('fastar acquire worker error', e.message);
    this.acquirer.onmessage = (e) => {
      const m = e.data as { type?: string; message?: string };
      if (m?.type === 'acquire-ready') this.stats.acquireReady = true;
      else if (m?.type === 'acquire-error') console.error('fastar acquire worker error', m.message);
    };
    const acqInit: AcquireInit = { type: 'init-acquire', targets: this.buffer.slice(0), width, height, K, port: ch.port2 };
    this.acquirer.postMessage(acqInit, [acqInit.targets, ch.port2]);
    // Replays calibrate from scratch: stored values belong to this device, not the recording's.
    const cal = this.replayLog ? null : this.loadCalibration();
    if (cal) this.axisSigns.set(cal.axisSigns);
    const init: ToTracker = {
      type: 'init', targets: this.buffer.slice(0), width, height, K, maxTrack: this.o.maxTrack, useImu,
      td: cal?.td, tdCalibrated: !!cal, axisSigns: cal?.axisSigns, mainTimeOrigin: performance.timeOrigin,
      acquirePort: ch.port1, eskf: this.o.filter, tracker: this.o.tracker, calibrateFocal: !this.replayLog && this.storedFocal() === null,
      deformable: this.deformable,
      surface: typeof this.o.deformable === 'object' && this.o.deformable.grid ? { grid: this.o.deformable.grid } : undefined,
      budget: this.o.budget ? { budgetMs: this.o.budget.trackMs, ...(this.o.budget.share !== undefined ? { share: this.o.budget.share } : {}) } : undefined,
      acquireBudgetMs: this.o.budget?.acquireMs,
    };
    this.tracker.postMessage(init, [init.targets, ch.port1]);
    this.tracker.onmessage = (e) => this.onWorker(e.data as FromTracker);
    this.tracker.onerror = (e) => console.error('fastar worker error', e.message);
    for (const [index, weight] of this.priorities) this.post({ type: 'priority', index, weight });
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
      case 'ready':
        this.stats.trackerReady = true;
        return;
      case 'result': return this.onResultMsg(m);
      case 'calibration':
        if (!this.replayLog) this.saveCalibration(m.td, m.axisSigns);
        if (m.axisSigns.some((v) => v < 0)) {
          this.imu.scaleAxes(m.axisSigns);
          for (let k = 0; k < 3; k++) this.axisSigns[k] *= m.axisSigns[k];
          this.accelSigns.reset();
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
      this.stats.budgetMs = m.budget;
      this.pacer?.result(m.t, now);
      if (m.load) {
        this.stride = m.load.stride;
        this.governor.setTrackingLoad(m.load);
      }
    }
    this.stats.td = m.td;
    this.stats.tdCalibrated = m.tdCalibrated;
    for (const s of m.snapshots) {
      let p = this.predictors.get(s.index);
      if (!p) this.predictors.set(s.index, (p = new PosePredictor()));
      p.ingest(s.snap);
    }
    for (const r of m.targets) this.confidence.set(r.index, r.confidence);
    for (const c of m.clocks ?? []) this.clocks.set(c.index, c);
    for (const s of m.surfaces ?? []) {
      let f = this.surfaces.get(s.index);
      if (!f || f.cols !== s.cols || f.rows !== s.rows) {
        const meta = this.targets[s.index], keep = typeof this.o.deformable === 'object' ? this.o.deformable.smoothing ?? 0.5 : 0.5;
        f = new SurfaceFeed(s.cols, s.rows, meta.physicalWidth, (meta.physicalWidth * meta.height) / meta.width, keep);
        this.surfaces.set(s.index, f);
      }
      f.push(m.t, s.vertices);
    }
    for (const i of m.lost) {
      this.predictors.delete(i);
      this.confidence.set(i, 0);
      this.surfaces.delete(i);
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
    const now = performance.now();
    for (const i of m.near ?? []) {
      if (this.shown.has(i) || now - (this.nearAt.get(i) ?? -Infinity) < NEAR_INTERVAL) continue;
      this.nearAt.set(i, now);
      this.listeners.near.forEach((cb) => cb(i));
    }
    if (this.pacer) this.pacer.active = this.shown.size > 0;
    this.onResult?.(m);
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
    const t0 = performance.now();
    let captureTime = this.captureTimeOf(meta);
    if (this.replayLog) {
      // Recorded media time 0 is the first recorded frame; map onto the playback clock.
      if (Number.isNaN(this.replayShift)) this.replayShift = performance.now() - (this.replayLog.startTime + meta.mediaTime * 1000);
      captureTime = this.replayLog.startTime + meta.mediaTime * 1000 + this.replayShift;
    }
    this.measureCadence(captureTime, meta.presentedFrames);
    if (meta.captureTime !== undefined && this.stats.path === 'track-processor' && meta.presentedFrames % 10 === 0) {
      this.post({ type: 'clock', mediaTime: meta.mediaTime, captureTime: meta.captureTime });
    }
    // Copy path: one frame in `stride` (the worker says how many it keeps up with), and only when it is free.
    if (this.stats.path === 'copy' && !this.busy && this.grabber && this.pool.length && meta.presentedFrames - this.lastSent >= this.stride) {
      const buf = this.pool.pop()!;
      if (this.grabber.grab(this.video!, new Uint8Array(buf))) {
        this.busy = true;
        this.lastSent = meta.presentedFrames;
        // The filter applies a frame only once gyro data covers its capture time.
        this.flushImu();
        this.post({ type: 'frame', id: meta.presentedFrames, t: captureTime, luma: buf, sentAt: performance.now() }, [buf]);
      } else {
        this.pool.push(buf);
      }
    }
    this.recorder.onVideoFrame(captureTime, meta, this.video!);
    if (this.pacer) {
      this.queueDisplay(captureTime, meta, t0);
      this.pace();
    } else {
      // Pose for exactly this presented frame (no prediction to "now").
      this.emitFrame(captureTime, meta);
    }
    this.governor.addMain(performance.now() - t0);
    this.scheduleFrame();
  }

  /** Every display frame: show the paced frame that is due, and keep the load measurements going. */
  private readonly onAnimationFrame = (now: number): void => {
    if (!this.running) return;
    this.rafHandle = requestAnimationFrame(this.onAnimationFrame);
    const t0 = performance.now();
    this.governor.animationFrame(now);
    if (this.pacer) {
      this.pace();
      this.governor.display.delay = this.pacer.extraDelay;
      this.governor.display.late = this.pacer.shown ? this.pacer.late / this.pacer.shown : 0;
    }
    const t1 = performance.now();
    this.governor.addMain(t1 - t0);
    if (this.governor.update(t1)) this.onPerformanceChange();
  };

  private onPerformanceChange(): void {
    const duty = this.governor.acquireDuty;
    if (duty !== this.acquireDuty) this.post({ type: 'throttle', acquireDuty: (this.acquireDuty = duty) });
    if (this.perfListeners.size) {
      const s = this.governor.state;
      this.perfListeners.forEach((cb) => cb(s));
    }
  }

  /** The camera's frame interval from the presented frames' capture times; the worker budgets by it. */
  private measureCadence(captureTime: number, presented: number): void {
    const n = presented - this.prevPresented, d = captureTime - this.prevCapture;
    this.prevPresented = presented;
    this.prevCapture = captureTime;
    if (!(n >= 1 && n <= 8) || !(d > 0)) return;
    const iv = d / n;
    if (!(iv > 4 && iv < 200)) return;
    this.setCameraInterval(Number.isNaN(this.camInterval) ? iv : this.camInterval + 0.05 * (iv - this.camInterval));
  }

  private setCameraInterval(iv: number): void {
    this.camInterval = iv;
    if (this.pacer) this.pacer.capacity = Math.min(12, Math.ceil(this.pacer.maxDelay / iv) + 3);
    if (Number.isNaN(this.sentInterval) || Math.abs(iv - this.sentInterval) > 0.02 * this.sentInterval) {
      this.sentInterval = iv;
      this.post({ type: 'cadence', interval: iv });
    }
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
      const pose: Pose = {
        matrix: poseToGl(this.R, this.t),
        R: this.R.slice(), t: this.t.slice(),
        confidence: this.confidence.get(i) ?? 0,
      };
      const surface = this.surfaces.get(i)?.at(captureTime);
      if (surface) pose.surface = surface;
      if (this.upInCamera()) {
        // Down in target axes (R^T applied to -up), then to the app's y up, z out of the target.
        const R = this.R, u = this.up;
        pose.gravity = Float32Array.of(
          -(R[0] * u[0] + R[3] * u[1] + R[6] * u[2]),
          R[1] * u[0] + R[4] * u[1] + R[7] * u[2],
          R[2] * u[0] + R[5] * u[1] + R[8] * u[2],
        );
      }
      const vt = this.videoTimeAt(i, captureTime);
      if (vt) {
        pose.videoTime = vt.time;
        pose.videoTimeStd = vt.std;
        pose.videoTimeLocked = vt.locked;
      }
      poses.set(i, pose);
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

  /** Unit up direction in camera axes from the latest accelerometer samples (into this.up), if any. */
  private upInCamera(): boolean {
    const imu = this.imu;
    let sx = 0, sy = 0, sz = 0, n = 0;
    for (let i = imu.count - 1; i >= 0 && n < 8; i--) {
      const s = imu.slot(i);
      if (!imu.hasAccel[s]) continue;
      sx += imu.a[3 * s]; sy += imu.a[3 * s + 1]; sz += imu.a[3 * s + 2];
      n++;
    }
    const m = Math.hypot(sx, sy, sz);
    if (!n || !(m > 0)) return false;
    this.accelSigns.observe(imu);
    const g = this.accelSigns.signs;
    this.up[0] = (g[0] * sx) / m; this.up[1] = (g[1] * sy) / m; this.up[2] = (g[2] * sz) / m;
    return true;
  }

  /** syncDisplay: copy the presented frame and hand it to the pacer. */
  private queueDisplay(captureTime: number, meta: VideoFrameCallbackMetadata, arrival: number): void {
    const v = this.video!;
    const image = this.nextDisplayImage(v.videoWidth, v.videoHeight);
    const ctx = image.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
    if (!ctx) return this.emitFrame(captureTime, meta);
    ctx.drawImage(v, 0, 0, image.width, image.height);
    this.pacer!.push({ captureTime, arrival, payload: { meta, image } });
  }

  /** Copies are reused round-robin: one more than the pacer holds, plus the one on screen. */
  private nextDisplayImage(w: number, h: number): HTMLCanvasElement | OffscreenCanvas {
    const n = this.pacer!.capacity + 2;
    const k = this.displayImageNext % n;
    this.displayImageNext = (k + 1) % n;
    let c = this.displayImages[k];
    if (!c) {
      c = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
      this.displayImages[k] = c;
    }
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    return c;
  }

  /** Show the frame that is due now, if any. */
  private pace(): void {
    const f: PacedFrame<DisplayFrame> | null = this.pacer!.next(performance.now());
    if (f) this.emitFrame(f.captureTime, f.payload.meta, f.payload.image);
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
