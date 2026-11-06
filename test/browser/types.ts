// JSON shapes shared by the browser harness and the Playwright assertions.

/** Texture coordinates [u, v] at which bending targets' surfaces are sampled and checked. */
export const SURFACE_GRID: Array<[number, number]> = [];
for (let j = 0; j < 4; j++) for (let i = 0; i < 5; i++) SURFACE_GRID.push([0.1 + 0.2 * i, 0.1 + (0.8 * j) / 3]);

export interface PoseSample {
  index: number;
  R: number[];
  t: number[];
  confidence: number;
  videoTime: number | null;
  locked: boolean;
  /** Bending targets: pose.surface at SURFACE_GRID, [x, y, z] each (target axes of pose.matrix). */
  surface?: number[];
}

export interface FrameSample {
  captureTime: number;
  mediaTime: number;
  presentedFrames: number;
  hasImage: boolean;
  poses: PoseSample[];
  /** Video element size in CSS pixels. */
  view: [number, number];
  projection: number[];
}

export interface StatsSample {
  path: string;
  sensors: string;
  motionPermission: string;
  imuSamples: number;
  fps: number;
  latency: number;
  trackerReady: boolean;
  acquireReady: boolean;
  td: number;
  tdCalibrated: boolean;
  budgetMs: number;
  targets: Array<{ index: number; status: string; confidence: number; inliers: number; rms: number; priority?: number; points?: number; every?: number }>;
}

export interface Diagnostics {
  userAgent: string;
  devicePixelRatio: number;
  screen: [number, number];
  requestVideoFrameCallback: boolean;
  mediaStreamTrackProcessor: boolean;
  webgl2: boolean;
  deviceMotionEvent: boolean;
  needsMotionPermission: boolean;
  secureContext: boolean;
  processing?: [number, number];
  videoSize?: [number, number];
  worker?: string;
}

export interface HarnessResult {
  done: boolean;
  error: string | null;
  mode: string;
  scenario: string;
  startedAt: number;
  endedAt: number;
  frames: FrameSample[];
  /** Target event times (performance.now()), and the indices found in order. */
  events: { found: number[]; lost: number[]; near?: number[]; foundIndex?: number[] };
  stats: StatsSample | null;
  diagnostics: Diagnostics;
  /** Console errors/warnings, which is where worker failures surface. */
  logs: string[];
  /** A sampled video frame at the end of the run, for black-frame debugging. */
  probe: {
    videoTime: number;
    paused: boolean;
    readyState: number;
    videoWidth: number;
    videoHeight: number;
    meanLuma: number;
    offscreenLuma: number;
    error: string | null;
  } | null;
  /** Present for mode=record: the result of replaying the recorded session. */
  replay: { frames: FrameSample[]; stats: StatsSample | null; error: string | null } | null;
}

export interface TruthFrame {
  t: number;
  mediaTime: number;
  R: number[];
  /** Target-in-camera translation (m). */
  tv: number[];
  visible: boolean;
  videoTime: number | null;
  /** Bent sheets: SURFACE_GRID points in camera coordinates (OpenCV, m), [x, y, z] each. */
  surface?: number[];
  /** Several targets: each one's pose, and how much of it is in view (0 to 1). R and tv above are target 0's. */
  targets?: Array<{ R: number[]; tv: number[]; inView: number }>;
}

export interface Truth {
  name: string;
  fps: number;
  /** Camera frame size. */
  width: number;
  height: number;
  /** Target image size, used for its physical height. */
  targetWidth: number;
  targetHeight: number;
  posterWidthM: number;
  intrinsics: { fx: number; fy: number; cx: number; cy: number; width: number; height: number };
  targetFile: string;
  duration: number;
  video: { duration: number; start: number } | null;
  frames: TruthFrame[];
}
