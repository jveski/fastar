// JSON shapes shared by the browser harness and the Playwright assertions.
export interface PoseSample {
  index: number;
  R: number[];
  t: number[];
  confidence: number;
  videoTime: number | null;
  locked: boolean;
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
  td: number;
  tdCalibrated: boolean;
  targets: Array<{ index: number; status: string; confidence: number; inliers: number; rms: number }>;
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
}

export interface HarnessResult {
  done: boolean;
  error: string | null;
  mode: string;
  scenario: string;
  startedAt: number;
  endedAt: number;
  frames: FrameSample[];
  events: { found: number[]; lost: number[] };
  stats: StatsSample | null;
  diagnostics: Diagnostics;
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
