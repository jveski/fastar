// Messages between the main thread, the tracking worker and the acquisition
// worker. Times are main-thread performance.now() milliseconds.
import type { Intrinsics } from '../vision/camera';
import type { ClockState, FrameTargetInfo, FrameTimings } from '../track/pipeline';
import type { EskfOptions } from '../fusion/eskf';
import type { TrackerOptions } from '../track/tracker';
import type { AcquireTime } from '../track/acquire';
import type { SurfaceOptions } from '../deform/surface-tracker';
import type { BudgetOptions } from '../track/budget';

export interface TrackerInit {
  type: 'init';
  targets: ArrayBuffer;
  width: number;
  height: number;
  K: Intrinsics;
  maxTrack: number;
  useImu: boolean;
  /** Persisted t_d (s) and whether it is calibrated. */
  td?: number;
  tdCalibrated?: boolean;
  axisSigns?: [number, number, number];
  mainTimeOrigin: number;
  acquirePort: MessagePort;
  eskf?: Partial<EskfOptions>;
  tracker?: Partial<TrackerOptions>;
  /** Collect homographies for focal-length self-calibration. */
  calibrateFocal?: boolean;
  /** Targets tracked as bending surfaces, and their mesh options. */
  deformable?: number[];
  surface?: Partial<SurfaceOptions>;
  /** Tracking time per frame and how it is shared among targets. */
  budget?: Partial<BudgetOptions>;
  /** Matching time per acquisition pass (ms, after feature extraction). */
  acquireBudgetMs?: number;
}

export interface AcquireInit {
  type: 'init-acquire';
  targets: ArrayBuffer;
  width: number;
  height: number;
  K: Intrinsics;
  port: MessagePort;
}

export type ToTracker =
  | TrackerInit
  | { type: 'frame'; id: number; t: number; luma: ArrayBuffer }
  | { type: 'stream'; readable: ReadableStream }
  | { type: 'imu'; samples: Float64Array }
  | { type: 'clock'; mediaTime: number; captureTime: number }
  | { type: 'intrinsics'; K: Intrinsics }
  | { type: 'priority'; index: number; weight: number };

export interface ResultMessage {
  type: 'result';
  frameId: number;
  /** Capture time of the processed frame (main ms). */
  t: number;
  /** Main-clock time when the result was produced (for latency stats). */
  doneAt: number;
  targets: FrameTargetInfo[];
  found: number[];
  lost: number[];
  /** Targets the last acquisition nearly found (likely in view soon). */
  near?: number[];
  timings: FrameTimings & { acquire?: number };
  /** Tracking time allowed per frame (ms). */
  budget: number;
  /** Filter snapshots for the render-side predictor (seconds, main clock). */
  snapshots: Array<{ index: number; snap: Float64Array }>;
  td: number;
  tdCalibrated: boolean;
  /** Video targets' clocks (render side computes the video time of any frame). */
  clocks: ClockState[];
  /**
   * Bending targets tracked in this frame: mesh vertices (cols x rows,
   * row-major from the top-left) in target metres, OpenCV axes (x right,
   * y down, z into the target), relative to the target's pose.
   */
  surfaces?: Array<{ index: number; cols: number; rows: number; vertices: Float32Array }>;
  /** Luma buffer handed back for reuse (copy path). */
  buffer?: ArrayBuffer;
  /** Extra report sent right after an acquisition, not the result of a submitted frame. */
  acquisition?: boolean;
}

export type FromTracker =
  | ResultMessage
  | { type: 'ready' }
  /** axisSigns: correction relative to the signs given at init. */
  | { type: 'calibration'; td: number; axisSigns: [number, number, number]; correlation: number }
  | { type: 'focal'; f: number; samples: number }
  | { type: 'error'; message: string };

export interface WireAcquireResult {
  target: number;
  R: Float64Array;
  t: Float64Array;
  H: Float64Array;
  inliers: number;
  cov: Float64Array;
  frame: Float32Array;
  tgt: Float32Array;
  time?: AcquireTime;
}

export type TrackerToAcquire =
  | {
    type: 'acquire'; frameId: number; luma: ArrayBuffer; exclude: number[]; up: number[] | null;
    /** Most targets to return; targets to try first; matching time (ms); frame quads already taken. */
    maxResults?: number; priority?: number[]; budgetMs?: number; occupied?: number[];
  }
  | { type: 'intrinsics'; K: Intrinsics };

export type AcquireToTracker = { type: 'acquired'; frameId: number; results: WireAcquireResult[]; near: number[]; ms: number };
