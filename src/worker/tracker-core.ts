// Tracking worker logic, independent of the Worker global so it can be
// driven directly in tests.
import { decodeTargets, type TargetData } from '../target/format';
import { Pipeline } from '../track/pipeline';
import { TargetTracker } from '../track/tracker';
import type { AcquireResult } from '../track/acquire';
import { videoFrameToLuma } from './luma';
import type {
  AcquireToTracker, FromTracker, ResultMessage, ToTracker, TrackerInit, TrackerToAcquire,
} from './protocol';
import { FocalCalibrator } from '../vision/focal';

export interface PortLike {
  postMessage(msg: unknown, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent) => void) | null;
}

type Post = (msg: FromTracker, transfer?: Transferable[]) => void;

/** Matching time per acquisition pass (ms): a few targets on a phone, all of a handful on a laptop. */
export const DEFAULT_ACQUIRE_BUDGET = 20;
/** Most of the time acquisition may take while something is tracked (the rest is left to tracking and the page). */
export const DEFAULT_ACQUIRE_DUTY = 0.5;
/** Window over which the acquisition worker's busy share is measured (ms). */
const ACQUIRE_WINDOW = 2000;
/** Seconds a near miss keeps its target first in line for acquisition. */
const NEAR_KEEP = 2;

export class TrackerCore {
  pipeline: Pipeline | null = null;
  targets: TargetData[] = [];
  private acq: PortLike | null = null;
  private acqBusy = false;
  private lastAcqMs = 0;
  private width = 0;
  private height = 0;
  /** worker clock + offset = main clock (ms). */
  private clockOffset = 0;
  private reportedCalibration = false;
  private luma: Uint8Array | null = null;
  // MSTP timestamp mapping (VideoFrame.timestamp us -> main ms).
  private mediaOffset = NaN;
  private arrivalOffset = Infinity;
  private focal: FocalCalibrator | null = null;
  private acquireBudgetMs = DEFAULT_ACQUIRE_BUDGET;
  private acquireDuty = DEFAULT_ACQUIRE_DUTY;
  /** When the last acquisition finished (worker ms), and recent ones: [finished at, ms]. */
  private acqDoneAt = -Infinity;
  private readonly acqLog: Array<[number, number]> = [];
  /** Stream path: capture time of the last frame processed (main ms), and the time to get its pixels. */
  private lastStreamT = -Infinity;
  private prepMs = 0;
  /** Near misses from acquisition: target -> camera time (s). */
  private readonly near = new Map<number, number>();

  constructor(private readonly post: Post, private readonly now: () => number = () => performance.now()) {}

  handle(msg: ToTracker): void {
    try {
      switch (msg.type) {
        case 'init': return this.init(msg);
        case 'imu': return this.onImu(msg.samples);
        case 'frame': {
          // Time the frame waited for this worker (busy with other messages) counts against its budget like reading pixels does.
          const wait = msg.sentAt !== undefined ? Math.max(0, this.now() + this.clockOffset - msg.sentAt) : 0;
          return this.onFrame(new Uint8Array(msg.luma), msg.t, msg.luma, wait);
        }
        case 'stream': void this.readStream(msg.readable); return;
        case 'clock': return this.onClock(msg.mediaTime, msg.captureTime);
        case 'intrinsics':
          this.pipeline?.setIntrinsics(msg.K);
          this.acq?.postMessage({ type: 'intrinsics', K: msg.K } satisfies TrackerToAcquire);
          return;
        case 'priority':
          this.pipeline?.setPriority(msg.index, msg.weight);
          return;
        case 'cadence':
          this.pipeline?.setFrameInterval(msg.interval / 1000);
          return;
        case 'throttle':
          if (msg.acquireDuty > 0) this.acquireDuty = Math.min(1, Math.max(0.02, msg.acquireDuty));
          return;
      }
    } catch (e) {
      this.post({ type: 'error', message: (e as Error).message ?? String(e) });
    }
  }

  private init(m: TrackerInit): void {
    this.targets = decodeTargets(m.targets);
    this.width = m.width;
    this.height = m.height;
    this.clockOffset = (typeof performance !== 'undefined' ? performance.timeOrigin : 0) - m.mainTimeOrigin;
    const p = new Pipeline(this.targets, m.K, {
      width: m.width, height: m.height, maxTrack: m.maxTrack, useImu: m.useImu, eskf: m.eskf, tracker: m.tracker,
      deformable: m.deformable, surface: m.surface, budget: m.budget,
    });
    if (m.acquireBudgetMs !== undefined) this.acquireBudgetMs = m.acquireBudgetMs;
    if (m.axisSigns) p.axisSigns.set(m.axisSigns);
    if (m.td !== undefined) p.setTd(m.td, !!m.tdCalibrated);
    this.reportedCalibration = !!m.tdCalibrated;
    this.pipeline = p;
    this.luma = new Uint8Array(m.width * m.height);
    if (m.calibrateFocal) this.focal = new FocalCalibrator(m.width, m.height);
    this.acq = m.acquirePort as unknown as PortLike;
    this.acq.onmessage = (e: MessageEvent) => this.onAcquired(e.data as AcquireToTracker);
    this.post({ type: 'ready' });
  }

  /** samples: [t(ms), gx, gy, gz, ax, ay, az] * n, main clock. */
  private onImu(s: Float64Array): void {
    const p = this.pipeline;
    if (!p) return;
    for (let i = 0; i + 6 < s.length; i += 7) p.pushImu(s[i] / 1000, s[i + 1], s[i + 2], s[i + 3], s[i + 4], s[i + 5], s[i + 6]);
  }

  /** Main-thread rVFC pairs give the exact VideoFrame.timestamp -> capture time mapping. */
  private onClock(mediaTime: number, captureTime: number): void {
    const off = captureTime - mediaTime * 1000;
    this.mediaOffset = Number.isNaN(this.mediaOffset) ? off : 0.9 * this.mediaOffset + 0.1 * off;
  }

  /** Process one luma frame captured at main-clock time t (ms); prepMs: time spent getting its pixels. */
  onFrame(luma: Uint8Array, t: number, transferBack?: ArrayBuffer, prepMs = 0): void {
    const p = this.pipeline;
    if (!p) return;
    this.prepMs = prepMs;
    const res = p.processFrame(luma, t / 1000, prepMs);
    const tracking = p.slots.filter((s) => s.status === 'tracking').length;
    // While something is tracked, acquisition rests in proportion to how
    // long it takes, so it gets at most acquireDuty of the time: on a phone
    // with two fast cores it otherwise competes with tracking and the page.
    const rested = tracking === 0 || this.now() - this.acqDoneAt >= this.lastAcqMs * (1 / this.acquireDuty - 1);
    if (res.wantAcquire && !this.acqBusy && this.acq && rested) {
      const copy = luma.slice().buffer;
      this.acqBusy = true;
      p.markAcquire(res.frameId);
      const up = p.o.useImu ? p.upVector() : null;
      this.acq.postMessage({
        type: 'acquire', frameId: res.frameId, luma: copy, exclude: p.trackedIndices(), up: up ? Array.from(up) : null,
        maxResults: Math.max(1, p.o.maxTrack - tracking), priority: this.acquirePriority(res.time), budgetMs: this.acquireBudgetMs,
        occupied: p.trackedQuads(),
      } satisfies TrackerToAcquire, [copy]);
    }
    if (this.focal) this.collectFocal();
    const msg = this.buildResult(res.frameId, t, res.targets, res.found, res.lost, { ...res.timings, acquire: this.lastAcqMs });
    const surfaces = p.surfaces();
    if (surfaces.length) {
      msg.surfaces = surfaces.map((s) => ({ index: s.index, cols: s.cols, rows: s.rows, vertices: Float32Array.from(s.vertices) }));
    }
    if (transferBack) msg.buffer = transferBack;
    this.post(msg, [
      ...msg.snapshots.map((s) => s.snap.buffer as ArrayBuffer),
      ...(msg.surfaces ?? []).map((s) => s.vertices.buffer as ArrayBuffer),
      ...(transferBack ? [transferBack] : []),
    ]);
    if (p.tdCalibrated && !this.reportedCalibration && p.tdResult) {
      this.reportedCalibration = true;
      this.post({
        type: 'calibration', td: p.td, axisSigns: p.tdResult.axisSigns, correlation: p.tdResult.correlation,
      });
    }
  }

  private buildResult(
    frameId: number, t: number, targets: ResultMessage['targets'], found: number[], lost: number[], timings: ResultMessage['timings'],
  ): ResultMessage {
    const p = this.pipeline!;
    return {
      type: 'result', frameId, t, doneAt: this.now() + this.clockOffset, targets, found, lost, timings, budget: p.budgetMs,
      load: {
        stride: p.stride, overload: p.strideControl.overload, interval: p.frameInterval * 1000,
        acquireBusy: this.acquireBusy(), prep: this.prepMs,
      },
      snapshots: p.slots.map((s) => ({ index: s.index, snap: s.eskf.exportSnapshot(s.eskf.t - 0.4) })),
      td: p.td, tdCalibrated: p.tdCalibrated, clocks: p.clockStates(),
    };
  }

  /** Share of the last ACQUIRE_WINDOW ms the acquisition worker spent working. */
  private acquireBusy(): number {
    const now = this.now(), log = this.acqLog;
    while (log.length && now - log[0][0] > ACQUIRE_WINDOW) log.shift();
    let ms = 0;
    for (const [, d] of log) ms += d;
    return Math.min(1, ms / ACQUIRE_WINDOW);
  }

  /** Targets to try first when acquiring: recent near misses, then recently lost ones. */
  private acquirePriority(tCam: number): number[] {
    const out: number[] = [];
    for (const [i, t] of this.near) {
      if (tCam - t > NEAR_KEEP) this.near.delete(i);
      else out.push(i);
    }
    for (const i of this.pipeline!.recentlyLost()) if (!out.includes(i)) out.push(i);
    return out;
  }

  private onAcquired(m: AcquireToTracker): void {
    this.acqBusy = false;
    this.lastAcqMs = m.ms;
    this.acqDoneAt = this.now();
    this.acqLog.push([this.acqDoneAt, m.ms]);
    const p = this.pipeline;
    if (!p) return;
    const tFrame = p.frameTime(m.frameId);
    for (const i of m.near) this.near.set(i, Number.isNaN(tFrame) ? p.frameTime(p.latestFrameId) : tFrame);
    const found: number[] = [];
    for (const r of m.results) {
      const res: AcquireResult = { ...r, timings: { features: 0, match: 0, total: m.ms } };
      const f = p.applyAcquisition(res, m.frameId);
      if (f === null) continue;
      found.push(f);
      this.near.delete(f);
      if (!p.o.deformable?.includes(r.target)) this.focal?.addHomography(r.H, this.targets[r.target]);
    }
    const near = m.near.filter((i) => !p.slot(i));
    if (!found.length && !near.length) return;
    // Report immediately so rendering (or loading content) can start before the next frame.
    const lid = p.latestFrameId;
    const targets = found.map((i) => {
      const slot = p.slot(i)!;
      return { index: i, status: slot.status, confidence: 1, inliers: slot.tracker.inliers, rms: slot.tracker.rms, nis: NaN };
    });
    const msg = this.buildResult(lid, p.frameTime(lid) * 1000, targets, [], [], { pyramid: 0, track: 0, fusion: 0, total: 0, klt: 0, ncc: 0, solve: 0, acquire: m.ms });
    msg.acquisition = true;
    if (near.length) msg.near = near;
    this.post(msg, msg.snapshots.map((s) => s.snap.buffer as ArrayBuffer));
  }

  private collectFocal(): void {
    const p = this.pipeline!;
    for (const s of p.slots) {
      // Bending targets are not planar: their homographies say nothing reliable about the focal length.
      if (s.status === 'tracking' && s.tracker instanceof TargetTracker) this.focal!.addTracked(s.tracker, this.targets[s.index]);
    }
    const f = this.focal!.estimate();
    if (f) {
      this.post({ type: 'focal', f: f.f, samples: f.samples });
      this.focal = null; // one-shot per session; the main thread persists it
    }
  }

  /** Chrome: frames straight from MediaStreamTrackProcessor (maxBufferSize 1). */
  private async readStream(readable: ReadableStream): Promise<void> {
    const reader = readable.getReader() as ReadableStreamDefaultReader<VideoFrame>;
    for (;;) {
      const { value: frame, done } = await reader.read();
      if (done || !frame) break;
      try {
        const arrival = this.now() + this.clockOffset;
        const tsMs = frame.timestamp / 1000;
        this.arrivalOffset = Math.min(this.arrivalOffset, arrival - tsMs);
        const t = frameCaptureTime(tsMs, arrival, this.mediaOffset, this.arrivalOffset);
        // Tracking every n-th frame: skip the others before reading their pixels.
        const p = this.pipeline;
        if (p && p.stride > 1 && t - this.lastStreamT < (p.stride - 0.5) * p.frameInterval * 1000) {
          frame.close();
          continue;
        }
        const t0 = this.now();
        const luma = await videoFrameToLuma(frame, this.width, this.height, this.luma!);
        frame.close();
        this.prepMs = this.now() - t0;
        this.lastStreamT = t;
        if (luma) this.onFrame(luma, t, undefined, this.prepMs);
      } catch (e) {
        frame.close();
        this.post({ type: 'error', message: (e as Error).message });
      }
    }
    reader.releaseLock();
  }
}

/**
 * Main-clock capture time (ms) of a VideoFrame with timestamp `tsMs`: from
 * the rVFC mediaTime -> captureTime mapping when it is plausible (captured
 * before it arrived, and less than a second before), else from the lower
 * envelope of arrival times. Some cameras (Chrome's fake device, for one)
 * stamp VideoFrames on a different base than rVFC's mediaTime.
 */
export function frameCaptureTime(tsMs: number, arrival: number, mediaOffset: number, arrivalOffset: number): number {
  const viaMedia = tsMs + mediaOffset;
  if (viaMedia === viaMedia && viaMedia <= arrival + 5 && arrival - viaMedia < 1000) return viaMedia;
  return tsMs + arrivalOffset;
}
