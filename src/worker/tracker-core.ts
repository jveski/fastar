// Tracking worker logic, independent of the Worker global so it can be
// driven directly in tests.
import { decodeTargets, type TargetData } from '../target/format';
import { Pipeline } from '../track/pipeline';
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

  constructor(private readonly post: Post, private readonly now: () => number = () => performance.now()) {}

  handle(msg: ToTracker): void {
    try {
      switch (msg.type) {
        case 'init': return this.init(msg);
        case 'imu': return this.onImu(msg.samples);
        case 'frame': return this.onFrame(new Uint8Array(msg.luma), msg.t, msg.luma);
        case 'stream': void this.readStream(msg.readable); return;
        case 'clock': return this.onClock(msg.mediaTime, msg.captureTime);
        case 'intrinsics':
          this.pipeline?.setIntrinsics(msg.K);
          this.acq?.postMessage({ type: 'intrinsics', K: msg.K } satisfies TrackerToAcquire);
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
    });
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

  /** Process one luma frame captured at main-clock time t (ms). */
  onFrame(luma: Uint8Array, t: number, transferBack?: ArrayBuffer): void {
    const p = this.pipeline;
    if (!p) return;
    const res = p.processFrame(luma, t / 1000);
    if (res.wantAcquire && !this.acqBusy && this.acq) {
      const copy = luma.slice().buffer;
      this.acqBusy = true;
      p.markAcquire(res.frameId);
      const up = p.o.useImu ? p.upVector() : null;
      this.acq.postMessage({
        type: 'acquire', frameId: res.frameId, luma: copy, exclude: p.trackedIndices(), up: up ? Array.from(up) : null,
      } satisfies TrackerToAcquire, [copy]);
    }
    if (this.focal) this.collectFocal();
    const msg = this.buildResult(res.frameId, t, res.targets, res.found, res.lost, { ...res.timings, acquire: this.lastAcqMs });
    if (transferBack) msg.buffer = transferBack;
    this.post(msg, [...msg.snapshots.map((s) => s.snap.buffer as ArrayBuffer), ...(transferBack ? [transferBack] : [])]);
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
      type: 'result', frameId, t, doneAt: this.now() + this.clockOffset, targets, found, lost, timings,
      snapshots: p.slots.map((s) => ({ index: s.index, snap: s.eskf.exportSnapshot(s.eskf.t - 0.4) })),
      td: p.td, tdCalibrated: p.tdCalibrated,
    };
  }

  private onAcquired(m: AcquireToTracker): void {
    this.acqBusy = false;
    this.lastAcqMs = m.ms;
    const p = this.pipeline;
    if (!p || !m.result) return;
    const r = m.result;
    const res: AcquireResult = { ...r, timings: { features: 0, match: 0, total: m.ms } };
    const found = p.applyAcquisition(res, m.frameId);
    if (found !== null) {
      this.focal?.addHomography(r.H, this.targets[r.target]);
      // Report immediately so rendering can start before the next frame.
      const lid = p.latestFrameId;
      const slot = p.slot(found)!;
      const msg = this.buildResult(lid, p.frameTime(lid) * 1000, [{
        index: found, status: slot.status, confidence: 1, inliers: slot.tracker.inliers, rms: slot.tracker.rms, nis: NaN,
      }], [], [], { pyramid: 0, track: 0, fusion: 0, total: 0, klt: 0, ncc: 0, solve: 0, acquire: m.ms });
      msg.acquisition = true;
      this.post(msg, msg.snapshots.map((s) => s.snap.buffer as ArrayBuffer));
    }
  }

  private collectFocal(): void {
    const p = this.pipeline!;
    for (const s of p.slots) {
      if (s.status === 'tracking') this.focal!.addTracked(s.tracker, this.targets[s.index]);
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
        const t = tsMs + (Number.isNaN(this.mediaOffset) ? this.arrivalOffset : this.mediaOffset);
        const luma = await videoFrameToLuma(frame, this.width, this.height, this.luma!);
        frame.close();
        if (luma) this.onFrame(luma, t);
      } catch (e) {
        frame.close();
        this.post({ type: 'error', message: (e as Error).message });
      }
    }
    reader.releaseLock();
  }
}
