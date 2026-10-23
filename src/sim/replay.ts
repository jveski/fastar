// Offline replay of recorded sessions (JSON log + raw luma frames) through
// the same pipeline as the browser, for regression tests in Node.
import type { SessionLog } from '../shell/recorder';
import type { TargetData } from '../target/format';
import { Acquirer } from '../track/acquire';
import { Pipeline, type PipelineOptions } from '../track/pipeline';
import { intrinsicsFromFov } from '../vision/camera';
import { RegularClock } from '../fusion/clock';
import { PoseSmoother, type SmoothingOptions } from '../fusion/smooth';

export interface ReplayFrame {
  t: number;
  status: 'tracking' | 'coasting' | 'none';
  visible: boolean;
  R: Float64Array | null;
  tvec: Float64Array | null;
  gyroRate: number;
  ms: number;
  acquired: boolean;
  inliers: number;
  nis: number;
}

export interface ReplaySummary {
  frames: ReplayFrame[];
  /** tracking -> not tracking transitions. */
  trackLosses: number;
  acquisitions: number;
  /** Translation std (mm) over visible frames where the device was still (|w| < 0.05 rad/s). */
  stillJitterMm: number;
  medianFrameMs: number;
  p95FrameMs: number;
  tdCalibrated: boolean;
  td: number;
}

/**
 * Replay a recording. With `smoothing`, poses go through the render-side
 * smoother as the page would draw them.
 */
export function replaySession(
  log: SessionLog, luma: Uint8Array, targets: TargetData[], opts: Partial<PipelineOptions> = {},
  smoothing: Partial<SmoothingOptions> | null = null,
): ReplaySummary {
  const { width, height } = log.device;
  const K = log.device.intrinsics ?? intrinsicsFromFov(width, height);
  const useImu = log.imu.length > 0;
  const p = new Pipeline(targets, K, { width, height, useImu, ...opts });
  if (opts.calibrateTd === false && log.device.tdCalibrated) p.setTd(log.device.td, true);
  const acq = new Acquirer(targets, width, height, K);
  const idx = log.lumaFrames ?? log.frames.map((_, i) => i);
  // Older recordings carry raw WebKit stamps; regularize them as the live path now does.
  if (log.device.sensors === 'devicemotion' && log.imu.length > 1) {
    const c = new RegularClock(1000 / 60, Infinity);
    log = { ...log, imu: log.imu.map((s) => [c.map(s[0]), ...s.slice(1)]) };
  }
  if (log.frames.length > 1 && log.frames.every((f) => f.t === f.expectedDisplayTime)) {
    const c = new RegularClock(1000 / 30, 0.6);
    log = { ...log, frames: log.frames.map((f) => ({ ...f, t: c.map(f.t) })) };
  }
  const size = width * height;
  let ii = 0;
  const frames: ReplayFrame[] = [];
  let losses = 0, acquisitions = 0, wasTracking = false;
  const R = new Float64Array(9), t = new Float64Array(3), Rp = new Float64Array(9), tpv = new Float64Array(3);
  const smoother = smoothing ? new PoseSmoother(smoothing) : null;
  for (let k = 0; k < idx.length && (k + 1) * size <= luma.length; k++) {
    const fr = log.frames[idx[k]];
    // IMU that would have arrived before this frame was processed (~20 ms later).
    while (ii < log.imu.length && log.imu[ii][0] <= fr.t + 20) {
      const s = log.imu[ii++];
      p.pushImu(s[0] / 1000, s[1], s[2], s[3], s[4], s[5], s[6]);
    }
    const img = luma.subarray(k * size, (k + 1) * size);
    const res = p.processFrame(img, fr.t / 1000);
    let acquired = false;
    if (res.wantAcquire) {
      p.markAcquire(res.frameId);
      const a = acq.acquire(img, { exclude: p.trackedIndices(), up: useImu ? p.upVector() : null });
      if (a && p.applyAcquisition(a, res.frameId) !== null) { acquisitions++; acquired = true; }
    }
    const s = p.slots[0];
    const tracking = s?.status === 'tracking';
    if (wasTracking && !tracking) losses++;
    wasTracking = tracking;
    const visible = !!s && p.isVisible(s, fr.t / 1000);
    const has = visible && p.poseAt(s!.index, fr.t / 1000, R, t);
    if (has && smoother) {
      const tp = smoother.previousTime;
      const ok = Number.isFinite(tp) && p.poseAt(s!.index, tp, Rp, tpv);
      smoother.update(fr.t / 1000, R, t, ok ? Rp : null, ok ? tpv : null);
    } else smoother?.reset();
    let w = 0;
    if (useImu && ii > 0) { const g = log.imu[ii - 1]; w = Math.hypot(g[1], g[2], g[3]); }
    frames.push({
      t: fr.t, status: s?.status ?? 'none', visible, R: has ? R.slice() : null, tvec: has ? t.slice() : null,
      gyroRate: w, ms: res.timings.total, acquired, inliers: s?.tracker.inliers ?? 0, nis: s?.nis ?? NaN,
    });
  }
  const still = frames.filter((f) => f.tvec && f.gyroRate < 0.05);
  let jitter = NaN;
  if (still.length > 10) {
    const m = [0, 1, 2].map((k) => still.reduce((a, f) => a + f.tvec![k], 0) / still.length);
    jitter = Math.sqrt(still.reduce((a, f) => a + [0, 1, 2].reduce((b, k) => b + (f.tvec![k] - m[k]) ** 2, 0), 0) / (still.length - 1)) * 1000;
  }
  const ms = frames.map((f) => f.ms).sort((a, b) => a - b);
  return {
    frames, trackLosses: losses, acquisitions, stillJitterMm: jitter,
    medianFrameMs: ms[ms.length >> 1] ?? NaN, p95FrameMs: ms[Math.floor(ms.length * 0.95)] ?? NaN,
    tdCalibrated: p.tdCalibrated, td: p.td,
  };
}
