import { Pyramid } from '../image/pyramid';
import type { Intrinsics } from '../vision/camera';
import type { TargetData } from '../target/format';
import { TargetTracker, type TrackerOptions } from './tracker';
import { Eskf, type EskfOptions } from '../fusion/eskf';
import { ImuBuffer } from '../fusion/imu';
import type { AcquireResult } from './acquire';
import { TdEstimator, type TdResult } from '../fusion/td';
import { rateAt } from '../fusion/predict';
import { mat3, mat3FromQuat, mat3TVec } from '../math/rot';
import { RollingShutterEstimator } from './rolling-shutter';

export interface PipelineOptions {
  width: number;
  height: number;
  maxTrack: number;
  useImu: boolean;
  /** Frames kept so late acquisition results can be applied. */
  ringSize: number;
  /** Min frames between acquisition runs while something is tracked. */
  acquireInterval: number;
  /** Max time without vision before the target is hidden (s). */
  coastTimeout: number;
  /** Hide earlier if the relative position std exceeds this (m). */
  maxPosStd: number;
  /** Estimate the camera-to-IMU offset (and axis signs) from the first seconds of motion. */
  calibrateTd: boolean;
  tracker?: Partial<TrackerOptions>;
  eskf?: Partial<EskfOptions>;
}

export const DEFAULT_PIPELINE: Omit<PipelineOptions, 'width' | 'height'> = {
  maxTrack: 1,
  useImu: false,
  ringSize: 6,
  acquireInterval: 3,
  coastTimeout: 0.7,
  maxPosStd: 0.15,
  calibrateTd: true,
};

export type TargetStatus = 'tracking' | 'coasting';

export interface TargetSlot {
  index: number;
  tracker: TargetTracker;
  eskf: Eskf;
  status: TargetStatus;
  /** Camera time of the last successful vision update. */
  lastSeen: number;
  nis: number;
}

export interface FrameTargetInfo {
  index: number;
  status: TargetStatus;
  inliers: number;
  rms: number;
  nis: number;
  confidence: number;
}

export interface FrameTimings {
  pyramid: number;
  track: number;
  fusion: number;
  total: number;
  klt: number;
  ncc: number;
  solve: number;
}

export interface FrameResult {
  frameId: number;
  time: number;
  targets: FrameTargetInfo[];
  found: number[];
  lost: number[];
  wantAcquire: boolean;
  timings: FrameTimings;
}

export class Pipeline {
  readonly o: PipelineOptions;
  readonly imu = new ImuBuffer(512);
  readonly slots: TargetSlot[] = [];
  private readonly ring: Pyramid[];
  private readonly ringId: Int32Array;
  private readonly ringTime: Float64Array;
  private frameId = 0;
  private lastAcqFrame = -1000;
  /** Shared calibration carried across target slots. */
  readonly bg = new Float64Array(3);
  td = 0;
  private readonly R = new Float64Array(9);
  private readonly t = new Float64Array(3);
  private readonly cov6 = new Float64Array(36);
  private readonly up = new Float64Array(3);
  private readonly Rwc = mat3();
  private pendingFound: number[] = [];
  readonly tdEstimator = new TdEstimator();
  /** True once t_d comes from the estimator (or a persisted calibration). */
  tdCalibrated = false;
  tdResult: TdResult | null = null;
  /** Gyro/accel axis sign corrections found by the startup check. */
  readonly axisSigns = Float64Array.of(1, 1, 1);
  private tdCheckFrame = 0;
  /** Rolling-shutter readout, estimated once t_d is known (the gyro must line up with the frames). */
  readonly rollingShutter = new RollingShutterEstimator();

  constructor(readonly targets: TargetData[], public K: Intrinsics, opts: Partial<PipelineOptions> & { width: number; height: number }) {
    this.o = { ...DEFAULT_PIPELINE, ...opts };
    this.ring = Array.from({ length: this.o.ringSize }, () => new Pyramid(opts.width, opts.height, 3));
    this.ringId = new Int32Array(this.o.ringSize).fill(-1);
    this.ringTime = new Float64Array(this.o.ringSize);
  }

  setIntrinsics(K: Intrinsics): void {
    this.K = K;
    for (const s of this.slots) s.tracker.K = K;
  }

  pushImu(t: number, gx: number, gy: number, gz: number, ax = NaN, ay = NaN, az = NaN): void {
    const sg = this.axisSigns;
    this.imu.push(t, gx * sg[0], gy * sg[1], gz * sg[2], ax * sg[0], ay * sg[1], az * sg[2]);
  }

  /** Use a known offset (e.g. persisted per device) and let the filters refine it. */
  setTd(td: number, calibrated = true): void {
    this.td = td;
    this.tdCalibrated = calibrated;
    for (const s of this.slots) {
      if (calibrated) s.eskf.enableTd(td);
      else s.eskf.td = td;
    }
  }

  private applyTdResult(r: TdResult): void {
    this.tdResult = r;
    if (r.axisSigns.some((v) => v < 0)) {
      for (let k = 0; k < 3; k++) this.axisSigns[k] *= r.axisSigns[k];
      this.imu.scaleAxes(r.axisSigns);
      this.bg.fill(0);
    }
    // Restart the filters from the latest vision pose with the new offset
    // (and corrected axes): the trajectory so far was integrated against the
    // wrong time base.
    for (const s of this.slots) {
      const bg = r.axisSigns.some((v) => v < 0) ? undefined : s.eskf.bg.slice();
      s.eskf.initialize(s.lastSeen, s.tracker.R, s.tracker.t, s.tracker.cov, this.upVector(), r.td, bg);
      s.eskf.propagateTo(this.imu.latestTime, this.imu);
    }
    this.setTd(r.td, true);
  }

  /** Unit up-vector in camera axes from recent accelerometer samples, or null. */
  upVector(): Float64Array | null {
    const imu = this.imu;
    let sx = 0, sy = 0, sz = 0, n = 0;
    for (let i = imu.count - 1; i >= 0 && n < 10; i--) {
      const s = imu.slot(i);
      if (!imu.hasAccel[s]) continue;
      sx += imu.a[3 * s]; sy += imu.a[3 * s + 1]; sz += imu.a[3 * s + 2];
      n++;
    }
    const m = Math.hypot(sx, sy, sz);
    if (n === 0 || m === 0) return null;
    this.up[0] = sx / m; this.up[1] = sy / m; this.up[2] = sz / m;
    return this.up;
  }

  get latestFrameId(): number {
    return this.frameId;
  }

  frameTime(id: number): number {
    const k = this.ringSlot(id);
    return k < 0 ? NaN : this.ringTime[k];
  }

  private ringSlot(id: number): number {
    const k = id % this.o.ringSize;
    return this.ringId[k] === id ? k : -1;
  }

  /** Indices currently tracked (not coasting), excluded from acquisition. */
  trackedIndices(): number[] {
    return this.slots.filter((s) => s.status === 'tracking').map((s) => s.index);
  }

  processFrame(luma: Uint8Array, tCam: number): FrameResult {
    const t0 = now();
    const id = ++this.frameId;
    const k = id % this.o.ringSize;
    const prevK = this.ringSlot(id - 1);
    this.ring[k].build(luma);
    this.ringId[k] = id;
    this.ringTime[k] = tCam;
    const t1 = now();
    const cur = this.ring[k];
    const found = this.pendingFound, lost: number[] = [];
    this.pendingFound = [];
    const imu = this.o.useImu ? this.imu : null;
    let trackT = 0, fusionT = 0, klt = 0, ncc = 0, solve = 0;
    for (let si = this.slots.length - 1; si >= 0; si--) {
      const s = this.slots[si];
      const ta = now();
      s.eskf.flushPending(imu);
      // Bring the filter up to date: newest gyro sample, or the frame time without an IMU.
      s.eskf.propagateTo(imu ? imu.latestTime : tCam + s.eskf.td, imu);
      if (!s.eskf.poseAt(tCam, this.R, this.t, imu)) { this.R.set(s.tracker.R); this.t.set(s.tracker.t); }
      if (imu && imu.count > 0) {
        const w = rateAt(imu, tCam + s.eskf.td, s.tracker.omega), b = s.eskf.bg;
        w[0] -= b[0]; w[1] -= b[1]; w[2] -= b[2];
        const Rwc = mat3FromQuat(s.eskf.qwc, this.Rwc), vw = s.eskf.vwc;
        mat3TVec(Rwc, vw[0], vw[1], vw[2], s.tracker.velocity);
        const rs = this.rollingShutter, use = this.tdCalibrated && rs.ready;
        s.tracker.rsX = use ? rs.rsX : 0;
        s.tracker.rsY = use ? rs.rsY : 0;
      }
      const tb = now();
      let ok: boolean;
      if (s.status === 'tracking' && prevK >= 0) {
        ok = s.tracker.track(this.ring[prevK], cur, this.R, this.t);
      } else {
        // Coasting: try to re-lock locally around the gyro-propagated pose.
        ok = s.tracker.init(cur, this.R, this.t, 80, 16);
      }
      const tc = now();
      trackT += tc - tb;
      klt += s.tracker.timings.klt; ncc += s.tracker.timings.ncc; solve += s.tracker.timings.solve;
      if (ok && imu && this.tdCalibrated && s === this.slots[0]) this.rollingShutter.observe(s.tracker);
      if (imu && !this.tdCalibrated && this.o.calibrateTd && s === this.slots[0]) {
        if (ok) this.tdEstimator.addVision(tCam, s.tracker.R);
        else this.tdEstimator.gap();
      }
      if (ok) {
        const r = s.eskf.updatePose(tCam, s.tracker.R, s.tracker.t, s.tracker.cov, imu);
        s.nis = r.nis;
        if (s.status === 'coasting') s.status = 'tracking';
        s.lastSeen = tCam;
      } else if (s.status === 'tracking') {
        s.status = 'coasting';
      }
      if (s.status === 'coasting' && !this.isVisible(s, tCam)) {
        this.slots.splice(si, 1);
        lost.push(s.index);
      }
      fusionT += (tb - ta) + (now() - tc);
      this.bg.set(s.eskf.bg);
      this.td = s.eskf.td;
    }
    if (imu && !this.tdCalibrated && this.o.calibrateTd && ++this.tdCheckFrame % 10 === 0) {
      const r = this.tdEstimator.estimate(this.imu, this.bg);
      if (r) this.applyTdResult(r);
    }
    const tracking = this.slots.filter((s) => s.status === 'tracking').length;
    const wantAcquire = tracking < this.o.maxTrack && (tracking === 0 || id - this.lastAcqFrame >= this.o.acquireInterval);
    const t2 = now();
    return {
      frameId: id,
      time: tCam,
      targets: this.slots.map((s) => ({
        index: s.index,
        status: s.status,
        inliers: s.tracker.inliers,
        rms: s.tracker.rms,
        nis: s.nis,
        confidence: this.confidence(s, tCam),
      })),
      found,
      lost,
      wantAcquire,
      timings: { pyramid: t1 - t0, track: trackT, fusion: fusionT, total: t2 - t0, klt, ncc, solve },
    };
  }

  /** Record that acquisition was started on frame `id` (for rate limiting). */
  markAcquire(id: number): void {
    this.lastAcqFrame = id;
  }

  /**
   * Apply an acquisition result computed on frame `frameId`. Seeds the
   * tracker on that frame and, if newer frames exist, catches up to the
   * latest frame with gyro-seeded KLT. Returns the target index if newly
   * found (also reported in the next frame result's `found`).
   */
  applyAcquisition(res: AcquireResult, frameId: number): number | null {
    let s = this.slots.find((x) => x.index === res.target);
    if (s && s.status === 'tracking') return null;
    const k = this.ringSlot(frameId);
    if (k < 0) return null;
    const tFrame = this.ringTime[k];
    const tracker = s?.tracker ?? new TargetTracker(this.targets[res.target], this.K, this.o.tracker);
    if (!tracker.init(this.ring[k], res.R, res.t)) return null;
    const eskf = s?.eskf ?? new Eskf({ ...this.o.eskf, useImu: this.o.useImu });
    const imu = this.o.useImu ? this.imu : null;
    eskf.initialize(tFrame, tracker.R, tracker.t, tracker.cov, this.o.useImu ? this.upVector() : null, this.td, this.bg);
    if (this.tdCalibrated) eskf.enableTd(this.td);
    if (!s && this.slots.length === 0) this.tdEstimator.gap();
    if (imu) eskf.propagateTo(Math.min(imu.latestTime, tFrame + eskf.td + 0.2), imu);
    const latest = this.frameId;
    if (frameId !== latest) {
      const lk = this.ringSlot(latest);
      const tL = this.ringTime[lk];
      eskf.poseAt(tL, this.R, this.t, imu);
      if (!tracker.track(this.ring[k], this.ring[lk], this.R, this.t)) return null;
      eskf.updatePose(tL, tracker.R, tracker.t, tracker.cov, imu);
    }
    const isNew = !s;
    if (!s) {
      s = { index: res.target, tracker, eskf, status: 'tracking', lastSeen: 0, nis: NaN };
      this.slots.push(s);
      this.pendingFound.push(res.target);
    }
    s.status = 'tracking';
    s.lastSeen = this.ringTime[this.ringSlot(latest)];
    return isNew ? res.target : null;
  }

  isVisible(s: TargetSlot, tCam: number): boolean {
    if (s.status === 'tracking') return true;
    if (tCam - s.lastSeen > this.o.coastTimeout) return false;
    s.eskf.relativeCovariance(this.cov6);
    const posStd = Math.sqrt(this.cov6[21] + this.cov6[28] + this.cov6[35]);
    return posStd < this.o.maxPosStd;
  }

  private confidence(s: TargetSlot, tCam: number): number {
    s.eskf.relativeCovariance(this.cov6);
    const posStd = Math.sqrt(this.cov6[21] + this.cov6[28] + this.cov6[35]);
    const cov = Math.max(0, 1 - posStd / this.o.maxPosStd);
    const age = s.status === 'tracking' ? 1 : Math.max(0, 1 - (tCam - s.lastSeen) / this.o.coastTimeout);
    const inl = s.status === 'tracking' ? Math.min(1, s.tracker.inliers / 30) : 1;
    return cov * age * inl;
  }

  /** Target pose in the camera at camera time tCam (for rendering a presented frame). */
  poseAt(index: number, tCam: number, R: Float64Array, t: Float64Array): boolean {
    const s = this.slots.find((x) => x.index === index);
    return !!s && s.eskf.poseAt(tCam, R, t, this.o.useImu ? this.imu : null);
  }

  slot(index: number): TargetSlot | undefined {
    return this.slots.find((x) => x.index === index);
  }
}

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
