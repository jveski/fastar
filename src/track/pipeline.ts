import { Pyramid } from '../image/pyramid';
import type { Intrinsics } from '../vision/camera';
import type { TargetData } from '../target/format';
import { Flips, TargetTracker, type TrackerOptions, type VideoWindow } from './tracker';
import { VideoClock, type ClockOptions, type ClockWindow } from '../fusion/video-clock';
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
  clock?: Partial<ClockOptions>;
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
  /** Video targets: estimated video time of this frame (s), its std, and whether it is locked. */
  videoTime?: number;
  videoTimeStd?: number;
  videoLocked?: boolean;
}

/** Video clock state for the render side: video time = (camera time - offset) mod duration. */
export interface ClockState {
  index: number;
  offset: number;
  std: number;
  duration: number;
  locked: boolean;
}

/** A video's time is lost when its credible window is wider than this (s). */
const TIME_LOST = 1.5;
/** Seconds between probe sweeps over the whole clip. */
const SWEEP_INTERVAL = 0.5;
/** Seconds between acquisition checks of a tracked video's time. */
const RANGE_CHECK = 1;
// Probe budget: candidate frames per frame, and NCC evaluations per frame.
const MAX_CANDIDATES = 96;
const PROBE_EVALS = 720;
/** Joint time and pose search: candidate times, and how far around the estimate it looks (frames). */
const SEARCH_CANDIDATES = 16;
const SEARCH_REACH = 15;
/** Below this share of the tracker's points, the time is probed further around its estimate. */
const WEAK_SUPPORT = 0.6;
/** Log-likelihood per agreeing point in the search's evidence for the clock. */
const SEARCH_LLR = 0.15;

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
  private lastTime = 0;
  private readonly rangeTime = new Map<number, number>();
  private readonly sweepTime = new WeakMap<VideoClock, number>();
  private sweepPhase = 0;
  /** Video clocks by target index; they outlive tracking losses (the video keeps playing). */
  readonly clocks = new Map<number, VideoClock>();
  private readonly candFrames = new Int32Array(MAX_CANDIDATES);
  private readonly searchFrames = new Int32Array(SEARCH_CANDIDATES);
  private readonly searchInliers = new Float64Array(SEARCH_CANDIDATES);
  /** Best candidate of the last searchTime: its frame and pose. */
  private searchBest = 0;
  private readonly Rs = new Float64Array(9);
  private readonly ts = new Float64Array(3);
  private llr = new Float64Array(0);
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

  /**
   * Indices currently tracked (not coasting), excluded from acquisition.
   * Video targets whose time is badly uncertain (e.g. after a jump in the
   * camera clock) stay in: an acquisition tells which part of the clip shows.
   */
  trackedIndices(): number[] {
    return this.slots.filter((s) => s.status === 'tracking' && !this.timeLost(s.index)).map((s) => s.index);
  }

  /**
   * The video's time needs an acquisition: its window is wide, or the last
   * check is over a second old. Probes only look near the current estimate,
   * so a periodic, estimate-free check is what pulls a wrong lock back.
   */
  private timeLost(index: number): boolean {
    const c = this.clocks.get(index);
    if (!c) return false;
    const w = c.window(this.lastTime);
    return w.hi - w.lo > TIME_LOST || this.lastTime - (this.rangeTime.get(index) ?? -Infinity) > RANGE_CHECK;
  }

  processFrame(luma: Uint8Array, tCam: number): FrameResult {
    const t0 = now();
    this.lastTime = tCam;
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
      const clock = this.clocks.get(s.index);
      const had = s.tracker.n;
      if (clock) this.setVideoTime(s.tracker, clock, tCam);
      let ok: boolean;
      // A video cut retires most tracked points: re-seed from the new shot's.
      const thin = clock && (s.tracker.n < s.tracker.o.minInliers + 4 || s.tracker.n < 0.4 * had);
      if (s.status === 'tracking' && prevK >= 0 && !thin) {
        const before = s.tracker.n;
        ok = s.tracker.track(this.ring[prevK], cur, this.R, this.t);
        // Video: most of the points still expected on screen failing at once
        // means the picture changed under them (a seek, or a cut the clock
        // did not expect); the few survivors are not to be trusted.
        if (ok && clock && s.tracker.inliers < 0.3 * before && s.tracker.inliers < 20) ok = s.tracker.ok = false;
      } else {
        // Coasting: try to re-lock locally around the gyro-propagated pose.
        ok = s.tracker.init(cur, this.R, this.t, thin ? s.tracker.numTrackPoints : 80, 16);
      }
      // Video: the tracked points fit, but hardly any other point can be
      // found where this time and pose put it. Search both again.
      let restarted = false;
      if (ok && clock && s.tracker.starved) {
        const inl = this.searchTime(clock, s.tracker, cur, tCam, s.tracker.R, s.tracker.t, Flips.Careful);
        if (inl >= 1.5 * s.tracker.inliers) {
          this.setVideoTime(s.tracker, clock, tCam);
          ok = s.tracker.init(cur, this.Rs, this.ts);
          restarted = ok;
        }
      }
      if (ok && clock) this.probeClock(clock, s.tracker, cur, tCam, id);
      const tc = now();
      trackT += tc - tb;
      klt += s.tracker.timings.klt; ncc += s.tracker.timings.ncc; solve += s.tracker.timings.solve;
      if (ok && imu && this.tdCalibrated && s === this.slots[0]) this.rollingShutter.observe(s.tracker);
      if (imu && !this.tdCalibrated && this.o.calibrateTd && s === this.slots[0]) {
        if (ok) this.tdEstimator.addVision(tCam, s.tracker.R);
        else this.tdEstimator.gap();
      }
      if (ok && restarted) {
        this.restartFilter(s.eskf, tCam, s.tracker);
        s.lastSeen = tCam;
      } else if (ok) {
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
    const timeLost = this.slots.some((s) => this.timeLost(s.index));
    const wantAcquire = (tracking < this.o.maxTrack || timeLost) && (tracking === 0 || id - this.lastAcqFrame >= this.o.acquireInterval);
    const t2 = now();
    return {
      frameId: id,
      time: tCam,
      targets: this.slots.map((s) => this.targetInfo(s, tCam)),
      found,
      lost,
      wantAcquire,
      timings: { pyramid: t1 - t0, track: trackT, fusion: fusionT, total: t2 - t0, klt, ncc, solve },
    };
  }

  private targetInfo(s: TargetSlot, tCam: number): FrameTargetInfo {
    const info: FrameTargetInfo = {
      index: s.index,
      status: s.status,
      inliers: s.tracker.inliers,
      rms: s.tracker.rms,
      nis: s.nis,
      confidence: this.confidence(s, tCam),
    };
    const clock = this.clocks.get(s.index);
    if (clock) {
      const w = clock.window(tCam);
      info.videoTime = w.tv;
      info.videoTimeStd = w.std;
      info.videoLocked = w.locked;
    }
    return info;
  }

  /** Video clock states for every video target seen so far. */
  clockStates(): ClockState[] {
    const out: ClockState[] = [];
    for (const [index, c] of this.clocks) {
      const md = c.window(0);
      out.push({ index, offset: c.offset(), std: md.std, duration: c.duration, locked: md.locked });
    }
    return out;
  }

  /**
   * Point positions for the video time at camera time t. When the time is
   * uncertain across a change of content (a cut, or the loop seam with an
   * unknown pause) few points are on screen for the whole window: then bet
   * on the most likely time; if that is wrong, the frame fails to track and
   * the next one tries again.
   */
  private setVideoTime(tr: TargetTracker, clock: VideoClock, t: number): void {
    const w = toFrameWindow(clock.window(t), clock.fps);
    tr.setTime(w);
    if (!w.locked && tr.usableCount() < 3 * tr.o.minInliers) tr.setTime({ x: w.x, lo: w.x, hi: w.x, locked: true });
  }

  /**
   * Probe the video time at the tracked pose: every frame around the current
   * estimate (a seek or stall shows up at once), and every half second over
   * the whole clip, so that a wrong lock cannot hold itself up.
   */
  private probeClock(clock: VideoClock, tr: TargetTracker, frame: Pyramid, tCam: number, id: number): void {
    const last = this.sweepTime.get(clock) ?? -Infinity;
    const wide = tCam - last > SWEEP_INTERVAL || tCam < last;
    if (wide) this.sweepTime.set(clock, tCam);
    // While few points agree with the pose, the time may be off by more than
    // the clock thinks: look further around it.
    const margin = tr.n > 0 && tr.n < WEAK_SUPPORT * tr.o.maxPoints ? SEARCH_REACH : 2;
    const n = wide ? clock.sweep(this.candFrames, MAX_CANDIDATES, this.sweepPhase++) : clock.candidates(tCam, this.candFrames, MAX_CANDIDATES, margin, id);
    const probes = wide ? 24 : Math.max(4, Math.min(24, Math.floor(PROBE_EVALS / n)));
    const llr = this.llrFor(clock);
    const used = tr.probeTime(frame.levels[0], this.candFrames, n, llr, probes);
    if (used > 0) {
      fillGaps(llr, this.candFrames, n, wide);
      clock.addFrameLikelihood(tCam, llr);
    }
    else clock.advance(tCam);
  }

  /** Start the filter over at the tracker's pose at time t. */
  private restartFilter(eskf: Eskf, t: number, tracker: TargetTracker): void {
    const imu = this.o.useImu ? this.imu : null;
    eskf.initialize(t, tracker.R, tracker.t, tracker.cov, this.o.useImu ? this.upVector() : null, this.td, this.bg);
    if (this.tdCalibrated) eskf.enableTd(this.td);
    if (imu) eskf.propagateTo(Math.min(imu.latestTime, t + eskf.td + 0.2), imu);
  }

  /**
   * Video: search the time and the pose together around the clock's
   * estimate (and over its whole credible window), starting from pose
   * (R, t). Content that moves as one makes a wrong time and a matching
   * wrong pose fit a few points; only the right pair fits most of them.
   * Feeds the clock with how well each time did. Returns the number of
   * points agreeing with the best (its frame in this.searchBest, its pose
   * in this.Rs, this.ts), or 0.
   */
  private searchTime(clock: VideoClock, tr: TargetTracker, frame: Pyramid, tCam: number, R: ArrayLike<number>, t: ArrayLike<number>, flips: Flips): number {
    const w = clock.window(tCam), fps = clock.fps, N = Math.round(clock.duration * fps);
    const fc = Math.floor(w.tv * fps);
    const f0 = Math.min(Math.floor(w.lo * fps), fc - SEARCH_REACH), f1 = Math.max(Math.floor(w.hi * fps), fc + SEARCH_REACH);
    const span = Math.min(f1 - f0, N - 1);
    const stride = Math.max(1, Math.ceil((span + 1) / SEARCH_CANDIDATES));
    const frames = this.searchFrames, inl = this.searchInliers;
    let n = 0;
    for (let f = f0; f <= f0 + span && n < SEARCH_CANDIDATES; f += stride) frames[n++] = f;
    const best = tr.searchTime(frame, R, t, frames, n, inl, this.Rs, this.ts, flips);
    if (best < 0) return 0;
    this.searchBest = frames[best];
    // Evidence: each frame scores like the nearest candidate; frames the
    // search did not reach like an average one.
    const llr = this.llrFor(clock);
    let mean = 0;
    for (let i = 0; i < n; i++) mean += inl[i] / n;
    llr.fill(SEARCH_LLR * mean);
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < stride; d++) {
        const f = frames[i] - (stride >> 1) + d;
        llr[((f % N) + N) % N] = SEARCH_LLR * inl[i];
      }
    }
    clock.addFrameLikelihood(tCam, llr);
    return inl[best];
  }

  private llrFor(clock: VideoClock): Float64Array {
    const N = Math.round(clock.duration * clock.fps);
    if (this.llr.length !== N) this.llr = new Float64Array(N);
    return this.llr;
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
    const k = this.ringSlot(frameId);
    if (k < 0) return null;
    const tFrame = this.ringTime[k];
    if (res.time) this.rangeTime.set(res.target, tFrame);
    if (s && s.status === 'tracking') {
      // Already tracked: only the timing is news.
      if (res.time) this.clocks.get(res.target)?.addRanges(tFrame, res.time.ranges);
      return null;
    }
    const tracker = s?.tracker ?? new TargetTracker(this.targets[res.target], this.K, this.o.tracker);
    let clock = this.clocks.get(res.target);
    let R0: ArrayLike<number> = res.R, t0: ArrayLike<number> = res.t, searched = false;
    if (tracker.video) {
      const vm = tracker.video;
      if (!clock) this.clocks.set(res.target, (clock = new VideoClock(vm.duration, vm.fps, this.o.clock)));
      if (res.time) clock.addRanges(tFrame, res.time.ranges);
      // The acquired pose fits the matched keyframe's content, which may have
      // moved on since: search the time and pose together, seed there, then
      // narrow the time down with probes.
      tracker.reset();
      searched = this.searchTime(clock, tracker, this.ring[k], tFrame, res.R, res.t, Flips.Free) > 0;
      if (searched) {
        // Seed at the time and pose found; the probes then have tracked
        // points to fit each candidate time with.
        const x = this.searchBest + 0.5;
        tracker.setTime({ x, lo: x, hi: x, locked: true });
        R0 = this.Rs.slice();
        t0 = this.ts.slice();
      } else {
        tracker.R.set(res.R);
        tracker.t.set(res.t);
        this.probeClock(clock, tracker, this.ring[k], tFrame, 0);
        this.setVideoTime(tracker, clock, tFrame);
      }
    }
    if (!tracker.init(this.ring[k], R0, t0)) return null;
    if (clock && searched) this.probeClock(clock, tracker, this.ring[k], tFrame, 0);
    const eskf = s?.eskf ?? new Eskf({ ...this.o.eskf, useImu: this.o.useImu });
    const imu = this.o.useImu ? this.imu : null;
    if (!s && this.slots.length === 0) this.tdEstimator.gap();
    this.restartFilter(eskf, tFrame, tracker);
    const latest = this.frameId;
    if (frameId !== latest) {
      const lk = this.ringSlot(latest);
      const tL = this.ringTime[lk];
      eskf.poseAt(tL, this.R, this.t, imu);
      if (clock) this.setVideoTime(tracker, clock, tL);
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

/**
 * Strided candidates leave frames in between without evidence, which would
 * count against them next to a candidate that matched (the true frame can
 * fall between two): give them the values interpolated from the candidates
 * either side (around the loop too when the candidates span the clip).
 */
function fillGaps(llr: Float64Array, frames: Int32Array, n: number, loop: boolean): void {
  const N = llr.length;
  const at = (f: number) => llr[((f % N) + N) % N];
  const gaps = loop && n > 1 ? n : n - 1;
  for (let i = 0; i < gaps; i++) {
    const a = frames[i], b = i + 1 < n ? frames[i + 1] : frames[0] + N;
    if (b - a < 2) continue;
    const va = at(a), vb = at(b);
    for (let f = a + 1; f < b; f++) llr[((f % N) + N) % N] = va + ((vb - va) * (f - a)) / (b - a);
  }
}

function toFrameWindow(w: ClockWindow, fps: number): VideoWindow {
  return { x: w.tv * fps, lo: w.lo * fps, hi: w.hi * fps, locked: w.locked };
}

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
