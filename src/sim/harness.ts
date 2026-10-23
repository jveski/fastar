// End-to-end simulation harness: renders a scenario, feeds frames (and IMU)
// through the pipeline with acquisition, and logs errors against ground truth.
import type { GrayImage } from '../image/gray';
import type { Intrinsics } from '../vision/camera';
import { compileTarget } from '../target/compiler';
import type { TargetData } from '../target/format';
import { Acquirer } from '../track/acquire';
import { Pipeline, type PipelineOptions } from '../track/pipeline';
import { createSceneRenderer, relativePose, simulateImu, type ImuSimOptions, type Scenario } from './scene';
import type { RenderOptions } from './renderer';
import { mat3MulAtB, so3Log, mat3, vec3, rotationAngle } from '../math/rot';
import { projectPoint } from '../vision/pose';

export interface RunOptions {
  K: Intrinsics;
  poster: GrayImage;
  posterWidthM: number;
  target?: TargetData;
  fps?: number;
  useImu?: boolean;
  /** True camera-to-IMU offset: t_imu = t_cam + td (s). */
  td?: number;
  imu?: ImuSimOptions;
  render?: RenderOptions;
  pipeline?: Partial<PipelineOptions>;
  /** Called once per frame after processing (e.g. to feed a t_d estimator). */
  onFrame?: (p: Pipeline, tCam: number, tTrue: number) => void;
  /** Initial t_d guess given to the pipeline. */
  tdGuess?: number;
}

export interface FrameLog {
  t: number;
  tCam: number;
  visible: boolean;
  status: string;
  overlayErrPx: number;
  transErr: number;
  rotErrDeg: number;
  estT: Float64Array;
  estR: Float64Array;
  inliers: number;
  acquired: boolean;
  trackMs: number;
}

export interface RunResult {
  frames: FrameLog[];
  acquisitions: number;
  acquireAttempts: number;
  firstLockTime: number;
  pipeline: Pipeline;
}

const targetCache = new WeakMap<GrayImage, Map<number, TargetData>>();
export function targetFor(poster: GrayImage, widthM: number): TargetData {
  let m = targetCache.get(poster);
  if (!m) targetCache.set(poster, (m = new Map()));
  let t = m.get(widthM);
  if (!t) m.set(widthM, (t = compileTarget(poster, { physicalWidth: widthM })));
  return t;
}

export function runScenario(scn: Scenario, o: RunOptions): RunResult {
  const fps = o.fps ?? 30;
  const td = o.td ?? 0;
  const K = o.K;
  const target = o.target ?? targetFor(o.poster, o.posterWidthM);
  const renderer = createSceneRenderer(scn, { K, poster: o.poster, posterWidthM: o.posterWidthM });
  const pipeline = new Pipeline([target], K, { width: K.width, height: K.height, useImu: !!o.useImu, ...o.pipeline });
  pipeline.td = o.tdGuess ?? 0;
  const acq = new Acquirer([target], K.width, K.height, K);
  const imu = o.useImu ? simulateImu(scn, o.imu) : [];
  let imuIdx = 0;
  const frames: FrameLog[] = [];
  let acquisitions = 0, attempts = 0, firstLock = NaN;
  const corners = [-1, -1, 1, -1, 1, 1, -1, 1].map((v, i) =>
    i % 2 === 0 ? (v * o.posterWidthM) / 2 : (v * o.posterWidthM * (o.poster.height / o.poster.width)) / 2);
  const estR = mat3(), estT = vec3();
  const pe = new Float32Array(2), pg = new Float32Array(2);
  const nFrames = Math.floor(scn.duration * fps);
  for (let k = 0; k < nFrames; k++) {
    const tTrue = k / fps;
    const tCam = tTrue - td;
    // Frames are processed after a pipeline latency, so IMU samples up to
    // ~20 ms past the capture time have arrived by then.
    while (imuIdx < imu.length && imu[imuIdx].t <= tTrue + 0.02) {
      const s = imu[imuIdx++];
      pipeline.pushImu(s.t, s.gyro[0], s.gyro[1], s.gyro[2], s.accel[0], s.accel[1], s.accel[2]);
    }
    const img = renderer.render(tTrue, scn.camera, { noise: 2, ...o.render });
    const res = pipeline.processFrame(img.data, tCam);
    let acquired = false;
    if (res.wantAcquire) {
      attempts++;
      pipeline.markAcquire(res.frameId);
      const a = acq.acquire(img.data, { exclude: pipeline.trackedIndices(), up: o.useImu ? pipeline.upVector() : null });
      if (a && pipeline.applyAcquisition(a, res.frameId) !== null) {
        acquisitions++;
        acquired = true;
      }
    }
    o.onFrame?.(pipeline, tCam, tTrue);
    const s = pipeline.slot(0);
    const visible = !!s && pipeline.isVisible(s, tCam);
    if (visible && Number.isNaN(firstLock)) firstLock = tTrue;
    const gt = relativePose(scn.camera(tTrue), scn.target(tTrue));
    let overlay = NaN, tErr = NaN, rErr = NaN;
    if (visible && pipeline.poseAt(0, tCam, estR, estT)) {
      let e = 0;
      for (let c = 0; c < 4; c++) {
        projectPoint(K, estR, estT, corners[2 * c], corners[2 * c + 1], pe);
        projectPoint(K, gt.R, gt.t, corners[2 * c], corners[2 * c + 1], pg);
        e += Math.hypot(pe[0] - pg[0], pe[1] - pg[1]);
      }
      overlay = e / 4;
      tErr = Math.hypot(estT[0] - gt.t[0], estT[1] - gt.t[1], estT[2] - gt.t[2]);
      rErr = (rotationAngle(estR, gt.R) * 180) / Math.PI;
    }
    frames.push({
      t: tTrue, tCam, visible, status: s?.status ?? 'none', overlayErrPx: overlay, transErr: tErr, rotErrDeg: rErr,
      estT: estT.slice(), estR: estR.slice(), inliers: s?.tracker.inliers ?? 0, acquired, trackMs: res.timings.total,
    });
  }
  return { frames, acquisitions, acquireAttempts: attempts, firstLockTime: firstLock, pipeline };
}

/** Std of translation (m) and rotation (deg) of the estimates over a window: static jitter. */
export function jitter(frames: FrameLog[]): { transStd: number; rotStdDeg: number } {
  const v = frames.filter((f) => f.visible);
  const n = v.length;
  if (n < 2) return { transStd: NaN, rotStdDeg: NaN };
  const m = [0, 0, 0];
  for (const f of v) for (let i = 0; i < 3; i++) m[i] += f.estT[i] / n;
  let ts = 0;
  for (const f of v) for (let i = 0; i < 3; i++) ts += (f.estT[i] - m[i]) ** 2;
  // Rotation: deviation from the first estimate's frame, mean-removed.
  const ref = v[0].estR;
  const ws = v.map((f) => so3Log(mat3MulAtB(ref, f.estR, mat3()), vec3()));
  const wm = [0, 1, 2].map((i) => ws.reduce((s, w) => s + w[i], 0) / n);
  let rs = 0;
  for (const w of ws) for (let i = 0; i < 3; i++) rs += (w[i] - wm[i]) ** 2;
  return { transStd: Math.sqrt(ts / (n - 1)), rotStdDeg: (Math.sqrt(rs / (n - 1)) * 180) / Math.PI };
}

export function percentile(xs: number[], p: number): number {
  const a = xs.filter((x) => Number.isFinite(x)).sort((x, y) => x - y);
  if (a.length === 0) return NaN;
  return a[Math.min(a.length - 1, Math.floor(p * a.length))];
}
