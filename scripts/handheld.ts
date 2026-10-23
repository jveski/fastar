// Realistic hand-held jitter study: npx tsx scripts/handheld.ts [key=value ...]
// Models what the clean synthetic tests leave out: focal-length error, rolling
// shutter, motion blur, frame timestamps quantized to the display (rVFC
// without captureTime), a 60 Hz IMU with timestamp jitter, and hand tremor.
// Jitter is measured on screen: the frame-to-frame change of the overlay error
// of a 6 cm cube against where the target appears in the (rolling-shutter)
// image, using the pose the renderer would have for each presented frame.
// Options (key=value): post=1 (syncDisplay: pose after the frame's own vision
// update; default is the prediction drawn over a live <video>), sm=0 (no
// smoothing), td, tq=0 (exact frame times), rs (readout s), fov/fova (true and
// assumed FOV), tremor, pan, ttrem (translational tremor), dur, seed, tdo=1
// (known t_d), acc/gn (filter noise), v (per-frame log).
import { intrinsicsFromFov } from '../src/vision/camera';
import { makePoster } from '../src/sim/poster';
import { createSceneRenderer, lookAt, pose, relativePose, angularVelocity, GRAVITY, WALL_R, type Scenario } from '../src/sim/scene';
import { targetFor, percentile } from '../src/sim/harness';
import { Pipeline } from '../src/track/pipeline';
import { Acquirer } from '../src/track/acquire';
import { mat3, mat3Mul, mat3MulAtB, mat3TVec, so3Exp, so3Log, vec3 } from '../src/math/rot';
import { Rng } from '../src/math/rng';
import { PoseSmoother } from '../src/fusion/smooth';

const arg = (k: string, d: number) => {
  const a = process.argv.find((x) => x.startsWith(`${k}=`));
  return a ? Number(a.slice(k.length + 1)) : d;
};
const W = 640, H = 480;
const fovTrue = arg('fov', 66), fovAssumed = arg('fova', 63);
const Ktrue = intrinsicsFromFov(W, H, fovTrue);
const Kpipe = intrinsicsFromFov(W, H, fovAssumed);
const readout = arg('rs', 0.025), exposure = arg('exp', 0.01), noise = arg('noise', 3), blur = arg('blur', 1);
const imuHz = arg('imuhz', 60), imuJit = arg('imujit', 0.003), td = arg('td', 0.04);
const vsync = arg('vsync', 1 / 60);
const quantize = arg('tq', 1) > 0;
const useImu = arg('imu', 1) > 0;
const duration = arg('dur', 6);
const tremor = arg('tremor', 1), pan = arg('pan', 1), ttrem = arg('ttrem', 1);
const seed = arg('seed', 3);
const verbose = process.argv.includes('v');
// post=1: pose after this frame's own vision update (video drawn once tracked).
const post = arg('post', 0) > 0;

// Hand-held: tremor (several Hz, a few deg/s) plus a slow look-around.
const scn: Scenario = {
  name: 'handheld',
  duration,
  camera: (t) => {
    const s = (f: number, ph: number) => Math.sin(2 * Math.PI * f * t + ph);
    const ox = pan * 0.06 * s(0.13, 0) + ttrem * tremor * (0.0015 * s(1.7, 1) + 0.0007 * s(4.3, 2));
    const oz = pan * 0.04 * s(0.17, 1) + ttrem * tremor * (0.001 * s(2.1, 3) + 0.0006 * s(5.1, 0.5));
    const dist = 0.3 + pan * 0.05 * s(0.11, 2) + ttrem * tremor * 0.001 * s(1.3, 4);
    const eye = [ox, -dist, oz];
    let R = lookAt(eye, [pan * 0.01 * s(0.2, 1), 0, 0]);
    const d = Math.PI / 180;
    const rx = tremor * (0.25 * d * s(2.3, 0.3) + 0.12 * d * s(6.1, 1.1) + 0.05 * d * s(9.7, 2));
    const ry = tremor * (0.3 * d * s(1.9, 2.3) + 0.1 * d * s(5.3, 0.1) + 0.05 * d * s(11.3, 1));
    const rz = tremor * (0.15 * d * s(1.1, 0.7) + 0.05 * d * s(7.7, 2));
    R = mat3Mul(R, so3Exp(rx, ry, rz, mat3()), mat3());
    return pose(R, eye);
  },
  target: () => pose(WALL_R, [0, 0, 0]),
};

const poster = makePoster(640, 480, 11);
const target = targetFor(poster, 0.2);
const renderer = createSceneRenderer(scn, { K: Ktrue, poster, posterWidthM: 0.2 });
const pipeline = new Pipeline([target], Kpipe, { width: W, height: H, useImu, calibrateTd: !arg('tdo', 0), eskf: { ...(arg('acc', 0) ? { accelNoise: arg('acc', 0) } : {}), ...(arg('gn', 0) ? { gyroNoise: arg('gn', 0) } : {}) } });
if (arg('tdo', 0)) pipeline.setTd(td, true);
const acq = new Acquirer([target], W, H, Kpipe);
const rng = new Rng(seed);

// IMU (iOS devicemotion-like): 60 Hz, bias-corrected, event timestamps jittered.
interface S { t: number; g: number[]; a: number[] }
const imu: S[] = [];
for (let i = 0; i <= duration * imuHz; i++) {
  const t = i / imuHz;
  const w = angularVelocity(scn, t);
  const h = 1e-3;
  const p0 = scn.camera(t - h).p, p1 = scn.camera(t).p, p2 = scn.camera(t + h).p;
  const aw = [0, 1, 2].map((k) => (p2[k] - 2 * p1[k] + p0[k]) / (h * h));
  const f = mat3TVec(scn.camera(t).R, aw[0], aw[1], aw[2] + GRAVITY, vec3());
  imu.push({
    t: t + td + (rng.next() - 0.5) * 2 * imuJit,
    g: [0, 1, 2].map((k) => w[k] + 0.003 * rng.gauss() + [0.002, -0.003, 0.001][k]),
    a: [0, 1, 2].map((k) => f[k] + 0.05 * rng.gauss()),
  });
}
for (let i = 1; i < imu.length; i++) if (imu[i].t <= imu[i - 1].t) imu[i].t = imu[i - 1].t + 1e-4;

// Cube (6 cm, on the target centre) corners in OpenCV target axes (z into the target).
const cube: number[][] = [];
for (const x of [-0.03, 0.03]) for (const y of [-0.03, 0.03]) for (const z of [0, -0.06]) cube.push([x, y, z]);
function project3(K: typeof Ktrue, R: ArrayLike<number>, t: ArrayLike<number>, X: number[], out: number[]) {
  const px = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
  const py = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
  const pz = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
  out[0] = (K.fx * px) / pz + K.cx; out[1] = (K.fy * py) / pz + K.cy;
}

const fps = 30;
const n = Math.floor(duration * fps);
let ii = 0;
const R = new Float64Array(9), tv = new Float64Array(3);
const errs: (number[] | null)[] = [];
let rejects = 0, movingFrames = 0, lost = 0;
const nisList: number[] = [];
const pe = [0, 0], pg = [0, 0];
const latency = 0.05;
let dbg = '';
// Where a target-fixed 3D point appears in the (rolling-shutter) image: the
// row it lands on is exposed at its own time.
function trueProject(X: number[], t: number, out: number[]) {
  let y = Ktrue.cy;
  for (let it = 0; it < 4; it++) {
    const tr = t + (y / (H - 1) - 0.5) * readout;
    const g = relativePose(scn.camera(tr), scn.target(tr));
    project3(Ktrue, g.R, g.t, X, out);
    y = Math.max(0, Math.min(H - 1, out[1]));
  }
}
let tTrueNow = 0;
const smoother = arg('sm', 0.75) > 0 ? new PoseSmoother({ keep: arg('sm', 0.75) }) : null;
const Rp = new Float64Array(9), tpv = new Float64Array(3);
function measure(tCam: number, gt: { R: Float64Array; t: Float64Array }) {
  // Pose the renderer would use for this presented frame.
  const slot = pipeline.slot(0);
  if (slot && pipeline.isVisible(slot, tCam) && pipeline.poseAt(0, tCam, R, tv)) {
    if (smoother) {
      const tp = smoother.previousTime;
      const ok = Number.isFinite(tp) && pipeline.poseAt(0, tp, Rp, tpv);
      smoother.update(tCam, R, tv, ok ? Rp : null, ok ? tpv : null);
    }
    const e: number[] = [];
    for (const X of cube) {
      project3(Kpipe, R, tv, X, pe);
      trueProject(X, tTrueNow, pg);
      e.push(pe[0] - pg[0], pe[1] - pg[1]);
    }
    errs.push(e);
    if (verbose) {
      const E = mat3MulAtB(gt.R, R, mat3());
      const w = so3Log(E, vec3());
      dbg = `rot=[${[...w].map((v) => (v * 57.3).toFixed(3)).join(',')}] dt_mm=[${[0, 1, 2].map((i) => ((tv[i] - gt.t[i]) * 1000).toFixed(2)).join(',')}]`;
    }
  } else errs.push(null);
}
for (let k = 0; k < n; k++) {
  const tTrue = k / fps + 0.0123;
  // Reported capture time: true (camera clock) or expectedDisplayTime - latency estimate.
  let tCam = tTrue;
  if (quantize) tCam = Math.ceil((tTrue + latency) / vsync) * vsync - latency - vsync / 2;
  // IMU delivered up to roughly when this frame is presented/processed.
  while (ii < imu.length && imu[ii].t <= tTrue + 0.05) {
    const s = imu[ii++];
    pipeline.pushImu(s.t, s.g[0], s.g[1], s.g[2], s.a[0], s.a[1], s.a[2]);
  }
  const gt = relativePose(scn.camera(tTrue), scn.target(tTrue));
  tTrueNow = tTrue;
  if (!post) measure(tCam, gt);
  const img = renderer.render(tTrue, scn.camera, { noise, blurPasses: blur, readout, exposure, exposureSamples: exposure > 0 ? 3 : 1 });
  const res = pipeline.processFrame(img.data, tCam);
  if (post) measure(tCam, gt);
  if (res.lost.length) lost++;
  const s2 = pipeline.slot(0);
  if (s2) {
    const nis = res.targets[0]?.nis;
    if (Number.isFinite(nis)) { nisList.push(nis); if (nis > s2.eskf.o.gate) rejects++; }
    if (s2.eskf.moving) movingFrames++;
  }
  if (res.wantAcquire) {
    pipeline.markAcquire(res.frameId);
    const a = acq.acquire(img.data, { exclude: pipeline.trackedIndices(), up: useImu ? pipeline.upVector() : null });
    if (a) pipeline.applyAcquisition(a, res.frameId);
  }
  if (verbose && errs[k]) {
    const e = errs[k]!;
    const p = errs[k - 1];
    const d = p ? Math.sqrt(e.reduce((a, v, i) => a + (v - p[i]) ** 2, 0) / 8) : NaN;
    const w = angularVelocity(scn, tTrue);
    console.log(`${tTrue.toFixed(3)} err=${Math.sqrt(e.reduce((a, v) => a + v * v, 0) / 8).toFixed(2)} d=${d.toFixed(2)} nis=${(res.targets[0]?.nis ?? NaN).toFixed(1)} inl=${res.targets[0]?.inliers} w=${(Math.hypot(w[0], w[1], w[2]) * 57.3).toFixed(1)}dps mv=${pipeline.slot(0)?.eskf.moving ? 1 : 0} tdc=${pipeline.tdCalibrated ? 1 : 0} td=${(pipeline.td * 1000).toFixed(1)} ${dbg}`);
  }
}
// Frame-to-frame jitter (px RMS per corner) after a 1.5 s settle (and t_d calibration).
const d: number[] = [], abs: number[] = [];
const start = Math.floor(1.5 * fps);
for (let k = start; k < n; k++) {
  const e = errs[k], p = errs[k - 1];
  if (!e) continue;
  abs.push(Math.sqrt(e.reduce((a, v) => a + v * v, 0) / 8));
  if (p) d.push(Math.sqrt(e.reduce((a, v, i) => a + (v - p[i]) ** 2, 0) / 8));
}
// Second difference removes slowly varying (bias-like) error and leaves the jumps.
const dd: number[] = [];
for (let k = start + 1; k < n; k++) {
  const a = errs[k], b = errs[k - 1], c = errs[k - 2];
  if (a && b && c) dd.push(Math.sqrt(a.reduce((s, v, i) => s + (v - 2 * b[i] + c[i]) ** 2, 0) / 8));
}
const rms = (xs: number[]) => Math.sqrt(xs.reduce((a, v) => a + v * v, 0) / Math.max(1, xs.length));
console.log(
  `jitter px: d1 rms=${rms(d).toFixed(3)} p95=${percentile(d, 0.95).toFixed(3)} max=${percentile(d, 1).toFixed(2)} | d2 rms=${rms(dd).toFixed(3)} | err median=${percentile(abs, 0.5).toFixed(2)} p95=${percentile(abs, 0.95).toFixed(2)} | visible=${errs.filter(Boolean).length}/${n} lost=${lost} rejects=${rejects} moving=${movingFrames} nis med=${percentile(nisList, 0.5).toFixed(1)} rScale=${pipeline.slot(0)?.eskf.rScale.toFixed(1)} tdCal=${pipeline.tdCalibrated} td=${(pipeline.td * 1000).toFixed(1)}ms`,
);
