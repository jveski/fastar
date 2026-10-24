// Scripted camera/target trajectories with known ground truth, plus the IMU
// streams a phone would produce along them. World frame: z up (gravity -z).
import { mat3, mat3MulAtB, mat3TVec, so3Exp, so3Log, mat3Mul, vec3, type Mat3 } from '../math/rot';
import { Rng } from '../math/rng';
import type { GrayImage } from '../image/gray';
import type { Intrinsics } from '../vision/camera';
import { Renderer, type RigidPose, type TexturedPlane } from './renderer';
import { makeBackground } from './poster';

export const GRAVITY = 9.81;

/** World-from-target rotation for a poster on a wall facing -y (target z into the wall = +y). */
export const WALL_R: Mat3 = Float64Array.of(1, 0, 0, 0, 0, 1, 0, -1, 0);

/** World-from-camera rotation (OpenCV camera axes) looking from eye at a point. */
export function lookAt(eye: ArrayLike<number>, at: ArrayLike<number>, up: ArrayLike<number> = [0, 0, 1]): Mat3 {
  let zx = at[0] - eye[0], zy = at[1] - eye[1], zz = at[2] - eye[2];
  const zn = Math.hypot(zx, zy, zz);
  zx /= zn; zy /= zn; zz /= zn;
  let xx = zy * up[2] - zz * up[1], xy = zz * up[0] - zx * up[2], xz = zx * up[1] - zy * up[0];
  const xn = Math.hypot(xx, xy, xz);
  xx /= xn; xy /= xn; xz /= xn;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  return Float64Array.of(xx, yx, zx, xy, yy, zy, xz, yz, zz);
}

export function pose(R: Mat3, p: ArrayLike<number>): RigidPose {
  return { R, p: Float64Array.from(p) };
}

export interface Scenario {
  name: string;
  duration: number;
  camera: (t: number) => RigidPose;
  target: (t: number) => RigidPose;
  occluded?: (t: number) => boolean;
}

const staticTarget = (): RigidPose => pose(WALL_R, [0, 0, 0]);

/** Camera at `dist` in front of the target, looking at it, with optional extra rotation (camera frame). */
function viewFrom(dist: number, offX = 0, offZ = 0, rot?: [number, number, number]): RigidPose {
  const eye = [offX, -dist, offZ];
  let R = lookAt(eye, [0, 0, 0]);
  if (rot) R = mat3Mul(R, so3Exp(rot[0], rot[1], rot[2], mat3()), mat3());
  return pose(R, eye);
}

export const scenarios = {
  static: (dist = 0.3, duration = 3): Scenario => ({
    name: 'static',
    duration,
    camera: () => viewFrom(dist, 0.02, 0.01),
    target: staticTarget,
  }),
  slowSweep: (dist = 0.35, duration = 4): Scenario => ({
    name: 'slow-sweep',
    duration,
    camera: (t) => {
      const a = 0.5 * Math.sin((2 * Math.PI * t) / duration);
      return viewFrom(dist * (1 + 0.15 * Math.sin(t)), dist * Math.sin(a), 0.05 * Math.sin(1.3 * t));
    },
    target: staticTarget,
  }),
  /** Yaw/pitch shake with a 300 deg/s peak rotation rate. */
  shake: (peakDegPerSec = 300, dist = 0.35, duration = 3, freqHz = 5): Scenario => {
    const amp = ((peakDegPerSec * Math.PI) / 180) / (2 * Math.PI * freqHz);
    return {
      name: `shake-${peakDegPerSec}`,
      duration,
      camera: (t) => {
        const ramp = Math.min(1, t / 0.5);
        const yaw = ramp * amp * Math.sin(2 * Math.PI * freqHz * t);
        const pitch = ramp * 0.3 * amp * Math.sin(2 * Math.PI * freqHz * 0.7 * t);
        return viewFrom(dist, 0, 0, [pitch, yaw, 0]);
      },
      target: staticTarget,
    };
  },
  /** ~2.5 s of moderate wobble (enough to calibrate t_d), then a 300 deg/s shake. */
  calibrateThenShake: (dist = 0.35, calib = 2.5, shakeDur = 2.5, freqHz = 5): Scenario => {
    const amp = ((300 * Math.PI) / 180) / (2 * Math.PI * freqHz);
    return {
      name: 'calibrate-then-shake',
      duration: calib + shakeDur,
      camera: (t) => {
        // Wobble: ~80 deg/s peaks with varying axes so all rates are excited.
        const wob = Math.min(1, t / 0.3);
        let pitch = wob * 0.12 * Math.sin(2 * Math.PI * 1.1 * t);
        let yaw = wob * 0.15 * Math.sin(2 * Math.PI * 0.8 * t + 1);
        let roll = wob * 0.1 * Math.sin(2 * Math.PI * 1.3 * t + 2);
        if (t > calib) {
          const ramp = Math.min(1, (t - calib) / 0.4);
          yaw += ramp * amp * Math.sin(2 * Math.PI * freqHz * (t - calib));
          pitch += ramp * 0.3 * amp * Math.sin(2 * Math.PI * freqHz * 0.7 * (t - calib));
          roll *= 1 - ramp;
        }
        return viewFrom(dist, 0.02 * Math.sin(0.9 * t), 0.01 * Math.sin(1.4 * t), [pitch, yaw, roll]);
      },
      target: staticTarget,
    };
  },
  occlusion: (dist = 0.35, duration = 3, start = 1.2, length = 0.6): Scenario => ({
    name: 'occlusion',
    duration,
    camera: (t) => viewFrom(dist, 0.03 * Math.sin(1.7 * t), 0.02 * Math.sin(1.1 * t), [0.08 * Math.sin(2.3 * t), 0.12 * Math.sin(1.9 * t), 0]),
    target: staticTarget,
    occluded: (t) => t >= start && t < start + length,
  }),
  movingTarget: (dist = 0.4, duration = 4): Scenario => ({
    name: 'moving-target',
    duration,
    camera: () => viewFrom(dist, 0.01, 0.0),
    target: (t) => {
      const s = t < 1.5 ? 0 : Math.min(1, (t - 1.5) / 1.0);
      const off = 0.08 * (s * s * (3 - 2 * s));
      const R = mat3Mul(WALL_R, so3Exp(0, 0, 0.3 * (s * s * (3 - 2 * s)), mat3()), mat3());
      return pose(R, [off, 0, 0.5 * off]);
    },
  }),
};

/** Ground-truth target-in-camera pose: R_ct = R_wc^T R_wt, t_ct = R_wc^T (p_wt - p_wc). */
export function relativePose(cam: RigidPose, tgt: RigidPose): { R: Float64Array; t: Float64Array } {
  const R = mat3MulAtB(cam.R, tgt.R, mat3());
  const t = mat3TVec(cam.R, tgt.p[0] - cam.p[0], tgt.p[1] - cam.p[1], tgt.p[2] - cam.p[2], vec3());
  return { R, t };
}

export interface SimSceneOptions {
  K: Intrinsics;
  poster: GrayImage;
  posterWidthM: number;
  seed?: number;
  /** The poster is a screen playing this (frames the size of `poster`). */
  textureAt?: (t: number) => GrayImage;
}

/** Wall + poster + optional occluder, wired to a scenario. */
export function createSceneRenderer(scn: Scenario, o: SimSceneOptions): Renderer {
  const mpt = o.posterWidthM / o.poster.width;
  const wall = makeBackground(512, (o.seed ?? 1) + 100);
  const behind = (p: RigidPose, dz: number): RigidPose => {
    const R = p.R;
    return pose(R, [p.p[0] + R[2] * dz, p.p[1] + R[5] * dz, p.p[2] + R[8] * dz]);
  };
  const planes: TexturedPlane[] = [
    { pose: () => behind(staticTarget(), 0.002), texture: wall, metersPerTexel: 0.002, tile: true },
    { pose: scn.target, texture: o.poster, textureAt: o.textureAt, metersPerTexel: mpt },
  ];
  if (scn.occluded) {
    const occ = makeBackground(256, 777);
    planes.push({
      pose: (t) => behind(scn.target(t), -0.08),
      texture: occ,
      metersPerTexel: (o.posterWidthM * 2) / 256,
      active: scn.occluded,
    });
  }
  return new Renderer(o.K, planes, o.seed ?? 1);
}

export interface ImuSample {
  t: number; // seconds, IMU clock
  gyro: Float64Array; // rad/s, camera axes
  accel: Float64Array; // m/s^2 specific force, camera axes
}

export interface ImuSimOptions {
  rate?: number;
  gyroBias?: [number, number, number];
  gyroNoise?: number; // rad/s per sample
  accelNoise?: number; // m/s^2 per sample
  seed?: number;
}

/** Angular velocity in camera axes by central difference. */
export function angularVelocity(scn: Scenario, t: number, h = 1e-4): Float64Array {
  const a = scn.camera(t - h).R, b = scn.camera(t + h).R;
  const d = mat3MulAtB(a, b, mat3());
  const w = so3Log(d, vec3());
  for (let i = 0; i < 3; i++) w[i] /= 2 * h;
  return w;
}

export function simulateImu(scn: Scenario, o: ImuSimOptions = {}): ImuSample[] {
  const rate = o.rate ?? 100;
  const bias = o.gyroBias ?? [0, 0, 0];
  const rng = new Rng(o.seed ?? 5);
  const gn = o.gyroNoise ?? 0.004;
  const an = o.accelNoise ?? 0.05;
  const out: ImuSample[] = [];
  const h = 1e-3;
  for (let i = 0; i <= scn.duration * rate; i++) {
    const t = i / rate;
    const w = angularVelocity(scn, t);
    const gyro = Float64Array.of(w[0] + bias[0] + rng.gauss() * gn, w[1] + bias[1] + rng.gauss() * gn, w[2] + bias[2] + rng.gauss() * gn);
    const p0 = scn.camera(t - h).p, p1 = scn.camera(t).p, p2 = scn.camera(t + h).p;
    const aw = [0, 1, 2].map((k) => (p2[k] - 2 * p1[k] + p0[k]) / (h * h));
    const R = scn.camera(t).R;
    const f = mat3TVec(R, aw[0], aw[1], aw[2] + GRAVITY, vec3());
    const accel = Float64Array.of(f[0] + rng.gauss() * an, f[1] + rng.gauss() * an, f[2] + rng.gauss() * an);
    out.push({ t, gyro, accel });
  }
  return out;
}
