// Error-state Kalman filter fusing gyro, gravity and delayed vision poses.
//
// World frame: gravity-aligned (z up), fixed at initialization.
// Nominal state x (21): q_wc(4) p_wc(3) v_wc(3) q_wt(4) p_wt(3) b_g(3) t_d(1)
// Error state (19):     dth_wc(3) dth_wt(3) dp_wt(3) dp_wc(3) dv_wc(3) db_g(3) dt_d(1)
// Rotation errors are left (global): R = Exp(dth) R_hat.
// Times are seconds; IMU time = camera time + t_d.
//
// Without an IMU the same filter runs a constant-velocity model: the gyro
// reading is 0 and b_g acts as minus the angular velocity.
import { ImuBuffer } from './imu';
import { rateAt, integrateCamera, SNAP_HEADER, SNAP_ENTRY, NEAR_ENTRY } from './predict';
import {
  mat3, mat3FromQuat, mat3MulAtB, mat3TVec, so3Log, quatIntegrateLeft, quatIntegrateRight, quat,
  mat3MulABt, mat3Mul, mat3Vec, quatFromMat3, vec3,
} from '../math/rot';
import { cholesky, choleskySolve } from '../math/linalg';

const N = 19;
const TH = 0, TT = 3, PT = 6, PC = 9, VC = 12, BG = 15, TD = 18;
const NX = 21;
const ENTRY = 1 + NX + N * N;

export interface EskfOptions {
  useImu: boolean;
  /** Gyro white noise density (rad/s/sqrt(Hz)). */
  gyroNoise: number;
  /**
   * Prior std of t_d (s) used until it is calibrated. With an unknown time
   * offset the gyro prediction is off by about |w_dot| * t_d, which is added
   * as rotation process noise (large in fast motion, negligible when slow).
   */
  tdPriorSigma: number;
  /** Gyro bias random walk (rad/s/sqrt(s)). */
  gyroBiasWalk: number;
  /**
   * Camera linear acceleration noise: std in m/s^2 of the acceleration over
   * one 30 Hz frame (spectral density = accelNoise^2 / 30).
   */
  accelNoise: number;
  /** Same, while coasting without vision (hand-held motion is gentle on average). */
  coastAccelNoise: number;
  /** While coasting, velocity decays toward zero with this time constant (s). */
  coastVelocityTau: number;
  /** Vision-only: angular acceleration noise (rad/s^2/sqrt(Hz)). */
  angAccelNoise: number;
  targetRotWalk: number;
  targetPosWalk: number;
  /** Time-offset random walk (s/sqrt(s)). */
  tdWalk: number;
  /** Allow online refinement of t_d (activated by enableTd once a coarse estimate exists). */
  estimateTd: boolean;
  /** Target random walks while the target is judged to be moving (hand-held). */
  movingRotWalk: number;
  movingPosWalk: number;
  /** Chi-square gate for 6-dof vision updates (99%: 16.81). */
  gate: number;
  /**
   * Updates with NIS between gate and gate * softGate are applied with the
   * noise inflated by NIS / gate (putting the innovation on the gate) instead
   * of rejected; beyond that they are rejected.
   */
  softGate: number;
  /**
   * Moving-target mode also needs the smoothed innovation to be this large
   * (Mahalanobis; about 1 for unbiased noise), so zero-mean errors do not trigger it.
   */
  movingThreshold: number;
  /** Largest factor the vision covariance may be scaled up by from innovation statistics. */
  maxNoiseScale: number;
  /** Adaptation rate of that scale per update (log units per unit of NIS/6 excess). */
  noiseScaleRate: number;
  /** Tilt measurement noise (unit-vector components). */
  gravityNoise: number;
  /** Propagation step without an IMU (s). */
  stepNoImu: number;
}

export const DEFAULT_ESKF: EskfOptions = {
  useImu: true,
  gyroNoise: 0.002,
  tdPriorSigma: 0.03,
  gyroBiasWalk: 2e-4,
  accelNoise: 2,
  coastAccelNoise: 0.5,
  coastVelocityTau: 1.0,
  angAccelNoise: 12,
  targetRotWalk: 2e-4,
  targetPosWalk: 1e-4,
  tdWalk: 2e-4,
  estimateTd: true,
  movingRotWalk: 0.5,
  movingPosWalk: 0.15,
  gate: 16.81,
  softGate: 6,
  maxNoiseScale: 100,
  movingThreshold: 30,
  noiseScaleRate: 0.05,
  gravityNoise: 0.08,
  stepNoImu: 0.01,
};

export interface UpdateResult {
  accepted: boolean;
  nis: number;
  reinitialized: boolean;
  pending?: boolean;
}

export class Eskf {
  readonly o: EskfOptions;
  readonly x = new Float64Array(NX);
  readonly P = new Float64Array(N * N);
  t = 0;
  initialized = false;
  /** Last raw gyro reading used for propagation (camera axes, rad/s). */
  readonly wRaw = new Float64Array(3);

  // Health / adaptivity.
  nisEma = 6;
  /** Learned scale on the vision covariance (see applyPose). */
  rScale = 1;
  /** Bit history of the last 10 updates whose NIS exceeded the 95% bound. */
  private excessBits = 0;
  /** Smoothed innovation and its Mahalanobis size (moving-target evidence). */
  private readonly rEma = new Float64Array(6);
  private persistentNis = 0;
  moving = false;
  /** 0..1 blend toward the moving-target process noise. */
  movingWeight = 0;
  consecutiveRejects = 0;
  updates = 0;
  /** IMU time of the last accepted vision update. */
  lastUpdateT = -Infinity;
  /** Whether t_d is currently a filter state (after a coarse estimate). */
  tdActive = false;
  private tdSigma = 0.003;

  private readonly hist: Float64Array;
  private histHead = 0;
  private histCount = 0;
  private readonly cap: number;

  // Scratch.
  private readonly Rwc = mat3();
  private readonly Rwt = mat3();
  private readonly Rct = mat3();
  private readonly M3 = mat3();
  private readonly v3 = vec3();
  private readonly H = new Float64Array(6 * N);
  private readonly PHt = new Float64Array(N * 6);
  private readonly S = new Float64Array(36);
  private readonly Sinv = new Float64Array(36);
  private readonly col6 = new Float64Array(6);
  private readonly r6 = new Float64Array(6);
  private readonly K = new Float64Array(N * 6);
  private readonly dx = new Float64Array(N);
  private readonly HP = new Float64Array(6 * N);
  private readonly JPo = new Float64Array(6 * N);
  private readonly S6a = new Float64Array(36);
  private readonly S6b = new Float64Array(36);
  private readonly S3g = new Float64Array(9);
  private readonly covScaled = new Float64Array(36);
  private readonly qa = quat();
  private readonly vt = vec3();
  private readonly wTmp = vec3();
  private readonly wPrev = vec3();
  private readonly bTmp = vec3();
  private readonly w0 = vec3();
  private readonly pv = vec3();
  private readonly vv = vec3();
  // Pending vision measurements waiting for IMU data to cover their time.
  private readonly pend = new Float64Array(4 * 52);
  private pendCount = 0;

  constructor(opts: Partial<EskfOptions> = {}, historySeconds = 0.6, rate = 100) {
    this.o = { ...DEFAULT_ESKF, ...opts };
    if (!this.o.useImu) this.o.estimateTd = false;
    this.cap = Math.ceil(historySeconds * Math.max(rate, 1 / this.o.stepNoImu)) + 8;
    this.hist = new Float64Array(this.cap * ENTRY);
  }

  get qwc(): Float64Array { return this.x.subarray(0, 4); }
  get pwc(): Float64Array { return this.x.subarray(4, 7); }
  get vwc(): Float64Array { return this.x.subarray(7, 10); }
  get qwt(): Float64Array { return this.x.subarray(10, 14); }
  get pwt(): Float64Array { return this.x.subarray(14, 17); }
  get bg(): Float64Array { return this.x.subarray(17, 20); }
  get td(): number { return this.x[20]; }
  set td(v: number) { this.x[20] = v; }

  /**
   * Initialize from a vision pose (target in camera) at camera time tCam.
   * `up` is the unit up-vector (opposite gravity) in camera axes, if known.
   */
  initialize(tCam: number, Rct: ArrayLike<number>, tct: ArrayLike<number>, cov6: ArrayLike<number>, up: ArrayLike<number> | null, td = this.x[20], bg?: ArrayLike<number>): void {
    const x = this.x;
    x.fill(0);
    x[20] = td;
    if (bg) { x[17] = bg[0]; x[18] = bg[1]; x[19] = bg[2]; }
    const Rwc = this.Rwc;
    if (up) worldFromUp(up, Rwc);
    else { Rwc.fill(0); Rwc[0] = Rwc[4] = Rwc[8] = 1; }
    quatFromMat3(Rwc, this.qwc);
    this.t = tCam + td;
    this.P.fill(0);
    const P = this.P;
    const tilt = up ? 1e-4 : 1e-6;
    for (let i = 0; i < 3; i++) {
      P[(TH + i) * N + TH + i] = tilt;
      P[(PC + i) * N + PC + i] = 1e-8;
      P[(VC + i) * N + VC + i] = 0.04;
      P[(BG + i) * N + BG + i] = this.o.useImu ? 4e-4 : 1.0;
    }
    P[TD * N + TD] = this.tdActive ? this.tdSigma * this.tdSigma : 0;
    this.setTargetFromMeasurement(Rct, tct, cov6);
    this.initialized = true;
    this.nisEma = 6;
    this.excessBits = 0;
    this.rEma.fill(0);
    this.persistentNis = 0;
    this.moving = false;
    this.movingWeight = 0;
    this.consecutiveRejects = 0;
    this.updates = 0;
    this.lastUpdateT = this.t;
    this.histCount = 0;
    this.histHead = 0;
    this.pushHistory();
  }

  /** Set t_d from an external estimate and let the filter refine it from there. */
  enableTd(td: number, sigma = 0.003): void {
    if (!this.o.estimateTd) { this.x[20] = td; return; }
    this.tdActive = true;
    this.tdSigma = sigma;
    this.x[20] = td;
    const P = this.P;
    for (let j = 0; j < N; j++) { P[TD * N + j] = 0; P[j * N + TD] = 0; }
    P[TD * N + TD] = sigma * sigma;
  }

  /** Target pose (and its covariance block) from a camera-relative measurement and the current camera state. */
  private setTargetFromMeasurement(Rct: ArrayLike<number>, tct: ArrayLike<number>, cov6: ArrayLike<number>): void {
    const Rwc = mat3FromQuat(this.qwc, this.Rwc);
    const Rwt = mat3Mul(Rwc, Rct, this.Rwt);
    quatFromMat3(Rwt, this.qwt);
    const p = this.pwc;
    const w = mat3Vec(Rwc, tct[0], tct[1], tct[2], this.v3);
    this.pwt[0] = p[0] + w[0]; this.pwt[1] = p[1] + w[1]; this.pwt[2] = p[2] + w[2];
    const P = this.P;
    // Target error from the measurement error e and the other states:
    // dx_t = G (e - Jo dx_o), where G = blockdiag(R_wc, R_wc) inverts the
    // target block of the relative Jacobian. Keeping the cross-covariance makes
    // the relative pose exactly as uncertain as the measurement; dropping it
    // would add the camera's unobservable absolute drift on every re-anchor.
    for (let i = 0; i < 6; i++) for (let j = 0; j < N; j++) { P[(TT + i) * N + j] = 0; P[j * N + TT + i] = 0; }
    const J = this.relativeJacobian(this.HP);
    for (let a = 0; a < 6; a++) for (let k = 0; k < 6; k++) J[a * N + TT + k] = 0;
    const JP = this.JPo, S = this.S6a, T = this.S6b;
    for (let a = 0; a < 6; a++) for (let j = 0; j < N; j++) {
      let s = 0;
      for (let i = 0; i < N; i++) { const ja = J[a * N + i]; if (ja !== 0) s += ja * P[i * N + j]; }
      JP[a * N + j] = s;
    }
    for (let a = 0; a < 6; a++) for (let b = 0; b < 6; b++) {
      let s = cov6[a * 6 + b];
      for (let j = 0; j < N; j++) s += JP[a * N + j] * J[b * N + j];
      S[a * 6 + b] = s;
    }
    // (G X)[a] = R_wc row (a mod 3) applied to the matching 3-row block of X.
    for (let a = 0; a < 6; a++) {
      const r = (a % 3) * 3, o = a < 3 ? 0 : 3;
      for (let j = 0; j < N; j++) {
        if (j >= TT && j < TT + 6) continue;
        const v = -(Rwc[r] * JP[o * N + j] + Rwc[r + 1] * JP[(o + 1) * N + j] + Rwc[r + 2] * JP[(o + 2) * N + j]);
        P[(TT + a) * N + j] = v;
        P[j * N + TT + a] = v;
      }
      for (let b = 0; b < 6; b++) T[a * 6 + b] = Rwc[r] * S[o * 6 + b] + Rwc[r + 1] * S[(o + 1) * 6 + b] + Rwc[r + 2] * S[(o + 2) * 6 + b];
    }
    for (let a = 0; a < 6; a++) for (let b = 0; b < 6; b++) {
      const r = (b % 3) * 3, o = b < 3 ? 0 : 3;
      P[(TT + a) * N + TT + b] = T[a * 6 + o] * Rwc[r] + T[a * 6 + o + 1] * Rwc[r + 1] + T[a * 6 + o + 2] * Rwc[r + 2];
    }
    // Rotational slack: the anchoring measurement may still carry t_d error,
    // and the first updates need room to correct it. Camera attitude
    // uncertainty stays small (gyro-driven), unlike the camera's position.
    for (let i = 0; i < 3; i++) P[(TT + i) * N + TT + i] += P[(TH + i) * N + TH + i];
  }

  /** Advance to IMU time t using samples from `imu` (or constant-velocity steps without one). */
  propagateTo(t: number, imu: ImuBuffer | null): void {
    if (!this.initialized || t <= this.t) return;
    if (!this.o.useImu || !imu || imu.count === 0) {
      while (this.t < t - 1e-9) {
        const dt = Math.min(this.o.stepNoImu, t - this.t);
        this.step(dt, 0, 0, 0);
        this.t += dt;
        this.pushHistory();
      }
      return;
    }
    let i = imu.firstAfter(this.t);
    // Current rate: latest sample at or before this.t.
    for (; i < imu.count; i++) {
      const s = imu.slot(i);
      const ts = imu.t[s];
      if (ts > t) break;
      const wx = imu.w[3 * s], wy = imu.w[3 * s + 1], wz = imu.w[3 * s + 2];
      // Trapezoid between the previous rate and this sample.
      const prevT = this.t;
      this.step(ts - prevT, 0.5 * (this.wRaw[0] + wx), 0.5 * (this.wRaw[1] + wy), 0.5 * (this.wRaw[2] + wz));
      this.wRaw[0] = wx; this.wRaw[1] = wy; this.wRaw[2] = wz;
      this.t = ts;
      if (imu.hasAccel[s] && Math.floor(ts * 10) !== Math.floor(prevT * 10)) {
        this.updateGravity(imu.a[3 * s], imu.a[3 * s + 1], imu.a[3 * s + 2]);
      }
      this.pushHistory();
    }
    if (t > this.t + 1e-9) {
      // Partial step to a time between samples (interpolated rate) or past
      // the newest sample (held rate).
      const w = rateAt(imu, t, this.wTmp);
      this.step(t - this.t, 0.5 * (this.wRaw[0] + w[0]), 0.5 * (this.wRaw[1] + w[1]), 0.5 * (this.wRaw[2] + w[2]));
      this.wRaw.set(w);
      this.t = t;
      this.pushHistory();
    }
  }

  /** One propagation step of nominal state and covariance with raw gyro (wx, wy, wz). */
  private step(dt: number, wx: number, wy: number, wz: number): void {
    if (dt <= 0) return;
    const x = this.x, o = this.o;
    const ox = wx - x[17], oy = wy - x[18], oz = wz - x[19];
    quatIntegrateRight(this.qwc, ox * dt, oy * dt, oz * dt);
    x[4] += x[7] * dt; x[5] += x[8] * dt; x[6] += x[9] * dt;
    const coasting = this.t - this.lastUpdateT > 0.1;
    const decay = coasting ? Math.exp(-dt / o.coastVelocityTau) : 1;
    if (coasting) { x[7] *= decay; x[8] *= decay; x[9] *= decay; }
    const R = mat3FromQuat(this.qwc, this.Rwc);
    const P = this.P;
    // Rows: th_wc += -R dt * b_g rows; p_wc += dt * v rows.
    for (let i = 0; i < 3; i++) {
      const ri = (TH + i) * N;
      const a0 = -R[i * 3] * dt, a1 = -R[i * 3 + 1] * dt, a2 = -R[i * 3 + 2] * dt;
      const b0 = BG * N, b1 = (BG + 1) * N, b2 = (BG + 2) * N;
      for (let j = 0; j < N; j++) P[ri + j] += a0 * P[b0 + j] + a1 * P[b1 + j] + a2 * P[b2 + j];
      const pi = (PC + i) * N, vi = (VC + i) * N;
      for (let j = 0; j < N; j++) P[pi + j] += dt * P[vi + j];
      if (decay !== 1) for (let j = 0; j < N; j++) P[vi + j] *= decay;
    }
    // Columns: same transforms.
    for (let r = 0; r < N; r++) {
      const ro = r * N;
      const g0 = P[ro + BG], g1 = P[ro + BG + 1], g2 = P[ro + BG + 2];
      for (let i = 0; i < 3; i++) {
        P[ro + TH + i] += -dt * (R[i * 3] * g0 + R[i * 3 + 1] * g1 + R[i * 3 + 2] * g2);
        P[ro + PC + i] += dt * P[ro + VC + i];
        if (decay !== 1) P[ro + VC + i] *= decay;
      }
    }
    // Process noise.
    let gyroQ = o.useImu ? o.gyroNoise * o.gyroNoise * dt : 1e-8 * dt;
    if (o.useImu && !this.tdActive && o.estimateTd && dt > 1e-3) {
      const ax = (wx - this.wPrev[0]) / dt, ay = (wy - this.wPrev[1]) / dt, az = (wz - this.wPrev[2]) / dt;
      const e = Math.hypot(ax, ay, az) * o.tdPriorSigma * dt;
      gyroQ += e * e;
    }
    if (dt > 1e-3) { this.wPrev[0] = wx; this.wPrev[1] = wy; this.wPrev[2] = wz; }
    const bgQ = o.useImu ? o.gyroBiasWalk * o.gyroBiasWalk * dt : o.angAccelNoise * o.angAccelNoise * dt;
    const m = this.movingWeight;
    const rw2 = o.targetRotWalk * o.targetRotWalk, pw2 = o.targetPosWalk * o.targetPosWalk;
    const ttQ = (rw2 + m * (o.movingRotWalk * o.movingRotWalk - rw2)) * dt;
    const ptQ = (pw2 + m * (o.movingPosWalk * o.movingPosWalk - pw2)) * dt;
    const an = coasting ? o.coastAccelNoise : o.accelNoise;
    const vQ = ((an * an) / 30) * dt;
    for (let i = 0; i < 3; i++) {
      P[(TH + i) * N + TH + i] += gyroQ;
      P[(TT + i) * N + TT + i] += ttQ;
      P[(PT + i) * N + PT + i] += ptQ;
      P[(PC + i) * N + PC + i] += vQ * dt * dt / 3;
      P[(VC + i) * N + VC + i] += vQ;
      P[(BG + i) * N + BG + i] += bgQ;
    }
    if (this.tdActive) P[TD * N + TD] += o.tdWalk * o.tdWalk * dt;
  }

  /**
   * Vision update with a target-in-camera pose measured at camera time tCam.
   * Rolls back to the capture time, applies the update and re-propagates.
   * With an IMU, measurements newer than the newest IMU sample are queued
   * (see flushPending) so the state never runs ahead of the gyro.
   */
  updatePose(tCam: number, Rm: ArrayLike<number>, tm: ArrayLike<number>, cov6: ArrayLike<number>, imu: ImuBuffer | null): UpdateResult {
    const tau = tCam + this.x[20];
    if (this.o.useImu && imu && tau > imu.latestTime) {
      if (this.pendCount >= 4) this.flushPending(imu, true);
      const o = this.pendCount++ * 52;
      this.pend[o] = tCam;
      for (let i = 0; i < 9; i++) this.pend[o + 1 + i] = Rm[i];
      for (let i = 0; i < 3; i++) this.pend[o + 10 + i] = tm[i];
      for (let i = 0; i < 36; i++) this.pend[o + 13 + i] = cov6[i];
      return { accepted: false, nis: NaN, reinitialized: false, pending: true };
    }
    return this.applyAt(tCam, Rm, tm, cov6, imu);
  }

  /** Apply queued measurements whose time is now covered by IMU data (or all, if forced). */
  flushPending(imu: ImuBuffer | null, force = false): UpdateResult | null {
    let last: UpdateResult | null = null;
    while (this.pendCount > 0) {
      const tCam = this.pend[0];
      if (!force && imu && tCam + this.x[20] > imu.latestTime) break;
      const Rm = this.pend.subarray(1, 10), tm = this.pend.subarray(10, 13), cov = this.pend.subarray(13, 49);
      last = this.applyAt(tCam, Rm.slice(), tm.slice(), cov.slice(), imu);
      this.pend.copyWithin(0, 52, this.pendCount * 52);
      this.pendCount--;
    }
    return last;
  }

  private applyAt(tCam: number, Rm: ArrayLike<number>, tm: ArrayLike<number>, cov6: ArrayLike<number>, imu: ImuBuffer | null): UpdateResult {
    const latest = this.t;
    const tau = tCam + this.x[20];
    if (tau < latest) {
      if (!this.rewindTo(tau)) return { accepted: false, nis: NaN, reinitialized: false };
      this.syncRate(imu);
    }
    this.propagateTo(tau, imu);
    const res = this.applyPose(Rm, tm, cov6);
    this.pushHistory();
    if (latest > this.t) this.propagateTo(latest, imu);
    return res;
  }

  private applyPose(Rm: ArrayLike<number>, tm: ArrayLike<number>, cov6: ArrayLike<number>): UpdateResult {
    const Rwc = mat3FromQuat(this.qwc, this.Rwc);
    const Rwt = mat3FromQuat(this.qwt, this.Rwt);
    const Rct = mat3MulAtB(Rwc, Rwt, this.Rct);
    const x = this.x;
    const dxw = x[14] - x[4], dyw = x[15] - x[5], dzw = x[16] - x[6];
    const tct = mat3TVec(Rwc, dxw, dyw, dzw, this.vt);
    // Residual.
    const E = mat3MulABt(Rm, Rct, this.M3);
    const rth = so3Log(E, this.v3);
    const r = this.r6;
    r[0] = rth[0]; r[1] = rth[1]; r[2] = rth[2];
    r[3] = tm[0] - tct[0]; r[4] = tm[1] - tct[1]; r[5] = tm[2] - tct[2];
    // Jacobian.
    const H = this.H;
    H.fill(0);
    // Camera-frame angular velocity (bias-corrected).
    const wx = this.wRaw[0] - x[17], wy = this.wRaw[1] - x[18], wz = this.wRaw[2] - x[19];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const RT = Rwc[j * 3 + i]; // (Rwc^T)[i][j]
        H[i * N + TH + j] = -RT;
        H[i * N + TT + j] = RT;
        H[(3 + i) * N + PT + j] = RT;
        H[(3 + i) * N + PC + j] = -RT;
      }
      // Rwc^T [d]x, row i: (Rwc^T)[i] . [d]x columns.
      const a0 = Rwc[i], a1 = Rwc[3 + i], a2 = Rwc[6 + i];
      H[(3 + i) * N + TH] = a1 * dzw - a2 * dyw;
      H[(3 + i) * N + TH + 1] = a2 * dxw - a0 * dzw;
      H[(3 + i) * N + TH + 2] = a0 * dyw - a1 * dxw;
    }
    if (this.tdActive) {
      H[0 * N + TD] = -wx; H[1 * N + TD] = -wy; H[2 * N + TD] = -wz;
      const rv = mat3TVec(Rwc, x[7], x[8], x[9], this.vv);
      H[3 * N + TD] = -(wy * tct[2] - wz * tct[1]) - rv[0];
      H[4 * N + TD] = -(wz * tct[0] - wx * tct[2]) - rv[1];
      H[5 * N + TD] = -(wx * tct[1] - wy * tct[0]) - rv[2];
    }
    // Measurement noise: the solver's covariance, scaled up by what the
    // innovations have shown it to understate (rolling shutter, timestamp
    // jitter, focal error and motion the model cannot follow all make real
    // pose errors larger and correlated across points).
    const Rs = this.covScaled, k = this.rScale;
    for (let i = 0; i < 36; i++) Rs[i] = cov6[i] * k;
    const nis = this.kalman(H, r, Rs, 6, true);
    if (nis < 0) return { accepted: false, nis: NaN, reinitialized: false };
    // Covariance matching: drift the scale until the mean NIS sits at its
    // expectation (6). Outliers count, but capped, so one bad frame cannot
    // swamp it.
    const v = Math.min(nis, 2 * this.o.gate);
    this.rScale = Math.min(this.o.maxNoiseScale, Math.max(1, k * Math.exp(this.o.noiseScaleRate * (v / 6 - 1))));
    this.nisEma = 0.8 * this.nisEma + 0.2 * Math.min(nis, 100);
    this.excessBits = ((this.excessBits << 1) | (nis > 12.59 ? 1 : 0)) & 0x3ff;
    // Persistence of the innovation: a moving target pushes it the same way
    // frame after frame, while timing, rolling-shutter and hand-tremor errors
    // flip sign and average out. Its Mahalanobis size uses the current S
    // (factor still in this.Sinv from the dry run).
    const re = this.rEma;
    for (let i = 0; i < 6; i++) re[i] = 0.7 * re[i] + 0.3 * r[i];
    const y = this.col6;
    y.set(re);
    choleskySolve(this.Sinv, 6, y);
    let q = 0;
    for (let i = 0; i < 6; i++) q += re[i] * y[i];
    this.persistentNis = q;
    if (nis > this.o.gate * this.o.softGate) {
      // Gross disagreement: skip, and re-anchor if it persists (the target
      // moved, or the filter diverged).
      this.consecutiveRejects++;
      if (this.consecutiveRejects >= 3) {
        this.reanchorTarget(Rm, tm, cov6);
        // Repeated re-anchoring after the filter settled means the target moves.
        if (this.updates > 5 && !this.moving && this.persistentNis > this.o.movingThreshold) {
          this.moving = true;
          this.movingWeight = 1;
        }
        return { accepted: false, nis, reinitialized: true };
      }
      this.adaptTargetNoise();
      return { accepted: false, nis, reinitialized: false };
    }
    this.consecutiveRejects = 0;
    if (nis > this.o.gate) {
      // Moderate excess: usually unmodelled error (timing, rolling shutter,
      // focal length, or a small target's tilt ambiguity) rather than a bad
      // pose. Snapping to it (re-anchoring) or skipping it (showing the bare
      // prediction) are both visible jumps, so apply it with the measurement
      // noise inflated enough to put the innovation on the gate.
      const s = nis / this.o.gate;
      for (let i = 0; i < 36; i++) Rs[i] *= s;
    }
    this.kalman(H, r, Rs, 6, false);
    this.updates++;
    this.lastUpdateT = this.t;
    this.adaptTargetNoise();
    return { accepted: true, nis, reinitialized: false };
  }

  /**
   * Moving-target detection: a sustained NIS excess (>= 5 of the last 10
   * updates above the 95% bound) raises target process noise and re-anchors.
   * Single outliers are left to gating.
   */
  private adaptTargetNoise(): void {
    let c = 0;
    for (let b = this.excessBits; b; b &= b - 1) c++;
    if (!this.moving && this.updates > 5 && c >= 5 && this.persistentNis > this.o.movingThreshold) {
      this.moving = true;
      this.movingWeight = 1;
      for (let i = 0; i < 3; i++) {
        this.P[(TT + i) * N + TT + i] += 0.01;
        this.P[(PT + i) * N + PT + i] += 1e-4;
      }
    } else if (this.moving && c <= 1) {
      this.movingWeight *= 0.85;
      if (this.movingWeight < 0.01) { this.movingWeight = 0; this.moving = false; }
    }
  }

  /** Reset the target pose to a measurement, keeping the camera state. */
  reanchorTarget(Rm: ArrayLike<number>, tm: ArrayLike<number>, cov6: ArrayLike<number>): void {
    this.setTargetFromMeasurement(Rm, tm, cov6);
    this.consecutiveRejects = 0;
    this.nisEma = 6;
    this.lastUpdateT = this.t;
  }

  /**
   * Kalman update with measurement Jacobian H (m x N), residual r and noise R.
   * With dryRun only computes the NIS. Returns NIS, or -1 if S is singular.
   */
  private kalman(H: Float64Array, r: Float64Array, Rm: ArrayLike<number>, m: number, dryRun: boolean): number {
    const P = this.P, PHt = this.PHt, S = this.S;
    // PHt = P H^T (N x m), exploiting zeros in H.
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < m; k++) {
        let s = 0;
        const hk = k * N, pi = i * N;
        for (let j = 0; j < N; j++) {
          const h = H[hk + j];
          if (h !== 0) s += P[pi + j] * h;
        }
        PHt[i * m + k] = s;
      }
    }
    for (let a = 0; a < m; a++) {
      for (let b = 0; b < m; b++) {
        let s = Rm[a * m + b];
        for (let j = 0; j < N; j++) s += H[a * N + j] * PHt[j * m + b];
        S[a * m + b] = s;
      }
    }
    // Cholesky of S; NIS = r^T S^-1 r.
    const L = this.Sinv;
    for (let i = 0; i < m * m; i++) L[i] = 0.5 * (S[i] + S[(i % m) * m + ((i / m) | 0)]);
    if (!cholesky(L, m)) return -1;
    const y = this.col6;
    for (let i = 0; i < m; i++) y[i] = r[i];
    choleskySolve(L, m, y);
    let nis = 0;
    for (let i = 0; i < m; i++) nis += r[i] * y[i];
    if (dryRun) return nis;
    // K = PHt S^-1: solve S K^T = PHt^T column by column.
    const K = this.K;
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < m; k++) y[k] = PHt[i * m + k];
      choleskySolve(L, m, y);
      for (let k = 0; k < m; k++) K[i * m + k] = y[k];
    }
    const dx = this.dx;
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let k = 0; k < m; k++) s += K[i * m + k] * r[k];
      dx[i] = s;
    }
    // P -= K (H P) = K PHt^T
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let k = 0; k < m; k++) s += K[i * m + k] * PHt[j * m + k];
        P[i * N + j] -= s;
      }
    }
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const v = 0.5 * (P[i * N + j] + P[j * N + i]);
        P[i * N + j] = v; P[j * N + i] = v;
      }
      if (P[i * N + i] < 1e-15) P[i * N + i] = 1e-15;
    }
    this.inject(dx);
    return nis;
  }

  private inject(dx: Float64Array): void {
    const x = this.x;
    quatIntegrateLeft(this.qwc, dx[TH], dx[TH + 1], dx[TH + 2]);
    quatIntegrateLeft(this.qwt, dx[TT], dx[TT + 1], dx[TT + 2]);
    for (let i = 0; i < 3; i++) {
      x[14 + i] += dx[PT + i];
      x[4 + i] += dx[PC + i];
      x[7 + i] += dx[VC + i];
      x[17 + i] += dx[BG + i];
    }
    if (this.tdActive) x[20] += Math.max(-0.02, Math.min(0.02, dx[TD]));
  }

  /** Weak tilt update from the accelerometer when it reads ~1 g. */
  updateGravity(ax: number, ay: number, az: number): boolean {
    const n = Math.hypot(ax, ay, az);
    if (Math.abs(n - 9.81) > 0.3) return false;
    const fx = ax / n, fy = ay / n, fz = az / n;
    const Rwc = mat3FromQuat(this.qwc, this.Rwc);
    // Predicted up in camera axes: R_wc^T e_z = third row of R_wc.
    const hx = Rwc[6], hy = Rwc[7], hz = Rwc[8];
    const r = this.r6;
    r[0] = fx - hx; r[1] = fy - hy; r[2] = fz - hz;
    const H = this.H;
    H.fill(0, 0, 3 * N);
    // d(R^T e_z)/dth = R^T [e_z]x ; [e_z]x = [[0,-1,0],[1,0,0],[0,0,0]].
    for (let i = 0; i < 3; i++) {
      H[i * N + TH] = Rwc[3 + i];
      H[i * N + TH + 1] = -Rwc[i];
      H[i * N + TH + 2] = 0;
    }
    const s2 = this.o.gravityNoise * this.o.gravityNoise;
    const Rn = this.S3g;
    Rn.fill(0);
    Rn[0] = Rn[4] = Rn[8] = s2;
    return this.kalman(H, r, Rn, 3, false) >= 0;
  }

  /**
   * Pose of the target in the camera at camera time tCam. The camera pose is
   * re-integrated with the gyro from the nearest history entry (matching the
   * filter's own propagation); the target uses the latest (static) estimate.
   */
  poseAt(tCam: number, Rout: Float64Array, tout: Float64Array, imu: ImuBuffer | null = null, maxExtrapolate = 0.1): boolean {
    if (!this.initialized) return false;
    let tau = tCam + this.x[20];
    const q = this.qa, p = this.v3, v = this.pv;
    const src = this.o.useImu && imu && imu.count > 0 ? imu : null;
    let t0: number, b0: number, b1: number, b2: number;
    let lo = tau >= this.t ? -2 : this.findEntry(tau);
    if (lo === -2 || this.histCount === 0) {
      const x = this.x;
      q.set(this.qwc);
      p[0] = x[4]; p[1] = x[5]; p[2] = x[6];
      v[0] = x[7]; v[1] = x[8]; v[2] = x[9];
      b0 = x[17]; b1 = x[18]; b2 = x[19];
      t0 = this.t;
      this.w0.set(this.wRaw);
    } else {
      // Entry layout: [t, q_wc(4), p_wc(3), v_wc(3), q_wt(4), p_wt(3), b_g(3), t_d, P].
      const h = this.hist;
      // Prefer a slightly newer entry when it is closer (see PosePredictor.poseAt).
      if (lo >= 0 && lo < this.histCount - 1) {
        const tb = h[this.histOff(lo + 1)];
        if (tb - tau < tau - h[this.histOff(lo)] && tb - tau < NEAR_ENTRY) lo++;
      }
      const o = this.histOff(Math.max(0, lo));
      t0 = h[o];
      if (lo < 0) tau = t0;
      q.set(h.subarray(o + 1, o + 5));
      p[0] = h[o + 5]; p[1] = h[o + 6]; p[2] = h[o + 7];
      v[0] = h[o + 8]; v[1] = h[o + 9]; v[2] = h[o + 10];
      b0 = h[o + 18]; b1 = h[o + 19]; b2 = h[o + 20];
      if (src) rateAt(src, t0, this.w0);
      else this.w0.fill(0);
    }
    const tEnd = Math.min(tau, Math.max(t0, this.t) + maxExtrapolate);
    this.bTmp[0] = b0; this.bTmp[1] = b1; this.bTmp[2] = b2;
    integrateCamera(q, p, v, this.bTmp, t0, this.w0, tEnd, src, this.wTmp);
    const Rwc = mat3FromQuat(q, this.Rwc);
    const Rwt = mat3FromQuat(this.qwt, this.Rwt);
    mat3MulAtB(Rwc, Rwt, Rout);
    const x = this.x;
    mat3TVec(Rwc, x[14] - p[0], x[15] - p[1], x[16] - p[2], tout);
    return true;
  }

  /** Index of the newest history entry with time <= tau (-1 if none). */
  private findEntry(tau: number): number {
    const h = this.hist;
    if (this.histCount === 0 || h[this.histOff(0)] > tau) return -1;
    let lo = 0, hi = this.histCount - 1;
    while (lo < hi) {
      const m = (lo + hi + 1) >> 1;
      if (h[this.histOff(m)] <= tau) lo = m;
      else hi = m - 1;
    }
    return lo;
  }

  /**
   * Compact state for the render-side PosePredictor: camera history since
   * `since` (IMU time, plus one earlier entry) and the current state last.
   */
  exportSnapshot(since: number): Float64Array {
    const h = this.hist;
    let first = this.findEntry(since);
    if (first < 0) first = 0;
    const n = this.histCount - first + 1;
    const out = new Float64Array(SNAP_HEADER + n * SNAP_ENTRY);
    out[0] = n;
    out[1] = this.x[20];
    out.set(this.qwt, 2);
    out.set(this.pwt, 6);
    out.set(this.wRaw, 9);
    let o = SNAP_HEADER;
    for (let i = first; i < this.histCount; i++, o += SNAP_ENTRY) {
      const e = this.histOff(i);
      out[o] = h[e];
      out.set(h.subarray(e + 1, e + 11), o + 1); // q(4) p(3) v(3)
      out.set(h.subarray(e + 18, e + 21), o + 11); // b_g
    }
    out[o] = this.t;
    out.set(this.x.subarray(0, 10), o + 1);
    out.set(this.x.subarray(17, 20), o + 11);
    return out;
  }

  /** 6x6 covariance of the relative pose [dth_ct, dt_ct] at the current state. */
  relativeCovariance(out: Float64Array): Float64Array {
    const J = this.relativeJacobian(this.HP);
    const P = this.P;
    for (let a = 0; a < 6; a++) for (let b = 0; b < 6; b++) {
      let s = 0;
      for (let i = 0; i < N; i++) {
        const ja = J[a * N + i];
        if (ja === 0) continue;
        for (let j = 0; j < N; j++) s += ja * P[i * N + j] * J[b * N + j];
      }
      out[a * 6 + b] = s;
    }
    return out;
  }

  /** 6xN Jacobian of the relative pose [dth_ct, dt_ct] w.r.t. the error state, into J. */
  private relativeJacobian(J: Float64Array): Float64Array {
    const Rwc = mat3FromQuat(this.qwc, this.Rwc);
    const x = this.x;
    const dxw = x[14] - x[4], dyw = x[15] - x[5], dzw = x[16] - x[6];
    J.fill(0);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const RT = Rwc[j * 3 + i];
        J[i * N + TH + j] = -RT; J[i * N + TT + j] = RT;
        J[(3 + i) * N + PT + j] = RT; J[(3 + i) * N + PC + j] = -RT;
      }
      const a0 = Rwc[i], a1 = Rwc[3 + i], a2 = Rwc[6 + i];
      J[(3 + i) * N + TH] = a1 * dzw - a2 * dyw;
      J[(3 + i) * N + TH + 1] = a2 * dxw - a0 * dzw;
      J[(3 + i) * N + TH + 2] = a0 * dyw - a1 * dxw;
    }
    return J;
  }

  // History ring: [t, x(21), P(361)] per entry.
  private histOff(i: number): number {
    return ((this.histHead + i) % this.cap) * ENTRY;
  }

  private pushHistory(): void {
    let o: number;
    if (this.histCount > 0 && this.hist[this.histOff(this.histCount - 1)] === this.t) {
      // Same time (e.g. the state just updated by vision): replace, so lookups never find the stale one.
      o = this.histOff(this.histCount - 1);
    } else if (this.histCount < this.cap) {
      o = this.histOff(this.histCount);
      this.histCount++;
    } else {
      o = this.histHead * ENTRY;
      this.histHead = (this.histHead + 1) % this.cap;
    }
    const h = this.hist;
    h[o] = this.t;
    h.set(this.x, o + 1);
    h.set(this.P, o + 1 + NX);
  }

  /** Restore the newest history entry at or before tau and drop later ones. */
  private rewindTo(tau: number): boolean {
    const h = this.hist;
    const lo = this.findEntry(tau);
    if (lo < 0) return false;
    const o = this.histOff(lo);
    this.t = h[o];
    this.x.set(h.subarray(o + 1, o + 1 + NX));
    this.P.set(h.subarray(o + 1 + NX, o + ENTRY));
    this.histCount = lo + 1;
    return true;
  }

  /** Re-read the raw gyro rate at the current time from the IMU buffer (after a rewind). */
  private syncRate(imu: ImuBuffer | null): void {
    if (!this.o.useImu || !imu || imu.count === 0) return;
    rateAt(imu, this.t, this.wRaw);
  }

  /** Time of the oldest history entry (measurements older than this are dropped). */
  oldestTime(): number {
    return this.histCount ? this.hist[this.histOff(0)] : Infinity;
  }
}

/** World-from-camera rotation with world z along `up` (camera axes); yaw from the camera's forward axis. */
function worldFromUp(up: ArrayLike<number>, Rwc: Float64Array): Float64Array {
  const n = Math.hypot(up[0], up[1], up[2]);
  const zx = up[0] / n, zy = up[1] / n, zz = up[2] / n;
  // World y: camera forward (0,0,1) projected onto the horizontal plane.
  let yx = -zz * zx, yy = -zz * zy, yz = 1 - zz * zz;
  let yn = Math.hypot(yx, yy, yz);
  if (yn < 1e-3) { yx = 1 - zx * zx; yy = -zx * zy; yz = -zx * zz; yn = Math.hypot(yx, yy, yz); }
  yx /= yn; yy /= yn; yz /= yn;
  const xx = yy * zz - yz * zy, xy = yz * zx - yx * zz, xz = yx * zy - yy * zx;
  // R_cw columns are world axes in camera coords: [x y z]; R_wc = R_cw^T (rows).
  Rwc[0] = xx; Rwc[1] = xy; Rwc[2] = xz;
  Rwc[3] = yx; Rwc[4] = yy; Rwc[5] = yz;
  Rwc[6] = zx; Rwc[7] = zy; Rwc[8] = zz;
  return Rwc;
}
