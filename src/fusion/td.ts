// Camera-to-IMU time offset (t_imu = t_cam + t_d) by cross-correlating
// rotation rates from vision with rotation rates from the gyro, plus a
// per-axis sign check (wrong axis conventions show up as negative
// correlation on that axis).
import { ImuBuffer } from './imu';
import { rateAt } from './predict';
import { mat3MulABt, so3Log, mat3, vec3 } from '../math/rot';

export interface TdOptions {
  /** Seconds of tracked vision needed before estimating. */
  window: number;
  /** Search range for the offset (s). */
  maxOffset: number;
  /** Search step (s). */
  step: number;
  /** Minimum RMS rotation speed over the window (rad/s); hand tremor is enough. */
  minMotion: number;
  /** Minimum peak correlation (pooled over axes) to accept. */
  minCorrelation: number;
}

export const DEFAULT_TD: TdOptions = { window: 2, maxOffset: 0.15, step: 0.001, minMotion: 0.1, minCorrelation: 0.65 };

export interface TdResult {
  td: number;
  correlation: number;
  /** Multiply gyro (and accel) axes by these to fix the convention. */
  axisSigns: [number, number, number];
}

export class TdEstimator {
  readonly o: TdOptions;
  private readonly tv: number[] = []; // mid times (camera clock)
  private readonly dtv: number[] = []; // frame interval
  private readonly wv: number[] = []; // vision rate vectors (x3)
  private prevT = NaN;
  private readonly prevR = mat3();
  private readonly d = mat3();
  private readonly w = vec3();
  private readonly g = vec3();

  constructor(opts: Partial<TdOptions> = {}) {
    this.o = { ...DEFAULT_TD, ...opts };
  }

  reset(): void {
    this.tv.length = this.dtv.length = this.wv.length = 0;
    this.prevT = NaN;
  }

  /** Feed the tracked target rotation R_ct at camera time t (consecutive frames). */
  addVision(t: number, Rct: ArrayLike<number>): void {
    const dt = t - this.prevT;
    if (dt > 0 && dt < 0.1) {
      // Camera rotation between frames, in camera axes: w dt = Log(R_ct(k-1) R_ct(k)^T).
      mat3MulABt(this.prevR, Rct, this.d);
      so3Log(this.d, this.w);
      this.tv.push(this.prevT + dt / 2);
      this.dtv.push(dt);
      this.wv.push(this.w[0] / dt, this.w[1] / dt, this.w[2] / dt);
    }
    for (let i = 0; i < 9; i++) this.prevR[i] = Rct[i];
    this.prevT = t;
    // Keep a bounded window (2x the needed span).
    while (this.tv.length > 2 && this.tv[this.tv.length - 1] - this.tv[0] > 2 * this.o.window) {
      this.tv.shift(); this.dtv.shift(); this.wv.splice(0, 3);
    }
  }

  /** Break the frame-to-frame chain (e.g. after tracking was lost). */
  gap(): void {
    this.prevT = NaN;
  }

  get span(): number {
    return this.tv.length ? this.tv[this.tv.length - 1] - this.tv[0] : 0;
  }

  /** Try to estimate; returns the result once enough motion has been seen. */
  estimate(imu: ImuBuffer, bias: ArrayLike<number> = [0, 0, 0]): TdResult | null {
    const n = this.tv.length;
    if (n < 20 || this.span < this.o.window) return null;
    const vx = new Float64Array(n), vy = new Float64Array(n), vz = new Float64Array(n);
    let ms = 0;
    for (let i = 0; i < n; i++) {
      vx[i] = this.wv[3 * i]; vy[i] = this.wv[3 * i + 1]; vz[i] = this.wv[3 * i + 2];
      ms += vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
    }
    if (Math.sqrt(ms / n) < this.o.minMotion) return null;
    const { maxOffset, step } = this.o;
    const steps = Math.round(maxOffset / step);
    const corr = new Float64Array(2 * steps + 1);
    const gx = new Float64Array(n), gy = new Float64Array(n), gz = new Float64Array(n);
    let best = -2, bk = 0;
    for (let k = -steps; k <= steps; k++) {
      const off = k * step;
      let valid = true;
      for (let i = 0; i < n; i++) {
        const tm = this.tv[i] + off, h = this.dtv[i] / 2;
        if (tm - h < imu.oldestTime || tm + h > imu.latestTime) { valid = false; break; }
        this.meanRate(imu, tm, h, bias);
        gx[i] = this.g[0]; gy[i] = this.g[1]; gz[i] = this.g[2];
      }
      // Correlation of the rate vectors, per axis and pooled: sharper than
      // speed alone (and steadier under rolling shutter), and blind to axis
      // signs, which are checked below.
      const c = valid ? pooledAbsCorrelation(vx, vy, vz, gx, gy, gz, n) : -2;
      corr[k + steps] = c;
      if (c > best) { best = c; bk = k; }
    }
    if (best < this.o.minCorrelation) return null;
    // Sub-step refinement by a parabola through the peak.
    let td = bk * step;
    const i0 = bk + steps;
    if (i0 > 0 && i0 < corr.length - 1 && corr[i0 - 1] > -2 && corr[i0 + 1] > -2) {
      const a = corr[i0 - 1], b = corr[i0], c = corr[i0 + 1];
      const den = a - 2 * b + c;
      if (den < 0) td += (0.5 * (a - c) / den) * step;
    }
    // Per-axis sign check at the estimated offset.
    for (let i = 0; i < n; i++) {
      this.meanRate(imu, this.tv[i] + td, this.dtv[i] / 2, bias);
      gx[i] = this.g[0]; gy[i] = this.g[1]; gz[i] = this.g[2];
    }
    const signs = [pearson(vx, gx, n), pearson(vy, gy, n), pearson(vz, gz, n)].map((c) => (c < -0.3 ? -1 : 1)) as [number, number, number];
    return { td, correlation: best, axisSigns: signs };
  }

  /** Mean gyro rate over [t - h, t + h] (Simpson), bias removed, into this.g. */
  private meanRate(imu: ImuBuffer, t: number, h: number, bias: ArrayLike<number>): void {
    const a = rateAt(imu, t - h, this.w);
    const ax = a[0], ay = a[1], az = a[2];
    const b = rateAt(imu, t + h, this.w);
    const bx = b[0], by = b[1], bz = b[2];
    const m = rateAt(imu, t, this.w);
    this.g[0] = (ax + 4 * m[0] + bx) / 6 - bias[0];
    this.g[1] = (ay + 4 * m[1] + by) / 6 - bias[1];
    this.g[2] = (az + 4 * m[2] + bz) / 6 - bias[2];
  }
}

/** sum_axes |cov| / sum_axes sqrt(var_a var_b): a correlation in [0, 1] weighted toward the axes that move. */
function pooledAbsCorrelation(
  ax: Float64Array, ay: Float64Array, az: Float64Array, bx: Float64Array, by: Float64Array, bz: Float64Array, n: number,
): number {
  let num = 0, den = 0;
  for (const [a, b] of [[ax, bx], [ay, by], [az, bz]]) {
    let ma = 0, mb = 0;
    for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < n; i++) {
      const x = a[i] - ma, y = b[i] - mb;
      sab += x * y; saa += x * x; sbb += y * y;
    }
    num += Math.abs(sab);
    den += Math.sqrt(saa * sbb);
  }
  return den > 0 ? num / den : 0;
}

function pearson(a: Float64Array, b: Float64Array, n: number): number {
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    sab += x * y; saa += x * x; sbb += y * y;
  }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : 0;
}
