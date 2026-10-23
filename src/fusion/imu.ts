/**
 * Ring buffer of gyro (and optionally accelerometer) samples in camera axes.
 * Times are seconds on the IMU clock. Fixed capacity, no allocation.
 */
export class ImuBuffer {
  readonly t: Float64Array;
  readonly w: Float64Array;
  readonly a: Float64Array;
  readonly hasAccel: Uint8Array;
  private head = 0; // index of oldest
  count = 0;

  constructor(readonly capacity = 256) {
    this.t = new Float64Array(capacity);
    this.w = new Float64Array(capacity * 3);
    this.a = new Float64Array(capacity * 3);
    this.hasAccel = new Uint8Array(capacity);
  }

  /** Append a sample. Out-of-order samples older than the newest are dropped. */
  push(t: number, wx: number, wy: number, wz: number, ax = NaN, ay = NaN, az = NaN): boolean {
    if (this.count > 0 && t <= this.timeAt(this.count - 1)) return false;
    let slot: number;
    if (this.count < this.capacity) {
      slot = (this.head + this.count) % this.capacity;
      this.count++;
    } else {
      slot = this.head;
      this.head = (this.head + 1) % this.capacity;
    }
    this.t[slot] = t;
    this.w[3 * slot] = wx; this.w[3 * slot + 1] = wy; this.w[3 * slot + 2] = wz;
    const acc = ax === ax;
    this.hasAccel[slot] = acc ? 1 : 0;
    if (acc) { this.a[3 * slot] = ax; this.a[3 * slot + 1] = ay; this.a[3 * slot + 2] = az; }
    return true;
  }

  /** Physical slot of logical index i (0 = oldest). */
  slot(i: number): number {
    return (this.head + i) % this.capacity;
  }

  timeAt(i: number): number {
    return this.t[(this.head + i) % this.capacity];
  }

  get latestTime(): number {
    return this.count ? this.timeAt(this.count - 1) : -Infinity;
  }

  get oldestTime(): number {
    return this.count ? this.timeAt(0) : Infinity;
  }

  /** Logical index of the first sample with time > t (count if none). */
  firstAfter(t: number): number {
    let lo = 0, hi = this.count;
    while (lo < hi) {
      const m = (lo + hi) >> 1;
      if (this.timeAt(m) > t) hi = m;
      else lo = m + 1;
    }
    return lo;
  }

  /** Multiply every stored sample's axes by `signs` (axis-convention fix). */
  scaleAxes(signs: ArrayLike<number>): void {
    for (let i = 0; i < this.count; i++) {
      const s = this.slot(i);
      for (let k = 0; k < 3; k++) {
        this.w[3 * s + k] *= signs[k];
        this.a[3 * s + k] *= signs[k];
      }
    }
  }

  clear(): void {
    this.head = 0;
    this.count = 0;
  }
}
