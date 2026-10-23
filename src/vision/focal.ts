// Focal-length self-calibration from plane homographies with a known
// principal point and square pixels (Zhang's constraints): for each
// homography (columns h1, h2 after centring on the principal point),
//   (h11 h12 + h21 h22) / f^2 + h31 h32 = 0
//   (h11^2 + h21^2 - h12^2 - h22^2) / f^2 + (h31^2 - h32^2) = 0
import { homographyDLT } from './homography';
import type { TargetData } from '../target/format';
import type { TargetTracker } from '../track/tracker';
import { median } from '../math/select';

export interface FocalSample {
  f: number;
  /** Tilt of the target plane relative to the optical axis (rad). */
  tilt: number;
}

/** Focal length from one homography mapping plane coords to pixels, or NaN if degenerate. */
export function focalFromHomography(H: ArrayLike<number>, cx: number, cy: number): number {
  // Centre: Hc = T H with T = [[1,0,-cx],[0,1,-cy],[0,0,1]].
  const h11 = H[0] - cx * H[6], h12 = H[1] - cx * H[7];
  const h21 = H[3] - cy * H[6], h22 = H[4] - cy * H[7];
  const h31 = H[6], h32 = H[7];
  // Normalize scale so both equations are well balanced.
  const s = 1 / Math.hypot(h11, h12, h21, h22);
  const a11 = h11 * s, a12 = h12 * s, a21 = h21 * s, a22 = h22 * s, a31 = h31 * s, a32 = h32 * s;
  const A1 = a11 * a12 + a21 * a22, b1 = -a31 * a32;
  const A2 = a11 * a11 + a21 * a21 - a12 * a12 - a22 * a22, b2 = -(a31 * a31 - a32 * a32);
  const den = A1 * A1 + A2 * A2;
  if (den < 1e-12) return NaN;
  const inv2 = (A1 * b1 + A2 * b2) / den; // 1/f^2
  return inv2 > 0 ? 1 / Math.sqrt(inv2) : NaN;
}

/** Collects diverse homographies during tracking and estimates f once there are enough. */
export class FocalCalibrator {
  readonly samples: FocalSample[] = [];
  private frame = 0;
  private readonly normals: number[][] = [];
  minSamples = 5;
  minTilt = (20 * Math.PI) / 180;

  constructor(readonly width: number, readonly height: number) {}

  /** H maps target base pixels to frame pixels. */
  addHomography(H: ArrayLike<number>, _t?: TargetData, normal?: number[]): void {
    const cx = (this.width - 1) / 2, cy = (this.height - 1) / 2;
    const f = focalFromHomography(H, cx, cy);
    if (!Number.isFinite(f)) return;
    // Tilt from the plane normal r1 x r2 of K^-1 H (using this view's f).
    let tilt: number;
    if (normal) tilt = Math.acos(Math.min(1, Math.abs(normal[2])));
    else {
      const r1 = [(H[0] - cx * H[6]) / f, (H[3] - cy * H[6]) / f, H[6]];
      const r2 = [(H[1] - cx * H[7]) / f, (H[4] - cy * H[7]) / f, H[7]];
      const n = [r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2], r1[0] * r2[1] - r1[1] * r2[0]];
      tilt = Math.acos(Math.min(1, Math.abs(n[2]) / Math.hypot(n[0], n[1], n[2])));
    }
    if (tilt < this.minTilt) return;
    if (normal) {
      for (const n of this.normals) if (n[0] * normal[0] + n[1] * normal[1] + n[2] * normal[2] > Math.cos((10 * Math.PI) / 180)) return;
      this.normals.push(normal);
    }
    this.samples.push({ f, tilt });
  }

  /** Every 15th frame, fit a homography to the tracker's correspondences (independent of K). */
  addTracked(tr: TargetTracker, t: TargetData): void {
    if (++this.frame % 15 !== 0 || tr.n < 30) return;
    const n = tr.n;
    const src = new Float64Array(2 * n), dst = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      const j = tr.idx[i];
      src[2 * i] = tr.obj[2 * j]; src[2 * i + 1] = tr.obj[2 * j + 1];
      dst[2 * i] = tr.pos[2 * i]; dst[2 * i + 1] = tr.pos[2 * i + 1];
    }
    const H = new Float64Array(9);
    if (!homographyDLT(src, dst, n, H)) return;
    // Plane normal in camera axes (third column of R) for diversity.
    this.addHomography(H, t, [tr.R[2], tr.R[5], tr.R[8]]);
  }

  /** Median focal over diverse views once enough agree (spread < 5%). */
  estimate(): { f: number; samples: number } | null {
    if (this.samples.length < this.minSamples) return null;
    const fs = Float64Array.from(this.samples.map((s) => s.f));
    const m = median(fs.slice(), fs.length);
    const dev = Float64Array.from(fs, (f) => Math.abs(f - m));
    const mad = median(dev, dev.length);
    if (mad / m > 0.05) return null;
    return { f: m, samples: this.samples.length };
  }
}
