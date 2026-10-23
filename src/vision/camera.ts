// Pinhole camera model. Vision code uses the OpenCV convention: camera x
// right, y down, z forward; target x right, y down, z into the target, origin
// at the target centre, units in metres.

export interface Intrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
}

export const DEFAULT_HFOV_DEG = 63;

/**
 * Horizontal FOV of the rear main camera for common devices. Browsers expose
 * little model information, so matching is by user-agent substring.
 */
export const FOV_TABLE: Array<{ match: RegExp; hfovDeg: number }> = [
  { match: /iPhone/, hfovDeg: 65 },
  { match: /iPad/, hfovDeg: 60 },
  { match: /Pixel [6-9]/, hfovDeg: 68 },
  { match: /Pixel/, hfovDeg: 66 },
  { match: /SM-S9\d\d|SM-G99\d/, hfovDeg: 70 },
  { match: /SM-A\d\d\d/, hfovDeg: 66 },
];

export function lookupHfov(userAgent: string): number {
  for (const e of FOV_TABLE) if (e.match.test(userAgent)) return e.hfovDeg;
  return DEFAULT_HFOV_DEG;
}

/** Intrinsics with the principal point at the image centre. hfov applies to the long side. */
export function intrinsicsFromFov(width: number, height: number, hfovDeg = DEFAULT_HFOV_DEG): Intrinsics {
  const long = Math.max(width, height);
  const f = long / 2 / Math.tan((hfovDeg * Math.PI) / 360);
  return intrinsicsFromFocal(width, height, f);
}

export function intrinsicsFromFocal(width: number, height: number, f: number): Intrinsics {
  return { fx: f, fy: f, cx: (width - 1) / 2, cy: (height - 1) / 2, width, height };
}

export type FitMode = 'cover' | 'contain' | 'fill';

/**
 * OpenGL projection (column-major 4x4) for a viewport showing the processed
 * frame with CSS object-fit semantics, so overlays line up with the video.
 */
export function glProjection(
  K: Intrinsics,
  viewW: number,
  viewH: number,
  near = 0.01,
  far = 100,
  fit: FitMode = 'cover',
  out: Float32Array = new Float32Array(16),
): Float32Array {
  let sx: number, sy: number;
  if (fit === 'fill') {
    sx = viewW / K.width;
    sy = viewH / K.height;
  } else {
    const s = fit === 'cover' ? Math.max(viewW / K.width, viewH / K.height) : Math.min(viewW / K.width, viewH / K.height);
    sx = sy = s;
  }
  const ox = (viewW - K.width * sx) / 2, oy = (viewH - K.height * sy) / 2;
  out.fill(0);
  out[0] = (2 * sx * K.fx) / viewW;
  out[5] = (2 * sy * K.fy) / viewH;
  out[8] = -((2 * (ox + (K.cx + 0.5) * sx)) / viewW - 1);
  out[9] = -(1 - (2 * (oy + (K.cy + 0.5) * sy)) / viewH);
  out[10] = -(far + near) / (far - near);
  out[11] = -1;
  out[14] = (-2 * far * near) / (far - near);
  return out;
}

/**
 * Convert an OpenCV target-in-camera pose (R row-major, t) to a column-major
 * GL model-view matrix for a target frame with y up and z out of the target.
 * M = C * [R t] * C with C = diag(1, -1, -1).
 */
export function poseToGl(R: ArrayLike<number>, t: ArrayLike<number>, out: Float32Array = new Float32Array(16)): Float32Array {
  // C R C flips signs of rows 1,2 and columns 1,2: entry (i,j) * c_i * c_j.
  out[0] = R[0]; out[4] = -R[1]; out[8] = -R[2]; out[12] = t[0];
  out[1] = -R[3]; out[5] = R[4]; out[9] = R[5]; out[13] = -t[1];
  out[2] = -R[6]; out[6] = R[7]; out[10] = R[8]; out[14] = -t[2];
  out[3] = 0; out[7] = 0; out[11] = 0; out[15] = 1;
  return out;
}
