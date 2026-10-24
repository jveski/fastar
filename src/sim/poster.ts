// Procedural target textures with plenty of corners and texture, so tests
// need no image assets.
import { createGray, type GrayImage } from '../image/gray';
import { Rng } from '../math/rng';

export function makePoster(width: number, height: number, seed = 7, noise = 6, shapes = 140): GrayImage {
  const rng = new Rng(seed);
  const img = createGray(width, height);
  const d = img.data;
  // Smooth background gradient.
  const g0 = rng.range(60, 200), gx = rng.range(-60, 60), gy = rng.range(-60, 60);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      d[y * width + x] = clamp(g0 + (gx * x) / width + (gy * y) / height);
    }
  }
  const scale = Math.min(width, height);
  for (let s = 0; s < shapes; s++) {
    const kind = rng.int(4);
    const val = rng.range(0, 255);
    const cx = rng.range(0, width), cy = rng.range(0, height);
    const r = scale * rng.range(0.015, 0.12);
    if (kind === 0) {
      const a = rng.range(0, Math.PI);
      fillPolygon(img, rectCorners(cx, cy, r, r * rng.range(0.3, 1.5), a), val);
    } else if (kind === 1) {
      fillCircle(img, cx, cy, r * 0.7, val);
    } else if (kind === 2) {
      const pts: number[] = [];
      for (let k = 0; k < 3; k++) {
        const a = rng.range(0, 2 * Math.PI);
        pts.push(cx + r * Math.cos(a), cy + r * Math.sin(a));
      }
      fillPolygon(img, pts, val);
    } else {
      // "Text": rows of short bars.
      const rows = 2 + rng.int(4);
      for (let j = 0; j < rows; j++) {
        let x = cx;
        const y = cy + j * r * 0.35;
        const cols = 3 + rng.int(8);
        for (let k = 0; k < cols; k++) {
          const w = r * rng.range(0.08, 0.25);
          fillPolygon(img, [x, y, x + w, y, x + w, y + r * 0.2, x, y + r * 0.2], val);
          x += w * rng.range(1.3, 2.2);
        }
      }
    }
  }
  // Fine noise so textured areas exist at every scale.
  if (noise > 0) for (let i = 0; i < d.length; i++) d[i] = clamp(d[i] + rng.gauss() * noise);
  return img;
}

/** A large, lower-contrast texture used as the wall around the target. */
export function makeBackground(size: number, seed = 99): GrayImage {
  const img = makePoster(size, size, seed);
  const d = img.data;
  for (let i = 0; i < d.length; i++) d[i] = 90 + ((d[i] - 128) * 0.5) | 0;
  return img;
}

function clamp(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v | 0;
}

function rectCorners(cx: number, cy: number, w: number, h: number, a: number): number[] {
  const c = Math.cos(a), s = Math.sin(a);
  const out: number[] = [];
  for (const [u, v] of [[-w, -h], [w, -h], [w, h], [-w, h]]) out.push(cx + c * u - s * v, cy + s * u + c * v);
  return out;
}

function fillCircle(img: GrayImage, cx: number, cy: number, r: number, val: number): void {
  const { width, height, data } = img;
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(height - 1, Math.ceil(cy + r)); y++) {
    for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(width - 1, Math.ceil(cx + r)); x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) data[y * width + x] = val;
    }
  }
}

function fillPolygon(img: GrayImage, pts: number[], val: number): void {
  const { width, height, data } = img;
  const n = pts.length / 2;
  let minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) { minY = Math.min(minY, pts[2 * i + 1]); maxY = Math.max(maxY, pts[2 * i + 1]); }
  const xs: number[] = [];
  for (let y = Math.max(0, Math.floor(minY)); y <= Math.min(height - 1, Math.ceil(maxY)); y++) {
    xs.length = 0;
    const yc = y + 0.5;
    for (let i = 0; i < n; i++) {
      const x0 = pts[2 * i], y0 = pts[2 * i + 1];
      const x1 = pts[(2 * i + 2) % (2 * n)], y1 = pts[(2 * i + 3) % (2 * n)];
      if ((y0 <= yc && y1 > yc) || (y1 <= yc && y0 > yc)) xs.push(x0 + ((yc - y0) / (y1 - y0)) * (x1 - x0));
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      for (let x = Math.max(0, Math.ceil(xs[k] - 0.5)); x <= Math.min(width - 1, Math.floor(xs[k + 1] - 0.5)); x++) {
        data[y * width + x] = val;
      }
    }
  }
}
