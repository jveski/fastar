// Procedural looping motion-graphics clip for tests: two shots with a cut
// between them (and at the loop seam), a channel logo that stays put across
// both, a scrolling ticker, a sliding card, a fading badge and a panel that
// slides in and stops. Flat fills with little noise, like rendered graphics.
import { createGray, type GrayImage } from '../image/gray';
import { makePoster } from './poster';

export interface MotionClipOptions {
  width?: number;
  height?: number;
  fps?: number;
  /** Seconds. */
  duration?: number;
  /** Cut position as a fraction of the duration. */
  cutAt?: number;
  seed?: number;
}

/** A looping clip the sim can show on a screen. */
export interface VideoClip {
  width: number;
  height: number;
  fps: number;
  frameCount: number;
  duration: number;
  /** Frame i (greyscale, cached). */
  frame(i: number): GrayImage;
}

export interface MotionClip extends VideoClip {
  /** Frames where a new shot starts (besides frame 0). */
  cuts: number[];
  /** Where the sliding card's top-left corner is at frame i (base pixels), or null outside shot A. */
  cardAt(i: number): [number, number] | null;
}

export function makeMotionClip(o: MotionClipOptions = {}): MotionClip {
  const W = o.width ?? 640, H = o.height ?? 360, fps = o.fps ?? 30;
  const duration = o.duration ?? 6;
  const N = Math.round(duration * fps);
  const cut = Math.round(N * (o.cutAt ?? 0.5));
  const seed = o.seed ?? 3;
  const layoutA = makePoster(W, H, seed, 1.5, 90);
  const layoutB = makePoster(W, H, seed + 50, 1.5, 90);
  const logo = makePoster(Math.round(W * 0.16), Math.round(W * 0.16), seed + 7, 1.5, 25);
  const card = makePoster(Math.round(W * 0.26), Math.round(H * 0.32), seed + 11, 1.5, 30);
  const badge = makePoster(Math.round(W * 0.14), Math.round(H * 0.14), seed + 13, 1.5, 12);
  const panel = makePoster(Math.round(W * 0.3), Math.round(H * 0.5), seed + 17, 1.5, 40);
  const tickerH = Math.round(H * 0.12);
  const ticker = makePoster(W * 3, tickerH, seed + 23, 1.5, 160);
  const tickerSpeed = W / (2.5 * fps); // px per frame
  const cache = new Map<number, GrayImage>();

  const ease = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
  const cardAt = (i: number): [number, number] | null => {
    if (i >= cut) return null;
    const s = ease((i / fps - 0.3) / Math.max(0.5, cut / fps - 0.8));
    return [W * 0.05 + s * (W * 0.55), H * 0.3];
  };

  const frame = (i: number): GrayImage => {
    i = ((i % N) + N) % N;
    let img = cache.get(i);
    if (img) return img;
    img = createGray(W, H);
    const shotA = i < cut;
    img.data.set((shotA ? layoutA : layoutB).data);
    const ts = (shotA ? i : i - cut) / fps; // time in shot
    if (shotA) {
      const c = cardAt(i)!;
      blit(img, card, c[0], c[1], 1);
      blit(img, badge, W * 0.08, H * 0.06, ease(ts / 0.6 - 0.5));
    } else {
      const s = ease(ts / 0.8);
      blit(img, panel, W - s * (W * 0.36), H * 0.2, 1);
    }
    // Ticker scrolls continuously through both shots and wraps with the loop.
    const off = (i * tickerSpeed) % (ticker.width - W);
    blit(img, ticker, -off, H - tickerH, 1);
    blit(img, logo, W - logo.width - W * 0.03, H * 0.04, 1);
    cache.set(i, img);
    return img;
  };

  return { width: W, height: H, fps, frameCount: N, duration: N / fps, cuts: [cut], frame, cardAt };
}

/** Composite `src` with its top-left at subpixel (x, y), bilinear with soft edges. */
export function blit(dst: GrayImage, src: GrayImage, x: number, y: number, alpha: number): void {
  if (alpha <= 0) return;
  const dw = dst.width, dh = dst.height, sw = src.width, sh = src.height, d = dst.data, s = src.data;
  const x0 = Math.max(0, Math.floor(x) - 1), x1 = Math.min(dw - 1, Math.ceil(x + sw));
  const y0 = Math.max(0, Math.floor(y) - 1), y1 = Math.min(dh - 1, Math.ceil(y + sh));
  for (let py = y0; py <= y1; py++) {
    const v = py - y, vi = Math.floor(v), av = v - vi;
    for (let px = x0; px <= x1; px++) {
      const u = px - x, ui = Math.floor(u), au = u - ui;
      let val = 0, cov = 0;
      for (let j = 0; j < 2; j++) {
        const yy = vi + j;
        if (yy < 0 || yy >= sh) continue;
        const wy = j ? av : 1 - av;
        for (let k = 0; k < 2; k++) {
          const xx = ui + k;
          if (xx < 0 || xx >= sw) continue;
          const w = wy * (k ? au : 1 - au);
          val += w * s[yy * sw + xx];
          cov += w;
        }
      }
      if (cov <= 0) continue;
      const a = alpha * cov;
      const o = py * dw + px;
      d[o] = Math.round(d[o] * (1 - a) + (val / cov) * a);
    }
  }
}
