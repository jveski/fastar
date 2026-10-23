// Shared test fixtures.
import { intrinsicsFromFov, type Intrinsics } from '../src/vision/camera';
import { makePoster } from '../src/sim/poster';
import type { GrayImage } from '../src/image/gray';

export const K: Intrinsics = intrinsicsFromFov(640, 480, 63);
export const POSTER_W_M = 0.2;

let poster: GrayImage | undefined;
export function getPoster(): GrayImage {
  poster ??= makePoster(640, 480, 11);
  return poster;
}

/** Target pixel -> target metres. */
export function texelToMetres(img: GrayImage, widthM: number, u: number, v: number): [number, number] {
  const s = widthM / img.width;
  return [(u - (img.width - 1) / 2) * s, (v - (img.height - 1) / 2) * s];
}
