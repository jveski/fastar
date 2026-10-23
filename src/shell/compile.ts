// In-browser target compilation from images.
import { rgbaToGray, type GrayImage } from '../image/gray';
import { compileTarget, type CompileOptions } from '../target/compiler';
import { encodeTargets, type TargetData } from '../target/format';

export type ImageInput = Blob | ImageBitmap | HTMLImageElement | HTMLCanvasElement | OffscreenCanvas | ImageData;

/** Decode any browser image source to greyscale. */
export async function imageToGray(src: ImageInput): Promise<GrayImage> {
  if (typeof ImageData !== 'undefined' && src instanceof ImageData) return rgbaToGray(src.data, src.width, src.height);
  const bmp = src instanceof Blob ? await createImageBitmap(src) : (src as CanvasImageSource & { width: number; height: number });
  const w = (bmp as { naturalWidth?: number }).naturalWidth || bmp.width;
  const h = (bmp as { naturalHeight?: number }).naturalHeight || bmp.height;
  const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
  ctx.drawImage(bmp as CanvasImageSource, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h);
  return rgbaToGray(d.data, w, h);
}

/** Compile one image into target data (see the quality report in meta.quality). */
export async function compileImage(src: ImageInput, o: CompileOptions): Promise<TargetData> {
  return compileTarget(await imageToGray(src), o);
}

/** Compile several images into a targets file buffer for ImageTracker.create. */
export async function compileImages(list: Array<{ image: ImageInput } & CompileOptions>): Promise<{ buffer: ArrayBuffer; targets: TargetData[] }> {
  const targets: TargetData[] = [];
  for (const it of list) targets.push(await compileImage(it.image, it));
  return { buffer: encodeTargets(targets), targets };
}
