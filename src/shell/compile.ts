// In-browser target compilation from images and videos.
import { createGray, rgbaToGray, type GrayImage } from '../image/gray';
import { compileTarget, type CompileOptions } from '../target/compiler';
import { VideoTargetBuilder, type VideoCompileOptions } from '../target/video-compiler';
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

export type VideoSource = Blob | string | URL | HTMLVideoElement;

export interface CompileVideoOptions extends Omit<VideoCompileOptions, 'fps'> {
  /** Frame rate; measured by playing a few frames if omitted. */
  fps?: number;
  /** Called after each frame with the frames done and the total. */
  onProgress?: (done: number, total: number) => void;
  /** Abort compilation. */
  signal?: AbortSignal;
}

/**
 * Compile a video (file, URL or element) into target data, in the browser.
 * Frames are read by seeking a <video> element frame by frame, so this
 * takes a while (roughly real time); report progress with onProgress.
 */
export async function compileVideo(src: VideoSource, o: CompileVideoOptions): Promise<TargetData> {
  const own = !(typeof HTMLVideoElement !== 'undefined' && src instanceof HTMLVideoElement);
  const video = own ? document.createElement('video') : (src as HTMLVideoElement);
  let url: string | null = null;
  if (own) {
    if (typeof src === 'string' || src instanceof URL) {
      video.crossOrigin = 'anonymous';
      video.src = String(src);
    } else {
      video.src = url = URL.createObjectURL(src as Blob);
    }
  }
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  try {
    if (video.readyState < 1) await once(video, 'loadedmetadata');
    const fps = o.fps ?? (await detectFrameRate(video));
    const duration = video.duration;
    if (!Number.isFinite(duration) || duration <= 0) throw new Error('fastar: video has no duration');
    const total = Math.max(2, Math.round(duration * fps));
    const long = Math.max(video.videoWidth, video.videoHeight), maxDim = o.maxDim ?? 1024;
    const s = long > maxDim ? maxDim / long : 1;
    const W = Math.round(video.videoWidth * s), H = Math.round(video.videoHeight * s);
    const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(W, H) : Object.assign(document.createElement('canvas'), { width: W, height: H });
    const ctx = canvas.getContext('2d', { willReadFrequently: true }) as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
    const builder = new VideoTargetBuilder({ ...o, fps });
    const gray = createGray(W, H);
    for (let i = 0; i < total; i++) {
      if (o.signal?.aborted) throw new Error('fastar: compilation aborted');
      await seek(video, Math.min(duration - 1e-3, (i + 0.5) / fps));
      ctx.drawImage(video, 0, 0, W, H);
      rgbaToGray(ctx.getImageData(0, 0, W, H).data, W, H, gray);
      builder.addFrame(gray);
      o.onProgress?.(i + 1, total);
    }
    return builder.finish();
  } finally {
    if (url) URL.revokeObjectURL(url);
  }
}

const COMMON_RATES = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60];

/** Frame rate from the media times of a few played frames, snapped to a common rate. */
export async function detectFrameRate(video: HTMLVideoElement, frames = 12): Promise<number> {
  const v = video as HTMLVideoElement & { requestVideoFrameCallback?: (cb: (now: number, m: { mediaTime: number }) => void) => number };
  if (typeof v.requestVideoFrameCallback !== 'function') return 30;
  const times: number[] = [];
  const start = video.currentTime;
  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(resolve, 3000);
    const step = (_: number, m: { mediaTime: number }) => {
      times.push(m.mediaTime);
      if (times.length > frames) { clearTimeout(timeout); resolve(); } else v.requestVideoFrameCallback!(step);
    };
    v.requestVideoFrameCallback!(step);
    video.play().catch((e) => { clearTimeout(timeout); reject(e); });
  });
  video.pause();
  await seek(video, start);
  const d: number[] = [];
  for (let i = 1; i < times.length; i++) if (times[i] > times[i - 1]) d.push(times[i] - times[i - 1]);
  if (d.length < 3) return 30;
  d.sort((a, b) => a - b);
  const fps = 1 / d[d.length >> 1];
  const snap = COMMON_RATES.find((r) => Math.abs(r - fps) < 0.01 * r);
  return snap ?? Math.round(fps * 100) / 100;
}

function seek(video: HTMLVideoElement, t: number): Promise<void> {
  if (Math.abs(video.currentTime - t) < 1e-6 && video.readyState >= 2) return Promise.resolve();
  const p = once(video, 'seeked');
  video.currentTime = t;
  return p;
}

function once(el: HTMLMediaElement, ev: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const ok = () => { el.removeEventListener('error', bad); resolve(); };
    const bad = () => { el.removeEventListener(ev, ok); reject(new Error(`fastar: video error (${el.error?.message ?? 'cannot decode'})`)); };
    el.addEventListener(ev, ok, { once: true });
    el.addEventListener('error', bad, { once: true });
  });
}
