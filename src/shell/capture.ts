// Camera access and main-thread luma conversion (used when frames cannot be
// read directly in the worker).
import { rgbaToGray } from '../image/gray';
import type { Facing } from './sensors';

export async function openCamera(facing: Facing): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('fastar: camera access is not available (needs HTTPS)');
  return navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
  });
}

/** Processing size: 640 px on the long side, width a multiple of 4. */
export function processingSize(vw: number, vh: number): { width: number; height: number } {
  const s = Math.min(1, 640 / Math.max(vw, vh));
  return { width: Math.max(4, Math.round((vw * s) / 4) * 4), height: Math.max(1, Math.round(vh * s)) };
}

const VS = `#version 300 es
in vec2 p;
void main() { gl_Position = vec4(p, 0.0, 1.0); }`;

// Each output texel packs 4 horizontally adjacent luma values. Every luma
// value averages 4 bilinear taps spread over the source footprint, which is
// an exact 2x2 box for 2x downscaling and a reasonable prefilter otherwise.
const FS = `#version 300 es
precision highp float;
uniform sampler2D tex;
uniform vec2 outSize;
uniform vec2 tap;
out vec4 o;
float luma(vec2 uv) {
  vec3 c = texture(tex, uv + vec2(-tap.x, -tap.y)).rgb + texture(tex, uv + vec2(tap.x, -tap.y)).rgb
         + texture(tex, uv + vec2(-tap.x, tap.y)).rgb + texture(tex, uv + vec2(tap.x, tap.y)).rgb;
  return dot(c * 0.25, vec3(0.299, 0.587, 0.114));
}
void main() {
  float x0 = floor(gl_FragCoord.x) * 4.0;
  float y = (floor(gl_FragCoord.y) + 0.5) / outSize.y;
  o = vec4(luma(vec2((x0 + 0.5) / outSize.x, y)), luma(vec2((x0 + 1.5) / outSize.x, y)),
           luma(vec2((x0 + 2.5) / outSize.x, y)), luma(vec2((x0 + 3.5) / outSize.x, y)));
}`;

/** Video element -> 8-bit luma at processing size, via WebGL2 (or 2D canvas fallback). */
export class LumaGrabber {
  private gl: WebGL2RenderingContext | null = null;
  private ctx2d: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;
  private tex: WebGLTexture | null = null;

  constructor(readonly width: number, readonly height: number) {
    const canvas = makeCanvas(width / 4, height);
    const gl = canvas.getContext('webgl2', { antialias: false, depth: false, alpha: true, premultipliedAlpha: false }) as WebGL2RenderingContext | null;
    if (gl && this.setupGl(gl)) return;
    const c2 = makeCanvas(width, height);
    this.ctx2d = c2.getContext('2d', { willReadFrequently: true }) as CanvasRenderingContext2D | null;
    if (!this.ctx2d) throw new Error('fastar: no WebGL2 or 2D canvas for frame capture');
  }

  private setupGl(gl: WebGL2RenderingContext): boolean {
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null;
    };
    const vs = sh(gl.VERTEX_SHADER, VS), fs = sh(gl.FRAGMENT_SHADER, FS);
    if (!vs || !fs) return false;
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.bindAttribLocation(prog, 0, 'p');
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return false;
    gl.useProgram(prog);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.uniform1i(gl.getUniformLocation(prog, 'tex'), 0);
    gl.uniform2f(gl.getUniformLocation(prog, 'outSize'), this.width, this.height);
    // Taps at +/- a quarter of the footprint (in texture coords).
    gl.uniform2f(gl.getUniformLocation(prog, 'tap'), 0.25 / this.width, 0.25 / this.height);
    gl.viewport(0, 0, this.width / 4, this.height);
    this.gl = gl;
    return true;
  }

  /** Grab the current video frame into `out` (width*height bytes). */
  grab(video: HTMLVideoElement, out: Uint8Array): boolean {
    if (!video.videoWidth || !video.videoHeight) return false;
    if (this.gl) {
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.readPixels(0, 0, this.width / 4, this.height, gl.RGBA, gl.UNSIGNED_BYTE, out);
      return true;
    }
    const c = this.ctx2d!;
    c.drawImage(video, 0, 0, this.width, this.height);
    const img = c.getImageData(0, 0, this.width, this.height);
    rgbaToGray(img.data, this.width, this.height, { width: this.width, height: this.height, data: out });
    return true;
  }
}

function makeCanvas(w: number, h: number): HTMLCanvasElement | OffscreenCanvas {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

/** Whether frames can be read in the worker via MediaStreamTrackProcessor (Chrome). */
export function canUseTrackProcessor(): boolean {
  return typeof (globalThis as { MediaStreamTrackProcessor?: unknown }).MediaStreamTrackProcessor === 'function';
}

export function createTrackProcessor(track: MediaStreamTrack): ReadableStream<VideoFrame> {
  const MSTP = (globalThis as unknown as { MediaStreamTrackProcessor: new (o: { track: MediaStreamTrack; maxBufferSize?: number }) => { readable: ReadableStream<VideoFrame> } }).MediaStreamTrackProcessor;
  return new MSTP({ track, maxBufferSize: 1 }).readable;
}
