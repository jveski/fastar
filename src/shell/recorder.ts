// Session recorder: camera video (MP4 where supported, else WebM) plus a JSON
// log of presented-frame capture times, IMU samples and device info.
// Optionally also stores processed luma frames for exact offline replay.
import type { ImuReading } from './sensors';
import { LumaGrabber } from './capture';
import type { Intrinsics } from '../vision/camera';

export interface RecordedFrame {
  /** Capture time (performance.now() ms). */
  t: number;
  mediaTime: number;
  presentedFrames: number;
  expectedDisplayTime: number;
}

export interface SessionLog {
  version: 1;
  device: {
    userAgent: string;
    videoWidth: number;
    videoHeight: number;
    width: number;
    height: number;
    intrinsics: Intrinsics | null;
    sensors: string;
    path: string;
    td: number;
    tdCalibrated: boolean;
  };
  /** First recorded frame capture time; video mediaTime 0 corresponds to it. */
  startTime: number;
  frames: RecordedFrame[];
  /** [t(ms), gx, gy, gz, ax, ay, az] per sample, camera axes (before axis corrections). */
  imu: number[][];
  /** Present when recorded with { luma: true }: frame index for each luma frame. */
  lumaFrames?: number[];
}

export interface RecordedSession {
  log: SessionLog;
  video: Blob | null;
  mimeType: string;
  /** Concatenated width*height luma frames (only with { luma: true }). */
  luma: Uint8Array | null;
}

export interface RecordOptions {
  /** Also keep processed luma frames (large; for exact offline replay). */
  luma?: boolean;
  /** Skip the video file (log only). */
  noVideo?: boolean;
}

interface Host {
  readonly mediaStream: MediaStream | null;
  readonly processingSize: { width: number; height: number };
  readonly intrinsics: Intrinsics | null;
  readonly stats: { sensors: string; path: string; td: number; tdCalibrated: boolean };
}

export class Recorder {
  recording = false;
  private mr: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private log: SessionLog | null = null;
  private lumaChunks: Uint8Array[] = [];
  private grabber: LumaGrabber | null = null;
  private opts: RecordOptions = {};
  private mime = '';

  constructor(private readonly host: Host) {}

  /** Begin recording (the tracker must be started). */
  start(o: RecordOptions = {}): void {
    if (this.recording) return;
    const stream = this.host.mediaStream;
    const track = stream?.getVideoTracks()[0];
    const settings = track?.getSettings() ?? {};
    const { width, height } = this.host.processingSize;
    this.opts = o;
    this.log = {
      version: 1,
      device: {
        userAgent: navigator.userAgent,
        videoWidth: settings.width ?? 0,
        videoHeight: settings.height ?? 0,
        width, height,
        intrinsics: this.host.intrinsics,
        sensors: this.host.stats.sensors,
        path: this.host.stats.path,
        td: this.host.stats.td,
        tdCalibrated: this.host.stats.tdCalibrated,
      },
      startTime: NaN,
      frames: [],
      imu: [],
      lumaFrames: o.luma ? [] : undefined,
    };
    this.chunks = [];
    this.lumaChunks = [];
    if (o.luma) this.grabber = new LumaGrabber(width, height);
    if (stream && !o.noVideo && typeof MediaRecorder !== 'undefined') {
      const types = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'];
      this.mime = types.find((t) => MediaRecorder.isTypeSupported(t)) ?? '';
      this.mr = new MediaRecorder(stream, this.mime ? { mimeType: this.mime } : undefined);
      this.mr.ondataavailable = (e) => { if (e.data.size) this.chunks.push(e.data); };
      this.mr.start(1000);
    }
    this.recording = true;
  }

  async stop(): Promise<RecordedSession> {
    if (!this.recording || !this.log) throw new Error('fastar: recorder is not running');
    this.recording = false;
    const mr = this.mr;
    let video: Blob | null = null;
    if (mr) {
      await new Promise<void>((resolve) => {
        mr.onstop = () => resolve();
        mr.stop();
      });
      video = new Blob(this.chunks, { type: mr.mimeType || this.mime });
    }
    this.mr = null;
    let luma: Uint8Array | null = null;
    if (this.lumaChunks.length) {
      const n = this.lumaChunks.reduce((s, c) => s + c.length, 0);
      luma = new Uint8Array(n);
      let o = 0;
      for (const c of this.lumaChunks) { luma.set(c, o); o += c.length; }
    }
    const log = this.log;
    this.log = null;
    return { log, video, mimeType: video?.type ?? '', luma };
  }

  /** @internal */
  onImu(r: ImuReading): void {
    if (!this.recording || !this.log) return;
    this.log.imu.push([r.t, r.gx, r.gy, r.gz, r.ax, r.ay, r.az]);
  }

  /** @internal */
  onVideoFrame(captureTime: number, meta: VideoFrameCallbackMetadata, video: HTMLVideoElement): void {
    if (!this.recording || !this.log) return;
    if (Number.isNaN(this.log.startTime)) this.log.startTime = captureTime;
    this.log.frames.push({ t: captureTime, mediaTime: meta.mediaTime, presentedFrames: meta.presentedFrames, expectedDisplayTime: meta.expectedDisplayTime });
    if (this.opts.luma && this.grabber) {
      const { width, height } = this.host.processingSize;
      const buf = new Uint8Array(width * height);
      if (this.grabber.grab(video, buf)) {
        this.lumaChunks.push(buf);
        this.log.lumaFrames!.push(this.log.frames.length - 1);
      }
    }
  }
}

/** Serialize a session to `{ json, video, luma }` files for download or upload. */
export function sessionFiles(s: RecordedSession): { json: string; video: Blob | null; luma: Uint8Array | null } {
  return { json: JSON.stringify(s.log), video: s.video, luma: s.luma };
}
