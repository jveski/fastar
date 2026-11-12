// Worker side of the timecode listener: decodes the microphone audio that
// the audio worklet sends straight to this worker, and finds loopback
// calibration bursts in it.
import { TimecodeDecoder, type DecoderStats, type TimecodeFit } from '../timecode/decoder';
import { LoopbackDetector } from '../timecode/loopback';
import type { TimecodeOptions } from '../timecode/code';

export interface TimecodeInit {
  type: 'init-timecode';
  options: TimecodeOptions;
  sampleRate: number;
  /** Receives { frame, samples } from the audio worklet. */
  port: MessagePort;
}

/** The page plays the loopback burst from audio time `start` (s, as rendered): find it. */
export interface TimecodeCalibrate {
  type: 'calibrate';
  start: number;
}

/** Audio from the worklet: samples[0] is audio frame `frame` (AudioWorkletGlobalScope.currentFrame). */
export interface TimecodeAudio {
  frame: number;
  samples: Float32Array;
}

export type FromTimecode =
  | { type: 'timecode-ready' }
  | { type: 'timecode'; fit: TimecodeFit; stats: DecoderStats; audioTime: number }
  /** Delay (s) from the microphone to the audio clock, or null if the burst was not heard. */
  | { type: 'timecode-latency'; latency: number | null }
  | { type: 'timecode-error'; message: string };

export class TimecodeCore {
  private decoder: TimecodeDecoder | null = null;
  private loopback: LoopbackDetector | null = null;
  private dirty = false;

  constructor(private readonly post: (m: FromTimecode) => void) {}

  init(m: TimecodeInit): void {
    try {
      const d = (this.decoder = new TimecodeDecoder({ ...m.options, sampleRate: m.sampleRate }));
      d.onSymbol = () => { this.dirty = true; };
      m.port.onmessage = (e: MessageEvent) => this.audio(e.data as TimecodeAudio);
      this.post({ type: 'timecode-ready' });
    } catch (e) {
      this.post({ type: 'timecode-error', message: (e as Error).message ?? String(e) });
    }
  }

  handle(m: TimecodeCalibrate): void {
    if (m.type === 'calibrate' && this.decoder) this.loopback = new LoopbackDetector(this.decoder.sampleRate, m.start);
  }

  audio(a: TimecodeAudio): void {
    const d = this.decoder;
    if (!d) return;
    d.push(a.samples, a.frame);
    if (this.loopback) {
      const r = this.loopback.push(a.samples, a.frame);
      if (r !== undefined) {
        this.loopback = null;
        this.post({ type: 'timecode-latency', latency: r });
      }
    }
    if (!this.dirty) return;
    this.dirty = false;
    this.post({ type: 'timecode', fit: d.fit, stats: d.stats, audioTime: d.time });
  }
}
