// Timecode listener: hears the set's audio timecode through the microphone
// and gives the set time of any moment on the page clock (performance.now()).
//
// An audio worklet copies microphone samples straight to a worker, which
// decodes them on the audio clock (sample frames). Here, the audio clock is
// mapped onto performance.now(): getOutputTimestamp() relates the two at
// the output, and the delay from the microphone to the audio clock is taken
// off. Browsers do not report that delay reliably (on iPhones it is often
// hundreds of milliseconds, different every time), so it is measured: a
// short burst played through the speaker is found in what the microphone
// hears.
import { fitEstimate, type DecoderStats, type TimecodeFit } from '../timecode/decoder';
import { timecodeParams, type TimecodeOptions } from '../timecode/code';
import { loopbackSignal } from '../timecode/loopback';
import type { FromTimecode, TimecodeCalibrate, TimecodeInit } from '../worker/timecode-core';
import { spawnWorker } from './spawn';

export interface ListenOptions extends TimecodeOptions {
  /** Microphone stream to listen to (default: open the microphone, with voice processing off). */
  stream?: MediaStream;
  /**
   * Measure the delay from the microphone to the page by playing a short,
   * quiet burst of chirps through the speaker when listening starts, and
   * again if the delay changes (default true). Until it is measured, set
   * time is not reported as locked.
   */
  calibrate?: boolean;
  /** Delay (ms) from sound reaching the microphone to the page, if known: no calibration then. */
  latency?: number;
  /** Set time counts as locked once good to this (s, default 0.01). */
  lockStd?: number;
  /** Stay locked this long without hearing the timecode (s, default 600). */
  maxCoast?: number;
  /** Load the worker from a URL instead of the inlined source (strict CSP). */
  workerUrl?: string | URL;
}

export interface SetTime {
  /** Seconds into the set. */
  time: number;
  /** Standard deviation of `time` (s). */
  std: number;
  /** `time` is good to `lockStd`. */
  locked: boolean;
}

export interface TimecodeStats extends DecoderStats {
  /** Listening (microphone and decoder running). */
  running: boolean;
  /** Audio sample rate (Hz). */
  sampleRate: number;
  /** Delay taken off between sound reaching the microphone and the page hearing it (ms), and where it came from. */
  latency: number;
  latencySource: 'measured' | 'option' | 'browser' | 'assumed';
  /** Loopback calibration: measuring (set time is not locked meanwhile), done, failed (assumed delay), or off. */
  calibration: 'measuring' | 'done' | 'failed' | 'off';
  /** Why listening failed, if it did. */
  error: string | null;
}

/** Microphone settings for listening: voice processing filters out the timecode band. */
export const TIMECODE_AUDIO: MediaTrackConstraints = {
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
  channelCount: 1,
};

const WORKLET_NAME = 'fastar-timecode';

// Collects microphone samples into blocks and posts each, with the frame of
// its first sample, to the port it is given (the decoding worker's).
const WORKLET_SOURCE = `class P extends AudioWorkletProcessor {
  constructor(o) {
    super();
    this.size = o.processorOptions.size;
    this.buf = new Float32Array(this.size);
    this.fill = 0;
    this.frame = 0;
    this.out = null;
    this.port.onmessage = (e) => { this.out = e.data; };
  }
  flush() {
    if (this.fill) this.out.postMessage({ frame: this.frame, samples: this.buf.slice(0, this.fill) });
    this.fill = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || !this.out) return true;
    if (this.fill && currentFrame !== this.frame + this.fill) this.flush();
    let i = 0;
    while (i < ch.length) {
      if (this.fill === 0) this.frame = currentFrame + i;
      const n = Math.min(ch.length - i, this.size - this.fill);
      this.buf.set(ch.subarray(i, i + n), this.fill);
      this.fill += n;
      i += n;
      if (this.fill === this.size) {
        this.out.postMessage({ frame: this.frame, samples: this.buf }, [this.buf.buffer]);
        this.buf = new Float32Array(this.size);
        this.fill = 0;
      }
    }
    return true;
  }
}
registerProcessor('${WORKLET_NAME}', P);`;

type TimecodeEvent = 'lock' | 'unlock';

interface StatsTrack extends MediaStreamTrack {
  stats?: { latency?: number };
}

export class TimecodeListener {
  private readonly o: ListenOptions;
  private readonly ctx: AudioContext | null;
  private worker: Worker | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private node: AudioWorkletNode | null = null;
  private sink: GainNode | null = null;
  private stream: MediaStream | null = null;
  private ownsStream = false;
  private track: StatsTrack | null = null;
  private fit: TimecodeFit | null = null;
  private wasLocked = false;
  private readonly offsets: number[] = [];
  private lastSample = -Infinity;
  /** Measured delay from the microphone to the audio clock (ms), NaN until measured. */
  private measured = NaN;
  private calibrating = false;
  private attempts = 0;
  private calTimer: ReturnType<typeof setTimeout> | undefined;
  private lastCalibration = -Infinity;
  private seenJumps = 0;
  private readonly listeners: Record<TimecodeEvent, Set<() => void>> = { lock: new Set(), unlock: new Set() };
  private readonly onVisible = () => {
    if (document.visibilityState === 'visible' && this.ctx && this.ctx.state !== 'running' && this.st.running) void this.ctx.resume();
  };
  private readonly st: TimecodeStats = {
    symbols: 0, snr: 0, clear: 0, errors: 0, codeLocked: false, lockedAt: NaN, jumps: 0,
    running: false, sampleRate: 0, latency: 0, latencySource: 'assumed', calibration: 'off', error: null,
  };

  /**
   * Create inside a user gesture (a tap handler): the audio context must
   * be started from one on iOS. Then call start().
   */
  constructor(o: ListenOptions = {}) {
    this.o = o;
    timecodeParams(o); // fail early on bad options
    let ctx: AudioContext | null = null;
    try {
      ctx = new AudioContext({ latencyHint: 'interactive', sampleRate: 48000 });
    } catch {
      try { ctx = new AudioContext({ latencyHint: 'interactive' }); } catch { ctx = null; }
    }
    this.ctx = ctx;
    void ctx?.resume().catch(() => {});
  }

  /** Create and start in one step; call from a user gesture. */
  static listen(o: ListenOptions = {}): Promise<TimecodeListener> {
    const l = new TimecodeListener(o);
    return l.start(o.stream).then(() => l);
  }

  /**
   * Start listening: to the audio track of `stream` if it has one (open it
   * with TIMECODE_AUDIO), otherwise to the microphone, opened here.
   */
  async start(stream: MediaStream | undefined = this.o.stream): Promise<void> {
    if (this.st.running) return;
    const ctx = this.ctx;
    try {
      if (!ctx) throw new Error('Web Audio is not available');
      if (typeof AudioWorkletNode === 'undefined' || !ctx.audioWorklet) throw new Error('audio worklets are not available');
      let track = stream?.getAudioTracks()[0];
      if (!track) {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('microphone access is not available (needs HTTPS)');
        this.stream = await navigator.mediaDevices.getUserMedia({ audio: TIMECODE_AUDIO, video: false });
        this.ownsStream = true;
        track = this.stream.getAudioTracks()[0];
      }
      this.track = track as StatsTrack;
      const rate = track.getSettings?.().sampleRate;
      if (rate && rate < 44100) console.warn(`fastar: the microphone runs at ${rate} Hz, too low to hear the timecode (a Bluetooth headset?)`);

      const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'text/javascript' }));
      try { await ctx.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
      const worker = (this.worker = spawnWorker(this.o.workerUrl));
      worker.onmessage = (e) => this.onWorker(e.data as FromTimecode);
      worker.onerror = (e) => console.error('fastar timecode worker error', e.message);
      const ch = new MessageChannel();
      const init: TimecodeInit = {
        type: 'init-timecode', sampleRate: ctx.sampleRate, port: ch.port2,
        options: { key: this.o.key, band: this.o.band, symbol: this.o.symbol, degree: this.o.degree },
      };
      worker.postMessage(init, [ch.port2]);

      this.source = ctx.createMediaStreamSource(new MediaStream([track]));
      this.node = new AudioWorkletNode(ctx, WORKLET_NAME, {
        numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 1, channelCountMode: 'explicit',
        processorOptions: { size: 2048 },
      });
      this.node.port.postMessage(ch.port1, [ch.port1]);
      // Silent path to the output, so that every browser keeps the worklet running.
      this.sink = ctx.createGain();
      this.sink.gain.value = 0;
      this.source.connect(this.node);
      this.node.connect(this.sink);
      this.sink.connect(ctx.destination);
      if (ctx.state !== 'running') await ctx.resume().catch(() => {});
      this.st.sampleRate = ctx.sampleRate;
      this.st.running = true;
      this.st.error = null;
      if (typeof document !== 'undefined') document.addEventListener('visibilitychange', this.onVisible);
      if (this.o.calibrate !== false && this.o.latency === undefined) {
        this.st.calibration = 'measuring';
        // Once audio is flowing through the worklet.
        this.scheduleCalibration(300);
      }
    } catch (e) {
      this.st.error = (e as Error).message ?? String(e);
      this.teardown();
      throw new Error(`fastar: timecode listening failed: ${this.st.error}`, { cause: e });
    }
  }

  /** Stop listening (the microphone too, if opened here). */
  stop(): void {
    this.teardown();
    void this.ctx?.close().catch(() => {});
  }

  /** 'lock' / 'unlock': set time became good to lockStd, or stopped being. */
  on(event: TimecodeEvent, cb: () => void): () => void {
    this.listeners[event].add(cb);
    return () => this.listeners[event].delete(cb);
  }

  /**
   * Measure the delay from the microphone to the page again, by playing the
   * calibration burst (e.g. after the audio route changed). Happens by
   * itself when listening starts and when the delay seems to have changed.
   */
  calibrate(): void {
    const ctx = this.ctx, worker = this.worker;
    if (!ctx || !worker || !this.st.running || this.calibrating || ctx.state !== 'running') return;
    clearTimeout(this.calTimer);
    const signal = loopbackSignal(ctx.sampleRate);
    const buffer = ctx.createBuffer(1, signal.length, ctx.sampleRate);
    buffer.getChannelData(0).set(signal);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    src.onended = () => src.disconnect();
    const start = ctx.currentTime + 0.15;
    src.start(start);
    const m: TimecodeCalibrate = { type: 'calibrate', start };
    worker.postMessage(m);
    this.calibrating = true;
    this.lastCalibration = performance.now();
    if (this.st.calibration !== 'done') this.st.calibration = 'measuring';
  }

  /** Set time at page time `time` (performance.now() ms, e.g. a camera frame's capture time), or null before any. */
  timeAt(time: number): SetTime | null {
    const f = this.fit;
    if (!f || !f.points || !this.ctx) return null;
    const e = fitEstimate(f, this.audioTimeAt(time), this.o.lockStd, this.o.maxCoast);
    return { time: e.time, std: e.std, locked: e.locked && this.st.running && this.ctx.state === 'running' && this.st.calibration !== 'measuring' };
  }

  /** Set time now. */
  get time(): SetTime | null {
    return this.timeAt(performance.now());
  }

  get stats(): TimecodeStats {
    return { ...this.st };
  }

  /** Audio clock time (s) of sound that reached the microphone at page time `time` (ms). */
  audioTimeAt(time: number): number {
    const ctx = this.ctx!;
    this.sampleClock();
    const off = this.offsets.length ? medianOf(this.offsets) : performance.now() - ctx.currentTime * 1000;
    return (time - off + this.latency()) / 1000;
  }

  private scheduleCalibration(ms: number): void {
    clearTimeout(this.calTimer);
    this.calTimer = setTimeout(() => this.calibrate(), ms);
  }

  private teardown(): void {
    this.st.running = false;
    clearTimeout(this.calTimer);
    this.calibrating = false;
    if (typeof document !== 'undefined') document.removeEventListener('visibilitychange', this.onVisible);
    this.source?.disconnect();
    this.node?.disconnect();
    this.sink?.disconnect();
    this.node?.port.close();
    this.worker?.terminate();
    this.source = this.node = this.sink = null;
    this.worker = null;
    if (this.ownsStream) this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = this.track = null;
  }

  private onWorker(m: FromTimecode): void {
    if (m.type === 'timecode-error') {
      this.st.error = m.message;
      console.error('fastar timecode:', m.message);
      return;
    }
    if (m.type === 'timecode-latency') {
      this.calibrating = false;
      if (m.latency !== null) {
        this.measured = m.latency * 1000;
        this.st.calibration = 'done';
        this.attempts = 0;
      } else if (++this.attempts < 3) {
        // Not heard (a noisy moment, the volume down): try again shortly.
        this.scheduleCalibration(3000);
      } else if (this.st.calibration === 'measuring') {
        console.warn('fastar: could not measure the microphone delay (headphones, or the volume down?); assuming one');
        this.st.calibration = 'failed';
      }
      this.checkLock();
      return;
    }
    if (m.type !== 'timecode') return;
    this.fit = m.fit;
    Object.assign(this.st, m.stats);
    if (m.stats.jumps > this.seenJumps) {
      // The sweeps' arrival jumped: likely the microphone path's delay changed. Measure it again (at most every 10 s).
      this.seenJumps = m.stats.jumps;
      if (this.st.calibration !== 'off') this.scheduleCalibration(Math.max(500, this.lastCalibration + 10000 - performance.now()));
    }
    this.checkLock();
  }

  private checkLock(): void {
    this.sampleClock();
    const locked = !!this.timeAt(performance.now())?.locked;
    if (locked !== this.wasLocked) {
      this.wasLocked = locked;
      this.listeners[locked ? 'lock' : 'unlock'].forEach((cb) => cb());
    }
  }

  /**
   * Page-clock offset of the audio clock at the output: performance.now()
   * = audio time * 1000 + offset for a sample being played. A few recent
   * readings, so one late reading does not move it.
   */
  private sampleClock(): void {
    const ctx = this.ctx, now = performance.now();
    if (!ctx || now - this.lastSample < 250) return;
    this.lastSample = now;
    const ts = ctx.getOutputTimestamp?.();
    let off: number;
    if (ts && ts.performanceTime && ts.contextTime) off = ts.performanceTime - ts.contextTime * 1000;
    else if (ctx.currentTime > 0) off = now - ctx.currentTime * 1000 + this.outputSideLatency();
    else return;
    this.offsets.push(off);
    if (this.offsets.length > 9) this.offsets.shift();
  }

  /** From the audio clock's render time to its output (ms). */
  private outputSideLatency(): number {
    const ctx = this.ctx!;
    return ((ctx.baseLatency || 0) + ((ctx as AudioContext & { outputLatency?: number }).outputLatency || 0)) * 1000;
  }

  /**
   * Delay (ms) between sound reaching the microphone and the audio clock
   * time it is heard at, less the clock's offset at the output: measured by
   * loopback (which covers both ways), or the output side as reported plus
   * the input side as given, reported or assumed.
   */
  private latency(): number {
    let v: number, source: TimecodeStats['latencySource'];
    if (this.o.latency !== undefined) {
      v = this.outputSideLatency() + this.o.latency;
      source = 'option';
    } else if (!Number.isNaN(this.measured)) {
      v = this.measured;
      source = 'measured';
    } else {
      const s = this.track?.stats?.latency;
      const reported = typeof s === 'number' && s > 0 && s < 1000;
      v = this.outputSideLatency() + (reported ? s : 10);
      source = reported ? 'browser' : 'assumed';
    }
    this.st.latency = v;
    this.st.latencySource = source;
    return v;
  }
}

function medianOf(a: number[]): number {
  const s = a.slice().sort((x, y) => x - y);
  return s[s.length >> 1];
}
