// fastar-timecode: make and check the audio timecode that phones listen
// for (StartOptions.timecode / TimecodeListener).
//
//   fastar-timecode generate -o timecode.wav --duration 4h   play into the mix from the set's start
//   fastar-timecode mix -o out.wav set.wav                  add it to music, following its level
//   fastar-timecode decode recording.wav                    what a phone hears in a recording
import { once } from 'node:events';
import { TimecodeEncoder, LevelFollower } from '../src/timecode/encoder';
import { TimecodeDecoder, fitTime } from '../src/timecode/decoder';
import { timecodeParams, type TimecodeOptions } from '../src/timecode/code';
import { WavReader, WavWriter, deinterleave, interleave, type SampleFormat } from './wav';

const HELP = `Usage: fastar-timecode <command> [options]

Commands:
  generate   write timecode to a WAV file, to play into the mix from the start of the set
  mix        add timecode to music (a WAV file, or a live stream), at a level that follows the music
  decode     find the timecode in a recording, and report what a phone would hear

Timecode options (generator and listeners must agree):
      --key <text>          event key (default fastar)
      --band <lo-hi>        band in Hz (default 19000-20000)
      --symbol <s>          symbol duration in seconds (default 0.1)
      --degree <n>          sequence degree, 10 to 22 (default 18: repeats after 7.3 h)

fastar-timecode generate -o timecode.wav --duration 4h
  -o, --out <file>          output WAV (default timecode.wav)
  -d, --duration <time>     length: 4h, 90m, 1:30:00 or seconds (required)
      --start <time>        set time of the first sample (default 0)
      --level <dBFS>        peak level (default -12)
      --rate <Hz>           sample rate (default 48000)
      --bits <16|24|32f>    sample format (default 16)

fastar-timecode mix -o out.wav set.wav
<live input> | fastar-timecode mix --raw --rate 48000 --channels 2 | <live output>
  -o, --out <file>          output WAV, in the input's format
      --raw                 read 32-bit float interleaved samples from stdin, write them to stdout
      --rate <Hz>           raw sample rate (default 48000)
      --channels <n>        raw channels (default 2)
      --start <time>        set time of the first sample (default 0)
      --relative <dB>       timecode level relative to the music above 4 kHz (default -24)
      --max <dBFS>          highest timecode level (default -18)
      --gate <dBFS>         music level above 4 kHz below which the timecode fades out (default -50)

fastar-timecode decode recording.wav
      --every <s>           report every this many seconds (default 10)
      --json                print the report as JSON

  -h, --help                show this help`;

/** Seconds from 4h, 90m, 45s, 1:30:00, 90:00 or a plain number. */
function parseTime(s: string): number {
  const m = /^([0-9.]+)\s*(h|m|min|s|ms)?$/.exec(s.trim());
  if (m) return parseFloat(m[1]) * ({ h: 3600, m: 60, min: 60, s: 1, ms: 0.001 }[m[2] ?? 's'] ?? 1);
  const parts = s.split(':').map(Number);
  if (parts.length >= 2 && parts.length <= 3 && parts.every((v) => v >= 0)) return parts.reduce((a, v) => a * 60 + v, 0);
  throw new Error(`cannot read the time "${s}"`);
}

function clock(t: number): string {
  if (!Number.isFinite(t)) return '-';
  const sign = t < 0 ? '-' : '';
  t = Math.round(Math.abs(t) * 1000) / 1000;
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  return `${sign}${h}:${String(m).padStart(2, '0')}:${s.toFixed(3).padStart(6, '0')}`;
}

interface Args {
  cmd: string;
  files: string[];
  flags: Map<string, string>;
}

const BOOL = new Set(['--raw', '--json', '-h', '--help']);
const ALIAS: Record<string, string> = { '-o': '--out', '-d': '--duration', '-h': '--help' };

function parseArgs(argv: string[]): Args {
  const flags = new Map<string, string>(), files: string[] = [];
  let cmd = '';
  for (let i = 0; i < argv.length; i++) {
    const a = ALIAS[argv[i]] ?? argv[i];
    if (a.startsWith('-') && a.length > 1) {
      if (BOOL.has(a)) flags.set(a, '1');
      else if (i + 1 < argv.length) flags.set(a, argv[++i]);
      else throw new Error(`${a} needs a value`);
    } else if (!cmd) cmd = a;
    else files.push(a);
  }
  return { cmd, files, flags };
}

function num(f: Map<string, string>, k: string, def: number): number {
  const v = f.get(k);
  if (v === undefined) return def;
  const x = parseFloat(v);
  if (!Number.isFinite(x)) throw new Error(`${k} needs a number`);
  return x;
}

function timecodeOptions(f: Map<string, string>): TimecodeOptions {
  const o: TimecodeOptions = {};
  if (f.has('--key')) o.key = f.get('--key');
  if (f.has('--band')) {
    const m = /^([0-9.]+)-([0-9.]+)$/.exec(f.get('--band')!);
    if (!m) throw new Error('--band takes <lo>-<hi> in Hz, e.g. 19000-20000');
    o.band = [parseFloat(m[1]), parseFloat(m[2])];
  }
  if (f.has('--symbol')) o.symbol = num(f, '--symbol', 0.1);
  if (f.has('--degree')) o.degree = num(f, '--degree', 18);
  timecodeParams(o);
  return o;
}

function describe(o: TimecodeOptions): string {
  const p = timecodeParams(o);
  return `key "${p.key}", ${p.f0}-${p.f1} Hz, ${p.symbol * 1000} ms symbols, repeats after ${clock(p.period)}`;
}

function generate(a: Args): void {
  const f = a.flags, tc = timecodeOptions(f);
  if (!f.has('--duration')) throw new Error('generate needs --duration (e.g. 4h)');
  const out = f.get('--out') ?? 'timecode.wav', rate = num(f, '--rate', 48000), duration = parseTime(f.get('--duration')!);
  const start = f.has('--start') ? parseTime(f.get('--start')!) : 0;
  const bits = f.get('--bits') ?? '16';
  const format: SampleFormat | undefined = ({ '16': 'pcm16', '24': 'pcm24', '32f': 'float32' } as Record<string, SampleFormat>)[bits];
  if (!format) throw new Error('--bits takes 16, 24 or 32f');
  const enc = new TimecodeEncoder({ ...tc, sampleRate: rate, start, level: num(f, '--level', -12) });
  if (duration > enc.params.period - start) console.warn(`warning: the timecode repeats after ${clock(enc.params.period)}; use a larger --degree for longer sets`);
  const w = new WavWriter(out, rate, 1, format);
  const total = Math.round(duration * rate), block = rate, fade = Math.round(0.05 * rate);
  const buf = new Float32Array(block), gain = new Float32Array(block), level = 10 ** (num(f, '--level', -12) / 20);
  const t0 = Date.now();
  for (let done = 0; done < total; done += block) {
    const n = Math.min(block, total - done), o = buf.subarray(0, n), g = gain.subarray(0, n);
    for (let i = 0; i < n; i++) {
      const left = total - done - i;
      g[i] = left < fade ? (level * (1 - Math.cos((Math.PI * left) / fade))) / 2 : level;
    }
    enc.render(o, g);
    w.write([o]);
  }
  w.close();
  console.log(`wrote ${out}: ${clock(duration)} of timecode from set time ${clock(start)}, ${rate} Hz, ${describe(tc)} (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
}

async function mix(a: Args): Promise<void> {
  const f = a.flags, tc = timecodeOptions(f);
  const start = f.has('--start') ? parseTime(f.get('--start')!) : 0;
  const follow = { relative: num(f, '--relative', -24), max: num(f, '--max', -18), gate: num(f, '--gate', -50) };
  const make = (rate: number) => ({
    enc: new TimecodeEncoder({ ...tc, sampleRate: rate, start }),
    follower: new LevelFollower(rate, follow),
  });
  let on = 0, total = 0, gainSum = 0;
  const addTo = (s: ReturnType<typeof make>, ch: Float32Array[]) => {
    const n = ch[0].length, g = new Float32Array(n), t = new Float32Array(n);
    s.follower.process(ch, g);
    s.enc.render(t, g);
    for (const c of ch) for (let i = 0; i < n; i++) c[i] += t[i];
    for (let i = 0; i < n; i++) if (g[i] > 1e-4) { on++; gainSum += 20 * Math.log10(g[i]); }
    total += n;
  };

  if (f.has('--raw')) {
    const rate = num(f, '--rate', 48000), channels = num(f, '--channels', 2), fb = 4 * channels;
    const s = make(rate);
    let carry = Buffer.alloc(0);
    for await (const chunk of process.stdin) {
      const buf = carry.length ? Buffer.concat([carry, chunk as Buffer]) : (chunk as Buffer);
      const n = Math.floor(buf.length / fb);
      carry = Buffer.from(buf.subarray(n * fb));
      if (!n) continue;
      const ch = deinterleave(buf.subarray(0, n * fb), n, channels, 'float32');
      addTo(s, ch);
      if (!process.stdout.write(interleave(ch, n, 'float32').bytes)) await once(process.stdout, 'drain');
    }
    return;
  }

  if (a.files.length !== 1) throw new Error('mix takes one input WAV (or --raw)');
  const out = f.get('--out');
  if (!out) throw new Error('mix needs -o <output.wav>');
  const r = new WavReader(a.files[0]), info = r.info, s = make(info.sampleRate);
  const w = new WavWriter(out, info.sampleRate, info.channels, info.format);
  for (let ch = r.read(4096); ch; ch = r.read(4096)) {
    addTo(s, ch);
    w.write(ch);
  }
  r.close();
  w.close();
  const pct = total ? (100 * on) / total : 0;
  console.log(
    `wrote ${out}: ${clock(total / info.sampleRate)} with timecode from set time ${clock(start)}, ${describe(tc)}; ` +
      `timecode on ${pct.toFixed(0)}% of the time${on ? `, at ${(gainSum / on).toFixed(1)} dBFS on average` : ''}`,
  );
  if (w.clipped) console.warn(`warning: ${w.clipped} samples clipped; lower the music or use a float input`);
}

function decode(a: Args): void {
  const f = a.flags, tc = timecodeOptions(f);
  if (a.files.length !== 1) throw new Error('decode takes one recording (WAV)');
  const every = num(f, '--every', 10);
  const r = new WavReader(a.files[0]), info = r.info, fs = info.sampleRate;
  const dec = new TimecodeDecoder({ ...tc, sampleRate: fs });
  const timeline: Array<{ at: number; time: number; std: number; locked: boolean; snr: number; clear: number }> = [];
  let frame = 0, next = every;
  for (let ch = r.read(4096); ch; ch = r.read(4096)) {
    const n = ch[0].length, mono = new Float32Array(n);
    for (const c of ch) for (let i = 0; i < n; i++) mono[i] += c[i] / ch.length;
    dec.push(mono, frame);
    frame += n;
    const t = frame / fs;
    while (t >= next) {
      const e = dec.estimate(next), s = dec.stats;
      timeline.push({ at: next, time: e.time, std: e.std, locked: e.locked, snr: s.snr, clear: s.clear });
      next += every;
    }
  }
  r.close();
  const dur = frame / fs, fit = dec.fit, s = dec.stats, end = dec.estimate(dur);
  const sigma = fit.points ? Math.sqrt(fit.varTime * fit.points) : NaN;
  const report = {
    file: a.files[0], sampleRate: fs, channels: info.channels, duration: dur, timecode: timecodeParams(tc),
    found: fit.points > 0,
    lockedAfter: s.lockedAt,
    /** Set time at the start of the recording. */
    startTime: fit.points ? fitTime(fit, 0) : NaN,
    /** How much faster the generator's clock ran than the recorder's (ppm). */
    clockPpm: fit.points ? (fit.rate - 1) * 1e6 : NaN,
    /** Spread of the sweep times about the fit (s). */
    spread: sigma,
    lockedAtEnd: end.locked,
    snr: s.snr, clear: s.clear, errors: s.errors, symbols: s.symbols,
    timeline,
  };
  if (f.has('--json')) {
    console.log(JSON.stringify(report, (_, v) => (typeof v === 'number' && !Number.isFinite(v) ? null : v), 2));
  } else {
    console.log(`${a.files[0]}: ${clock(dur)}, ${fs} Hz, ${info.channels} channel${info.channels === 1 ? '' : 's'}; listening for ${describe(tc)}`);
    if (!report.found) {
      console.log(`no timecode found (${(100 * s.clear).toFixed(0)}% of symbols heard clearly at the end)`);
    } else {
      console.log(`locked after ${s.lockedAt.toFixed(1)} s; set time at the start of the recording: ${clock(report.startTime)}`);
      console.log(`clock: generator ${report.clockPpm >= 0 ? '+' : ''}${report.clockPpm.toFixed(1)} ppm against the recorder; sweep times within ${(sigma * 1000).toFixed(2)} ms of the fit`);
      console.log(`heard clearly: ${(100 * s.clear).toFixed(0)}% of recent symbols at ${s.snr.toFixed(1)} dB; ${(100 * s.errors).toFixed(1)}% disagreed with the code`);
    }
    for (const p of timeline) {
      console.log(
        `  ${clock(p.at)}  set ${clock(p.time)}  ${p.locked ? 'locked' : p.std < Infinity ? 'coarse' : 'searching'}` +
          `${p.std < Infinity ? `  ±${(p.std * 1000).toFixed(1)} ms` : ''}  ${p.snr.toFixed(1)} dB, ${(100 * p.clear).toFixed(0)}% clear`,
      );
    }
  }
  if (!report.found) process.exitCode = 2;
}

async function main(argv: string[]): Promise<void> {
  const a = parseArgs(argv);
  if (a.flags.has('--help') || !a.cmd) {
    console.log(HELP);
    if (!a.cmd && !a.flags.has('--help')) process.exitCode = 1;
    return;
  }
  if (a.cmd === 'generate') generate(a);
  else if (a.cmd === 'mix') await mix(a);
  else if (a.cmd === 'decode') decode(a);
  else throw new Error(`unknown command ${a.cmd} (generate, mix or decode)`);
}

main(process.argv.slice(2)).catch((e: Error) => {
  console.error(`fastar-timecode: ${e.message}`);
  process.exit(1);
});
