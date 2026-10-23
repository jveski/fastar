// On-device benchmark: per-stage timings, live jitter and latency plots,
// target outline overlay, and session recording.
import { ImageTracker, compileTarget, encodeTargets, rgbaToGray, sessionFiles, type FrameEvent } from '../../src/index';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>('video');
const overlay = $<HTMLCanvasElement>('overlay');
const plots = $<HTMLCanvasElement>('plots');
const statsEl = $('stats');
const msg = $('msg');

let targetsBuf: ArrayBuffer | null = null;

$<HTMLInputElement>('targetsFile').onchange = async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) { targetsBuf = await f.arrayBuffer(); msg.textContent = `loaded ${f.name}`; }
};

$<HTMLInputElement>('imageFile').onchange = async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (!f) return;
  msg.textContent = 'compiling…';
  const bmp = await createImageBitmap(f);
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext('2d')!;
  ctx.drawImage(bmp, 0, 0);
  const d = ctx.getImageData(0, 0, bmp.width, bmp.height);
  const t0 = performance.now();
  const t = compileTarget(rgbaToGray(d.data, d.width, d.height), {
    physicalWidth: parseFloat($<HTMLInputElement>('width').value) || 0.21, name: f.name,
  });
  targetsBuf = encodeTargets([t]);
  const q = t.meta.quality;
  msg.textContent = `compiled in ${(performance.now() - t0).toFixed(0)} ms: ${q.features} features, ` +
    `${(q.coverage * 100).toFixed(0)}% coverage${q.warnings.length ? ' — ' + q.warnings.join('; ') : ''}`;
};

// Rolling series for plots.
const N = 160;
const jitterSeries: number[] = [];
const latencySeries: number[] = [];
const recentT: number[][] = [];

function pushSeries(a: number[], v: number) {
  a.push(v);
  if (a.length > N) a.shift();
}

function drawPlots() {
  const ctx = plots.getContext('2d')!;
  const w = plots.width, h = plots.height;
  ctx.clearRect(0, 0, w, h);
  const plot = (a: number[], color: string, max: number, y0: number, hh: number, label: string) => {
    ctx.strokeStyle = '#444';
    ctx.strokeRect(0.5, y0 + 0.5, w - 1, hh - 1);
    ctx.strokeStyle = color;
    ctx.beginPath();
    a.forEach((v, i) => {
      const x = (i / (N - 1)) * w, y = y0 + hh - Math.min(1, v / max) * hh;
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.fillText(`${label} ${a.length ? a[a.length - 1].toFixed(2) : '-'} (max ${max})`, 4, y0 + 11);
  };
  plot(jitterSeries, '#5fd35f', 2, 0, h / 2, 'jitter mm');
  plot(latencySeries, '#f9b52d', 100, h / 2, h / 2, 'latency ms');
}

function drawOverlay(ar: ImageTracker, e: FrameEvent) {
  const dpr = devicePixelRatio || 1;
  const W = overlay.clientWidth, H = overlay.clientHeight;
  if (overlay.width !== W * dpr) { overlay.width = W * dpr; overlay.height = H * dpr; }
  const ctx = overlay.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const K = ar.intrinsics;
  if (!K) return;
  // Processing pixels -> CSS pixels for object-fit: cover.
  const s = Math.max(W / K.width, H / K.height);
  const ox = (W - K.width * s) / 2, oy = (H - K.height * s) / 2;
  for (const [i, p] of e.poses) {
    const m = ar.targets[i];
    const hw = m.physicalWidth / 2, hh = (m.physicalWidth * m.height) / m.width / 2;
    const pts = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([X, Y]) => {
      const R = p.R, t = p.t;
      const z = R[6] * X + R[7] * Y + t[2];
      const u = (K.fx * (R[0] * X + R[1] * Y + t[0])) / z + K.cx, v = (K.fy * (R[3] * X + R[4] * Y + t[1])) / z + K.cy;
      return [ox + (u + 0.5) * s, oy + (v + 0.5) * s];
    });
    ctx.lineWidth = 3;
    ctx.strokeStyle = `rgba(80, 255, 120, ${0.4 + 0.6 * p.confidence})`;
    ctx.beginPath();
    pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.stroke();
    // Jitter: std of translation over the last 30 presented frames.
    recentT.push([p.t[0], p.t[1], p.t[2]]);
    if (recentT.length > 30) recentT.shift();
  }
  if (e.poses.size === 0) recentT.length = 0;
  if (recentT.length >= 10) {
    const mean = [0, 1, 2].map((k) => recentT.reduce((a, v) => a + v[k], 0) / recentT.length);
    const v = recentT.reduce((a, x) => a + (x[0] - mean[0]) ** 2 + (x[1] - mean[1]) ** 2 + (x[2] - mean[2]) ** 2, 0) / (recentT.length - 1);
    pushSeries(jitterSeries, Math.sqrt(v) * 1000);
  }
}

function showStats(ar: ImageTracker) {
  const s = ar.stats, tm = s.timings;
  const f = (v: number | undefined) => (v === undefined ? '  -  ' : v.toFixed(2).padStart(5));
  const tg = s.targets.map((t) => `#${t.index} ${t.status} conf ${t.confidence.toFixed(2)} inl ${t.inliers} rms ${t.rms.toFixed(2)}`).join('\n');
  statsEl.textContent =
    `path ${s.path}  sensors ${s.sensors}  ${ar.processingSize.width}x${ar.processingSize.height}\n` +
    `fps ${s.fps}  latency ${s.latency.toFixed(1)} ms  t_d ${(s.td * 1000).toFixed(1)} ms ${s.tdCalibrated ? '(cal)' : ''}\n` +
    `pyramid ${f(tm?.pyramid)}  klt ${f(tm?.klt)}  ncc ${f(tm?.ncc)}  solve ${f(tm?.solve)}\n` +
    `fusion  ${f(tm?.fusion)}  track ${f(tm?.track)}  total ${f(tm?.total)}  acquire ${f(tm?.acquire)}\n` +
    (tg || 'searching…');
}

function download(name: string, data: Blob | string | Uint8Array) {
  const blob = data instanceof Blob ? data : new Blob([data as BlobPart]);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

$('start').onclick = async () => {
  try {
    const ar = await ImageTracker.create({ targets: targetsBuf ?? './targets.bin' });
    // start() requests motion permission first, inside this tap.
    await ar.start({ video, requestMotionPermission: true });
    $('setup').classList.add('hidden');
    $('panel').classList.remove('hidden');
    $('controls').classList.remove('hidden');
    ar.onResult = (r) => { if (!r.acquisition) pushSeries(latencySeries, r.doneAt - r.t); };
    ar.onFrame((e) => drawOverlay(ar, e));
    setInterval(() => { showStats(ar); drawPlots(); }, 200);
    const rec = async (luma: boolean, btn: HTMLButtonElement) => {
      if (!ar.recorder.recording) {
        ar.recorder.start({ luma });
        btn.textContent = 'Stop';
        return;
      }
      const s = await ar.recorder.stop();
      btn.textContent = luma ? 'Record + luma' : 'Record';
      const files = sessionFiles(s);
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      download(`session-${stamp}.json`, files.json);
      if (files.video) download(`session-${stamp}.${files.video.type.includes('mp4') ? 'mp4' : 'webm'}`, files.video);
      if (files.luma) download(`session-${stamp}.luma`, files.luma);
    };
    $<HTMLButtonElement>('rec').onclick = (e) => rec(false, e.target as HTMLButtonElement);
    $<HTMLButtonElement>('recLuma').onclick = (e) => rec(true, e.target as HTMLButtonElement);
  } catch (e) {
    msg.textContent = String((e as Error).message ?? e);
  }
};
