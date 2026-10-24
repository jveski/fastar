// The built site has fastar.js next to this file; served from the repository root
// (site/compile.html), it comes from dist/ (run npm run build first).
const fromSource = /\/site\//.test(location.pathname);
const { compileImage, compileVideo, encodeTargets } = await import(new URL(fromSource ? '../dist/fastar.js' : './fastar.js', import.meta.url).href);

const $ = (id) => document.getElementById(id);
const items = [];
let outputUrl = null;

function addFiles(files) {
  for (const file of files) {
    const video = file.type.startsWith('video/');
    if (!video && !file.type.startsWith('image/')) continue;
    const item = { file, video, url: URL.createObjectURL(file), name: file.name.replace(/\.[^.]+$/, ''), widthCm: video ? 100 : 20, aspect: 0, fps: '' };
    if (video) {
      const v = Object.assign(document.createElement('video'), { muted: true, preload: 'metadata', src: item.url });
      v.onloadedmetadata = () => {
        item.aspect = v.videoHeight / v.videoWidth;
        item.duration = v.duration;
        render();
      };
    } else {
      const img = new Image();
      img.onload = () => {
        item.aspect = img.naturalHeight / img.naturalWidth;
        render();
      };
      img.src = item.url;
    }
    items.push(item);
  }
  render();
}

function render() {
  $('images').replaceChildren(...items.map((item, i) => {
    const li = document.createElement('li');
    const img = item.video
      ? Object.assign(document.createElement('video'), { src: `${item.url}#t=0.1`, muted: true, preload: 'metadata' })
      : Object.assign(document.createElement('img'), { src: item.url, alt: '' });
    const name = Object.assign(document.createElement('span'), { className: 'name', textContent: item.name });
    if (item.video && item.duration) name.append(Object.assign(document.createElement('span'), { className: 'muted', textContent: ` · ${item.duration.toFixed(1)} s video` }));
    if (items.length > 1) name.prepend(Object.assign(document.createElement('span'), { className: 'muted', textContent: `#${i} ` }));

    const size = document.createElement('span');
    const height = document.createElement('span');
    height.className = 'muted';
    const showHeight = () => {
      height.textContent = item.aspect && item.widthCm > 0 ? ` × ${(item.widthCm * item.aspect).toFixed(1)} cm` : ' cm';
    };
    const width = Object.assign(document.createElement('input'), {
      type: 'number', value: String(item.widthCm), min: '0.1', step: '0.1', title: item.video ? 'Width on screen (cm)' : 'Printed width (cm)',
    });
    width.oninput = () => {
      item.widthCm = parseFloat(width.value);
      showHeight();
      changed();
    };
    showHeight();
    size.append(width, height);
    if (item.video) {
      const fps = Object.assign(document.createElement('input'), {
        type: 'number', value: item.fps, min: '1', step: '0.001', placeholder: 'auto', title: 'Frame rate (blank: detect)',
      });
      fps.className = 'fps';
      fps.oninput = () => {
        item.fps = fps.value;
        changed();
      };
      size.append(' ', fps, Object.assign(document.createElement('span'), { className: 'muted', textContent: ' fps' }));
    }

    const remove = Object.assign(document.createElement('button'), { className: 'remove', textContent: '✕', title: 'Remove' });
    remove.onclick = () => {
      URL.revokeObjectURL(item.url);
      items.splice(items.indexOf(item), 1);
      render();
      changed();
    };
    li.append(img, name, size, remove);
    return li;
  }));
  $('compile').disabled = items.length === 0;
}

/** Inputs changed: the last result no longer matches them. */
function changed() {
  $('result').hidden = true;
  $('status').textContent = '';
}

$('files').onchange = (e) => {
  addFiles(e.target.files);
  e.target.value = '';
  changed();
};
const drop = $('drop');
drop.ondragover = (e) => {
  e.preventDefault();
  drop.classList.add('over');
};
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove('over');
  addFiles(e.dataTransfer.files);
  changed();
};

$('compile').onclick = async () => {
  const status = $('status');
  const bad = items.find((it) => !(it.widthCm > 0));
  if (bad) {
    status.textContent = `Enter a width for ${bad.name}.`;
    return;
  }
  $('compile').disabled = true;
  changed();
  status.textContent = 'Compiling…';
  // Let the status paint: compiling blocks the main thread.
  await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
  try {
    const targets = [];
    for (const it of items) {
      const o = { physicalWidth: it.widthCm / 100, name: it.name };
      if (it.video) {
        status.textContent = `Compiling ${it.name}…`;
        targets.push(await compileVideo(it.file, {
          ...o,
          fps: parseFloat(it.fps) || undefined,
          onProgress: (done, total) => { status.textContent = `Compiling ${it.name}: frame ${done} of ${total}`; },
        }));
        it.poster = await firstFrame(it.url);
      } else {
        targets.push(await compileImage(it.file, o));
      }
    }
    const buffer = encodeTargets(targets);
    if (outputUrl) URL.revokeObjectURL(outputUrl);
    outputUrl = URL.createObjectURL(new Blob([buffer], { type: 'application/octet-stream' }));
    $('download').href = outputUrl;
    $('download').textContent = `Download targets.bin (${formatBytes(buffer.byteLength)})`;
    $('try').href = `demo/?targets=${encodeURIComponent(outputUrl)}`;
    $('targets').replaceChildren(...targets.map((t, i) => report(t, i, targets.length > 1, items[i])));
    $('result').hidden = false;
    status.textContent = '';
  } catch (e) {
    status.textContent = `Failed: ${e.message}`;
    console.error(e);
  } finally {
    $('compile').disabled = items.length === 0;
  }
};

function report(t, index, numbered, item) {
  const q = t.meta.quality, v = t.meta.video;
  const el = document.createElement('div');
  el.className = 'target';
  const summary = document.createElement('p');
  const title = Object.assign(document.createElement('strong'), { textContent: `${numbered ? `#${index} ` : ''}${t.meta.name}` });
  const verdict = q.warnings.length
    ? Object.assign(document.createElement('span'), { className: 'warn', textContent: q.warnings.join('; ') })
    : Object.assign(document.createElement('span'), { className: 'ok', textContent: 'Looks good' });
  const facts = Object.assign(document.createElement('span'), {
    className: 'muted',
    textContent: v
      ? ` (${v.duration.toFixed(1)} s at ${+v.fps.toFixed(3)} fps, ${q.features} features over ${v.keyframes.length} keyframes, ` +
        `${v.report.points} tracking points, at least ${v.report.minPoints} on screen, ${v.changes.length} cut${v.changes.length === 1 ? '' : 's'})`
      : ` (${q.features} features, ${Math.round(q.coverage * 100)}% coverage, ${q.trackingPoints} tracking points)`,
  });
  summary.append(title, ': ', verdict, facts);
  el.append(summary, v ? videoPreview(t, item.poster) : preview(t));
  return el;
}

/** First frame of a video, as a canvas. */
async function firstFrame(url) {
  const v = Object.assign(document.createElement('video'), { muted: true, playsInline: true, preload: 'auto', src: url });
  await new Promise((r, j) => { v.onloadeddata = r; v.onerror = j; });
  v.currentTime = 0.5 / 30;
  await new Promise((r) => { v.onseeked = r; });
  const c = Object.assign(document.createElement('canvas'), { width: v.videoWidth, height: v.videoHeight });
  c.getContext('2d').drawImage(v, 0, 0);
  return c;
}

/** First frame with the features and tracking points on screen then. */
function videoPreview(t, poster) {
  const { width, height } = t.meta;
  const canvas = Object.assign(document.createElement('canvas'), { width, height });
  const ctx = canvas.getContext('2d');
  if (poster) {
    ctx.filter = 'grayscale(1) contrast(0.75) brightness(1.1)';
    ctx.drawImage(poster, 0, 0, width, height);
    ctx.filter = 'none';
  }
  const r = Math.max(2, width / 300);
  ctx.lineWidth = Math.max(1, r / 2);
  ctx.strokeStyle = '#ff9f1c';
  const range = t.video.featureRange;
  for (let i = 0; i < t.kp.length / 4; i++) {
    if (range[2 * i] !== 0) continue;
    ctx.beginPath();
    ctx.arc(t.kp[4 * i] + 0.5, t.kp[4 * i + 1] + 0.5, r * Math.sqrt(t.kp[4 * i + 2]), 0, 2 * Math.PI);
    ctx.stroke();
  }
  ctx.fillStyle = '#2ec4b6';
  const pts = t.video.points, path = t.video.path;
  for (let j = 0; j < pts.length; j += 8) {
    if (pts[j] !== 0) continue;
    const o = pts[j + 2] * 3;
    ctx.fillRect(path[o + 1] + 0.5 - r, path[o + 2] + 0.5 - r, 2 * r, 2 * r);
  }
  return canvas;
}

/** The largest tracking image with features and tracking points drawn on it. */
function preview(t) {
  const k = t.images.length - 1;
  const { width, height, scale } = t.meta.images[k];
  const canvas = Object.assign(document.createElement('canvas'), { width, height });
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(width, height);
  const g = t.images[k];
  for (let i = 0, j = 0; i < g.length; i++, j += 4) {
    // Fade the image so the markers stand out.
    const v = 64 + ((g[i] * 3) >> 2);
    img.data[j] = img.data[j + 1] = img.data[j + 2] = v;
    img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const s = 1 / scale;
  const r = Math.max(2, width / 300);
  ctx.lineWidth = Math.max(1, r / 2);
  ctx.strokeStyle = '#ff9f1c';
  for (let i = 0; i < t.kp.length; i += 4) {
    ctx.beginPath();
    ctx.arc((t.kp[i] + 0.5) * s, (t.kp[i + 1] + 0.5) * s, r * Math.sqrt(t.kp[i + 2]), 0, 2 * Math.PI);
    ctx.stroke();
  }
  ctx.fillStyle = '#2ec4b6';
  for (let i = 0; i < t.trackPoints.length; i += 2) {
    ctx.fillRect((t.trackPoints[i] + 0.5) * s - r, (t.trackPoints[i + 1] + 0.5) * s - r, 2 * r, 2 * r);
  }
  return canvas;
}

function formatBytes(n) {
  return n > 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.round(n / 1024)} KB`;
}
