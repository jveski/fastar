import { compileImages } from './fastar.js';

const $ = (id) => document.getElementById(id);
const items = [];
let outputUrl = null;

function addFiles(files) {
  for (const file of files) {
    if (!file.type.startsWith('image/')) continue;
    const item = { file, url: URL.createObjectURL(file), name: file.name.replace(/\.[^.]+$/, ''), widthCm: 20, aspect: 0 };
    const img = new Image();
    img.onload = () => {
      item.aspect = img.naturalHeight / img.naturalWidth;
      render();
    };
    img.src = item.url;
    items.push(item);
  }
  render();
}

function render() {
  $('images').replaceChildren(...items.map((item, i) => {
    const li = document.createElement('li');
    const img = Object.assign(document.createElement('img'), { src: item.url, alt: '' });
    const name = Object.assign(document.createElement('span'), { className: 'name', textContent: item.name });
    if (items.length > 1) name.prepend(Object.assign(document.createElement('span'), { className: 'muted', textContent: `#${i} ` }));

    const size = document.createElement('span');
    const height = document.createElement('span');
    height.className = 'muted';
    const showHeight = () => {
      height.textContent = item.aspect && item.widthCm > 0 ? ` × ${(item.widthCm * item.aspect).toFixed(1)} cm` : ' cm';
    };
    const width = Object.assign(document.createElement('input'), {
      type: 'number', value: String(item.widthCm), min: '0.1', step: '0.1', title: 'Printed width (cm)',
    });
    width.oninput = () => {
      item.widthCm = parseFloat(width.value);
      showHeight();
      changed();
    };
    showHeight();
    size.append(width, height);

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
    const { buffer, targets } = await compileImages(
      items.map((it) => ({ image: it.file, physicalWidth: it.widthCm / 100, name: it.name })),
    );
    if (outputUrl) URL.revokeObjectURL(outputUrl);
    outputUrl = URL.createObjectURL(new Blob([buffer], { type: 'application/octet-stream' }));
    $('download').href = outputUrl;
    $('download').textContent = `Download targets.bin (${formatBytes(buffer.byteLength)})`;
    $('try').href = `demo/?targets=${encodeURIComponent(outputUrl)}`;
    $('targets').replaceChildren(...targets.map((t, i) => report(t, i, targets.length > 1)));
    $('result').hidden = false;
    status.textContent = '';
  } catch (e) {
    status.textContent = `Failed: ${e.message}`;
    console.error(e);
  } finally {
    $('compile').disabled = items.length === 0;
  }
};

function report(t, index, numbered) {
  const q = t.meta.quality;
  const el = document.createElement('div');
  el.className = 'target';
  const summary = document.createElement('p');
  const title = Object.assign(document.createElement('strong'), { textContent: `${numbered ? `#${index} ` : ''}${t.meta.name}` });
  const verdict = q.warnings.length
    ? Object.assign(document.createElement('span'), { className: 'warn', textContent: q.warnings.join('; ') })
    : Object.assign(document.createElement('span'), { className: 'ok', textContent: 'Looks good' });
  const facts = Object.assign(document.createElement('span'), {
    className: 'muted',
    textContent: ` (${q.features} features, ${Math.round(q.coverage * 100)}% coverage, ${q.trackingPoints} tracking points)`,
  });
  summary.append(title, ': ', verdict, facts);
  el.append(summary, preview(t));
  return el;
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
