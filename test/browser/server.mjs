// Tiny static server for the browser harness. Supports Range requests so
// <video> can seek/loop the fixture MP4s.
import { createServer } from 'node:http';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { extname, join, normalize, resolve } from 'node:path';

const ROOT = resolve('test/browser/public');
const PORT = Number(process.env.PORT ?? 4173);
const HOST = process.env.HOST ?? '127.0.0.1';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.mp4': 'video/mp4',
  '.y4m': 'video/x-yuv4mpeg',
  '.png': 'image/png',
  '.bin': 'application/octet-stream',
};

function safePath(url) {
  const path = decodeURIComponent((url ?? '/').split('?')[0]);
  const p = normalize(join(ROOT, path));
  return p.startsWith(ROOT) ? p : null;
}

const server = createServer((req, res) => {
  if (req.url === '/health') {
    res.writeHead(200, { 'content-type': 'text/plain' });
    res.end('ok');
    return;
  }
  let path = safePath(req.url);
  if (!path) { res.writeHead(403); res.end(); return; }
  if (existsSync(path) && statSync(path).isDirectory()) path = join(path, 'index.html');
  if (!existsSync(path)) { res.writeHead(404); res.end('not found'); return; }

  const st = statSync(path);
  const type = MIME[extname(path)] ?? 'application/octet-stream';
  res.setHeader('cache-control', 'no-store');
  res.setHeader('accept-ranges', 'bytes');
  res.setHeader('content-type', type);

  const range = req.headers.range;
  if (range) {
    const m = /^bytes=(\d*)-(\d*)$/.exec(range);
    if (m) {
      const start = m[1] ? Number(m[1]) : 0;
      const end = m[2] ? Math.min(Number(m[2]), st.size - 1) : st.size - 1;
      if (start > end || start >= st.size) {
        res.writeHead(416, { 'content-range': `bytes */${st.size}` });
        res.end();
        return;
      }
      res.writeHead(206, {
        'content-range': `bytes ${start}-${end}/${st.size}`,
        'content-length': end - start + 1,
      });
      createReadStream(path, { start, end }).pipe(res);
      return;
    }
  }
  res.writeHead(200, { 'content-length': st.size });
  createReadStream(path).pipe(res);
});

server.listen(PORT, HOST, () => {
  console.log(`browser harness: http://${HOST}:${PORT}/`);
});
