// Bundle the harness and assemble the directory Playwright serves.
// Run after `npm run build` (test:browser does both).
import * as esbuild from 'esbuild';
import { copyFileSync, existsSync, mkdirSync, cpSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

const OUT = 'test/browser/public';
rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

await esbuild.build({
  entryPoints: ['test/browser/harness/main.ts'],
  bundle: true,
  format: 'esm',
  target: 'es2020',
  outfile: `${OUT}/harness.js`,
  external: ['/fastar.js'],
  legalComments: 'none',
});

copyFileSync('test/browser/harness.html', `${OUT}/index.html`);
copyFileSync('dist/fastar.js', `${OUT}/fastar.js`);
cpSync('test/browser/fixtures/data', `${OUT}/fixtures`, { recursive: true });
if (existsSync('test/browser/fixtures/golden')) {
  cpSync('test/browser/fixtures/golden', `${OUT}/fixtures/golden`, { recursive: true });
}

// Chromium's fake camera needs Y4M. It is large, so generate it into the
// served directory instead of committing it; the camera spec skips if absent.
const y4m = `${OUT}/fixtures/sweep.y4m`;
if (existsSync('test/browser/fixtures/data/sweep.mp4')) {
  try {
    execFileSync('ffmpeg', [
      '-hide_banner', '-loglevel', 'error', '-y',
      '-i', 'test/browser/fixtures/data/sweep.mp4',
      '-pix_fmt', 'yuv420p', '-f', 'yuv4mpegpipe', y4m,
    ], { stdio: 'inherit' });
  } catch {
    console.warn('ffmpeg unavailable: camera fake-video test will be skipped');
  }
}

console.log(`browser harness built in ${OUT}`);
