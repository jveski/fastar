// Builds:
//   dist/fastar.js          single-file ESM library (worker code inlined)
//   dist/fastar-compile.js  Node CLI target compiler
//   dist/types/             type declarations
//   dist/bench/             on-device benchmark page
//   dist/site/              GitHub Pages site: target compiler and demo
import * as esbuild from 'esbuild';
import { execFileSync } from 'node:child_process';
import { gzipSync } from 'node:zlib';
import { mkdirSync, readFileSync, writeFileSync, copyFileSync, cpSync, rmSync } from 'node:fs';

const PAYLOAD_LIMIT = 400 * 1024;

rmSync('dist', { recursive: true, force: true });
mkdirSync('dist/bench', { recursive: true });

async function buildWorkerSource() {
  const result = await esbuild.build({
    entryPoints: ['src/worker/worker-main.ts'],
    bundle: true,
    format: 'iife',
    target: 'es2020',
    minify: true,
    write: false,
    legalComments: 'none',
  });
  return result.outputFiles[0].text;
}

function inlineWorkerPlugin(workerSource) {
  return {
    name: 'inline-worker',
    setup(build) {
      build.onResolve({ filter: /^virtual:worker-source$/ }, () => ({
        path: 'worker-source',
        namespace: 'inline-worker',
      }));
      build.onLoad({ filter: /.*/, namespace: 'inline-worker' }, () => ({
        contents: `export default ${JSON.stringify(workerSource)};`,
        loader: 'js',
      }));
    },
  };
}

const workerSource = await buildWorkerSource();
const plugins = [inlineWorkerPlugin(workerSource)];

await esbuild.build({
  entryPoints: ['src/index.ts'],
  bundle: true,
  format: 'esm',
  target: 'es2020',
  minify: true,
  sourcemap: false,
  outfile: 'dist/fastar.js',
  legalComments: 'none',
  plugins,
});

await esbuild.build({
  entryPoints: ['cli/compile.ts'],
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node18',
  outfile: 'dist/fastar-compile.js',
  banner: { js: '#!/usr/bin/env node' },
  plugins,
});

await esbuild.build({
  entryPoints: ['bench/page/main.ts'],
  bundle: true,
  format: 'esm',
  target: 'es2020',
  minify: true,
  outfile: 'dist/bench/main.js',
  plugins,
});
copyFileSync('bench/page/index.html', 'dist/bench/index.html');

cpSync('site', 'dist/site', { recursive: true });
copyFileSync('dist/fastar.js', 'dist/site/fastar.js');
copyFileSync('examples/basic/target.png', 'dist/site/demo/target.png');
copyFileSync('examples/basic/targets.bin', 'dist/site/demo/targets.bin');

execFileSync('npx', ['tsc', '-p', 'tsconfig.build.json'], { stdio: 'inherit' });

const lib = readFileSync('dist/fastar.js');
const gz = gzipSync(lib, { level: 9 }).length;
const report = {
  bytes: lib.length,
  gzipBytes: gz,
  workerBytes: workerSource.length,
  limitGzipBytes: PAYLOAD_LIMIT,
};
writeFileSync('dist/size.json', JSON.stringify(report, null, 2));
console.log(
  `dist/fastar.js: ${(lib.length / 1024).toFixed(1)} KB, ${(gz / 1024).toFixed(1)} KB gzipped ` +
    `(limit ${PAYLOAD_LIMIT / 1024} KB)`,
);
if (gz > PAYLOAD_LIMIT) {
  console.error('Payload budget exceeded');
  process.exit(1);
}
