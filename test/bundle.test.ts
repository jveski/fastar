import { describe, expect, it } from 'vitest';
import * as esbuild from 'esbuild';
import vm from 'node:vm';
import { encodeTargets } from '../src/target/format';
import { targetFor } from '../src/sim/harness';
import { K, POSTER_W_M, getPoster } from './fixtures';

describe('bundled worker', () => {
  it('evaluates as a classic worker script and answers init + frame', async () => {
    const out = await esbuild.build({
      entryPoints: ['src/worker/worker-main.ts'], bundle: true, format: 'iife', target: 'es2020', minify: true, write: false,
    });
    const src = out.outputFiles[0].text;
    const posted: Array<{ type: string }> = [];
    const self: { onmessage: ((e: { data: unknown }) => void) | null; postMessage: (m: { type: string }) => void } = {
      onmessage: null,
      postMessage: (m) => posted.push(m),
    };
    vm.runInNewContext(src, { self, performance, MessageChannel, setTimeout, console, TextDecoder, TextEncoder, Math, Float64Array });
    expect(typeof self.onmessage).toBe('function');
    const ch = new MessageChannel();
    self.onmessage!({
      data: {
        type: 'init', targets: encodeTargets([targetFor(getPoster(), POSTER_W_M)]), width: 640, height: 480, K,
        maxTrack: 1, useImu: false, mainTimeOrigin: performance.timeOrigin, acquirePort: ch.port1,
      },
    });
    self.onmessage!({ data: { type: 'frame', id: 1, t: 0, luma: new ArrayBuffer(640 * 480) } });
    expect(posted.map((m) => m.type)).toEqual(['ready', 'result']);
    ch.port1.close();
  });
});
