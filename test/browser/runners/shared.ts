// Shared plumbing for the Android/iOS runners. These do not use the
// Playwright test runner: Appium and Playwright's Android driver are driven
// from a normal TypeScript script and report pass/fail with the exit code.
import { spawn, type ChildProcess } from 'node:child_process';
import { strict as assert } from 'node:assert';
import { bestOffset, firstLockMs, percentile, settled, truth, visibleRatio } from '../metrics';
import type { HarnessResult } from '../types';

export const BASE = 'http://127.0.0.1:4173';

export function startServer(): ChildProcess {
  const p = spawn(process.execPath, ['test/browser/server.mjs'], {
    env: { ...process.env, HOST: '127.0.0.1', PORT: '4173' },
    stdio: ['ignore', 'inherit', 'inherit'],
  });
  p.on('exit', (code) => { if (code) console.error(`static server exited with ${code}`); });
  return p;
}

export async function waitForServer(): Promise<void> {
  for (let i = 0; i < 60; i++) {
    try {
      const res = await fetch(`${BASE}/health`);
      if (res.ok) return;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error('static server did not start');
}

export function checkReplay(name: string, r: HarnessResult, opts: { minFrames?: number; p95?: number } = {}): void {
  if (r.error) throw new Error(`${name}: harness error: ${r.error}`);
  assert.ok(r.frames.length > (opts.minFrames ?? 20), `${name}: too few frames (${r.frames.length})`);
  assert.ok(firstLockMs(r) < 2500, `${name}: slow lock (${firstLockMs(r)} ms)`);
  assert.ok(visibleRatio(r) > 0.6, `${name}: visible ratio ${visibleRatio(r).toFixed(2)}`);
  const t = truth(name);
  const errors = bestOffset(settled(r), t, 12).errors;
  assert.ok(errors.length > 10, `${name}: too few pose samples`);
  const p95 = percentile(errors, 0.95);
  assert.ok(p95 < (opts.p95 ?? 5), `${name}: overlay p95 ${p95.toFixed(2)} px`);
  console.log(`${name}: ${r.frames.length} frames, ${r.stats?.path}, lock ${firstLockMs(r).toFixed(0)} ms, p95 ${p95.toFixed(2)} px`);
}

export function checkVideo(r: HarnessResult): void {
  if (r.error) throw new Error(`video: harness error: ${r.error}`);
  assert.ok(r.frames.length > 20, 'video: too few frames');
  assert.ok(visibleRatio(r) > 0.7, `video: visible ratio ${visibleRatio(r).toFixed(2)}`);
  const poses = r.frames.flatMap((f) => f.poses);
  const locked = poses.filter((p) => p.locked).length / Math.max(1, poses.length);
  assert.ok(locked > 0.5, `video: clock locked ratio ${locked.toFixed(2)}`);
  console.log(`video: ${r.frames.length} frames, clock locked ${(locked * 100).toFixed(0)}%`);
}

export async function closeServer(p: ChildProcess | null): Promise<void> {
  if (!p) return;
  p.kill('SIGTERM');
  await new Promise((r) => setTimeout(r, 200));
}
