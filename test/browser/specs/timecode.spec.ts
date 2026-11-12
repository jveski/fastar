import { expect, test, type Page } from '@playwright/test';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { runHarness } from '../metrics';

interface Sample { at: number; time: number | null; std: number | null; locked: boolean; clear: number; snr: number; sampleRate: number; error: string | null; calibration: string; latencySource: string }

const fixture = join(process.cwd(), 'test/browser/public/fixtures/timecode.wav');

async function listen(page: Page, query = ''): Promise<{ logs: string[]; samples: Sample[] }> {
  await page.goto(`/timecode.html${query}`);
  await page.waitForFunction(() => typeof (window as unknown as { __timecode?: unknown }).__timecode !== 'undefined');
  await page.locator('#start').click();
  await page.waitForFunction(() => (window as unknown as { __timecode: { done: boolean } }).__timecode.done, undefined, { timeout: 75_000 });
  return page.evaluate(() => (window as unknown as { __timecode: { logs: string[]; samples: Sample[] } }).__timecode);
}

/** Locks, reads a set time in the fixture's range, and keeps time with the page clock. */
function expectLocked(r: { logs: string[]; samples: Sample[] }, slack: number): void {
  expect(r.logs).toEqual([]);
  const locked = r.samples.filter((s) => s.locked);
  expect(locked.length, `never locked: ${JSON.stringify(r.samples.at(-1))}`).toBeGreaterThan(8);
  for (const s of locked) {
    expect(s.time!).toBeGreaterThan(100);
    expect(s.time!).toBeLessThan(140);
  }
  // Set time minus page time is constant while locked (the fixture plays in real time).
  const d = locked.map((s) => s.time! - s.at / 1000);
  expect(Math.max(...d) - Math.min(...d)).toBeLessThan(slack);
}

test.describe('audio timecode', () => {
  test.skip(!existsSync(fixture), 'timecode fixture was not generated (run test/browser/build.mjs)');

  test('locks onto timecode played into a stream', async ({ page }, testInfo) => {
    test.skip(testInfo.project.name === 'audio-chromium', 'covered by the microphone test there');
    expectLocked(await listen(page), 0.03);
  });

  test('waits for the microphone delay to be measured, and goes on without it when it cannot be', async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'chromium-desktop' && testInfo.project.name !== 'webkit-desktop', 'desktop engines only');
    test.setTimeout(90_000);
    const r = await listen(page, '?calibrate=1&ms=45000');
    expect(r.samples.filter((s) => s.locked && s.calibration === 'measuring')).toEqual([]);
    expect(r.samples.some((s) => s.calibration === 'measuring')).toBe(true);
    expect(r.samples.at(-1)!.calibration).toBe('failed');
    expectLocked(r, 0.03);
  });

  test('locks onto timecode through the microphone', async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'audio-chromium', 'only the fake-microphone Chromium project runs this');
    const r = await listen(page, '?mic=1');
    expectLocked(r, 0.03);
    expect(r.samples.at(-1)!.sampleRate).toBeGreaterThanOrEqual(44100);
  });

  test('gives each camera frame its set time', async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'audio-chromium', 'only the fake-microphone Chromium project runs this');
    test.skip(!existsSync(join(process.cwd(), 'test/browser/public/fixtures/sweep.y4m')), 'fake camera fixture was not generated (needs ffmpeg)');
    const r = await runHarness(page, { mode: 'camera', scenario: 'sweep', useImu: '0', timecode: '1', runMs: 15000 });
    expect(r.error).toBeNull();
    const locked = r.frames.filter((f) => f.timecode?.locked);
    expect(locked.length).toBeGreaterThan(30);
    // Set time advances with the frames' capture times.
    const d = locked.map((f) => f.timecode!.time - f.captureTime / 1000);
    expect(Math.max(...d) - Math.min(...d)).toBeLessThan(0.03);
    expect(locked[0].timecode!.time).toBeGreaterThan(100);
  });
});
