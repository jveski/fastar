// Android Chrome on an emulator, driven through Appium/UiAutomator2. Playwright's
// `_android` driver relies on Chrome's command-line file, which stock Google
// Play images ignore, so Appium's Chromedriver path is the reliable option.
//
// The workflow boots the emulator, sets `adb reverse tcp:4173 tcp:4173`, starts
// Appium, then executes: npx tsx test/browser/runners/android.ts
import { BASE, checkReplay, checkVideo, closeServer, startServer, waitForServer } from './shared';
import { createSession, deleteSession, navigateAndRun, waitAppium } from './appium';
import type { HarnessResult } from '../types';

function dump(label: string, r: HarnessResult): void {
  const poses = r.frames.filter((f) => f.poses.length).length;
  console.log(`${label}: error=${r.error} frames=${r.frames.length} poses=${poses}`);
  console.log(`${label}: stats=${JSON.stringify(r.stats)}`);
  console.log(`${label}: diag=${JSON.stringify(r.diagnostics)} probe=${JSON.stringify(r.probe)}`);
  if (r.logs.length) console.log(`${label}: logs=${JSON.stringify(r.logs.slice(0, 10))}`);
}

const server = startServer();
let sessionId: string | null = null;

try {
  await waitForServer();
  await waitAppium();

  sessionId = await createSession({
    platformName: 'Android',
    browserName: 'Chrome',
    'appium:automationName': 'UiAutomator2',
    'appium:deviceName': 'Android Emulator',
    'appium:platformVersion': '14',
    'appium:autoGrantPermissions': true,
    'appium:chromedriverAutodownload': true,
    'appium:newCommandTimeout': 300,
    'appium:uiautomator2ServerLaunchTimeout': 120_000,
    'appium:chromeOptions': {
      args: ['--no-first-run', '--disable-fre', '--no-default-browser-check'],
    },
  });
  console.log('Android Appium session started');

  // The emulator's WebGL2 video upload is not reliable with the old Chrome in
  // the Play image; exercise the 2D luma fallback instead.
  const sweep = await navigateAndRun(sessionId, `${BASE}/?mode=replay&scenario=sweep&no-webgl2=1&no-offscreen=1&useImu=0&rate=0.5&runMs=8000`);
  dump('sweep', sweep);
  // The emulator renders the clip at ~3 fps, so a pose can be attached to a
  // frame one media tick late; 15 px on a 640-wide frame is still a useful
  // motion/lock check. Desktop and WebKit keep the tighter thresholds.
  checkReplay('sweep', sweep, { p95: 15, minFrames: 8, settle: 3, minErrors: 2 });

  const video = await navigateAndRun(sessionId, `${BASE}/?mode=replay&scenario=video&target=targets-video.bin&no-webgl2=1&no-offscreen=1&useImu=0&rate=0.5&runMs=8000`);
  dump('video', video);
  checkVideo(video, { minFrames: 3, locked: 0.3 });

  const stream = await navigateAndRun(sessionId, `${BASE}/?mode=stream&scenario=static&useImu=0&no-webgl2=1&no-offscreen=1&runMs=6000`);
  dump('stream', stream);
  if (stream.error) throw new Error(`stream: ${stream.error}`);
  if (stream.frames.length < 8) throw new Error(`stream: too few frames (${stream.frames.length})`);
  console.log(`stream: ${stream.frames.length} frames, ${stream.stats?.path}`);

  console.log('android browser suite passed');
} finally {
  if (sessionId) await deleteSession(sessionId);
  await closeServer(server);
}
