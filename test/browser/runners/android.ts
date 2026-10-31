// Android Chrome on an emulator, driven through Appium/UiAutomator2. Playwright's
// `_android` driver relies on Chrome's command-line file, which stock Google
// Play images ignore, so Appium's Chromedriver path is the reliable option.
//
// The workflow boots the emulator, sets `adb reverse tcp:4173 tcp:4173`, starts
// Appium, then executes: npx tsx test/browser/runners/android.ts
import { BASE, checkReplay, checkVideo, closeServer, startServer, waitForServer } from './shared';
import { createSession, deleteSession, navigateAndRun, waitAppium } from './appium';

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

  const sweep = await navigateAndRun(sessionId, `${BASE}/?mode=replay&scenario=sweep`);
  checkReplay('sweep', sweep, { p95: 6 });

  const video = await navigateAndRun(sessionId, `${BASE}/?mode=replay&scenario=video&target=targets-video.bin`);
  checkVideo(video);

  const stream = await navigateAndRun(sessionId, `${BASE}/?mode=stream&scenario=static&useImu=0`);
  if (stream.error) throw new Error(`stream: ${stream.error}`);
  if (stream.frames.length < 20) throw new Error(`stream: too few frames (${stream.frames.length})`);
  console.log(`stream: ${stream.frames.length} frames, ${stream.stats?.path}`);

  console.log('android browser suite passed');
} finally {
  if (sessionId) await deleteSession(sessionId);
  await closeServer(server);
}
