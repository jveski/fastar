// Android Chrome on an emulator, driven through Playwright's Android support.
// The workflow runs an emulator, sets `adb reverse tcp:4173 tcp:4173`, then
// executes: npx tsx test/browser/runners/android.ts
import { _android as android } from '@playwright/test';
import { BASE, checkReplay, checkVideo, closeServer, startServer, waitForServer } from './shared';
import { runHarness } from '../metrics';

type AndroidDevice = Awaited<ReturnType<typeof android.devices>>[number];
type AndroidContext = Awaited<ReturnType<AndroidDevice['launchBrowser']>>;

const server = startServer();
let device: AndroidDevice | null = null;
let context: AndroidContext | null = null;

try {
  await waitForServer();
  const [d] = await android.devices();
  if (!d) throw new Error('no Android device: check adb devices and KVM');
  device = d;
  console.log(`Android device: ${d.model()} (${d.serial()})`);

  // No Chrome command-line flags here: this is the baseline path that works
  // on stock, non-rooted Android. The app is served over adb reverse.
  context = await d.launchBrowser({ baseURL: BASE });
  const page = await context.newPage();
  page.setDefaultTimeout(60_000);

  const sweep = await runHarness(page, { mode: 'replay', scenario: 'sweep' });
  checkReplay('sweep', sweep, { p95: 6 });

  const video = await runHarness(page, { mode: 'replay', scenario: 'video', target: 'targets-video.bin' });
  checkVideo(video);

  const stream = await runHarness(page, { mode: 'stream', scenario: 'static', useImu: '0' });
  if (stream.error) throw new Error(`stream: ${stream.error}`);
  if (stream.frames.length < 20) throw new Error(`stream: too few frames (${stream.frames.length})`);
  console.log(`stream: ${stream.frames.length} frames, ${stream.stats?.path}`);

  await context.close();
  context = null;
  console.log('android browser suite passed');
} finally {
  await context?.close().catch(() => {});
  await device?.close().catch(() => {});
  await closeServer(server);
}
