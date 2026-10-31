// Real Mobile Safari on an iOS Simulator, driven through Appium/XCUITest.
// Simulators have no camera or motion hardware, so this checks replay and
// synthetic-stream paths in Safari's actual WebKit engine.
//
// Prerequisites (the workflow installs these): appium + the xcuitest driver,
// an Appium server on :4723, and Xcode with an iOS runtime.
import { execFileSync } from 'node:child_process';
import { strict as assert } from 'node:assert';
import { BASE, checkReplay, checkVideo, closeServer, startServer, waitForServer } from './shared';
import { createSession, deleteSession, navigateAndRun, waitAppium } from './appium';

function pickSimulator(): string {
  const out = execFileSync('xcrun', ['simctl', 'list', 'devices', 'available', '-j'], { encoding: 'utf8' });
  const data = JSON.parse(out) as { devices: Record<string, Array<{ udid: string; name: string; state: string }>> };
  const runtimes = Object.keys(data.devices)
    .filter((k) => /iOS[-.]/.test(k))
    .sort()
    .reverse();
  for (const runtime of runtimes) {
    const phone = data.devices[runtime].find((d) => d.name.startsWith('iPhone'));
    if (phone) {
      try { execFileSync('xcrun', ['simctl', 'boot', phone.udid], { stdio: 'ignore' }); } catch { /* already booted */ }
      execFileSync('xcrun', ['simctl', 'bootstatus', phone.udid, '-b'], { stdio: 'inherit' });
      console.log(`iOS simulator: ${phone.name} (${runtime}) ${phone.udid}`);
      return phone.udid;
    }
  }
  throw new Error('no available iPhone simulator runtime; install one in Xcode');
}

const server = startServer();
let sessionId: string | null = null;

try {
  await waitForServer();
  await waitAppium();
  const udid = pickSimulator();

  sessionId = await createSession({
    platformName: 'iOS',
    browserName: 'Safari',
    'appium:automationName': 'XCUITest',
    'appium:udid': udid,
    'appium:autoAcceptAlerts': true,
    'appium:newCommandTimeout': 300,
    'appium:simulatorStartupTimeout': 300_000,
    'appium:wdaLaunchTimeout': 300_000,
    'appium:wdaConnectionTimeout': 300_000,
    'appium:showXcodeLog': true,
  });
  console.log('iOS Appium session started');

  // Simulator Safari cannot ask for motion permission, and there is no camera.
  const sweep = await navigateAndRun(sessionId, `${BASE}/?mode=replay&scenario=sweep&useImu=0`);
  checkReplay('sweep', sweep, { p95: 6 });

  const video = await navigateAndRun(sessionId, `${BASE}/?mode=replay&scenario=video&target=targets-video.bin&useImu=0`);
  checkVideo(video);

  const stream = await navigateAndRun(sessionId, `${BASE}/?mode=stream&scenario=static&useImu=0`);
  if (stream.error) throw new Error(`stream: ${stream.error}`);
  assert.ok(stream.frames.length > 20, `stream: too few frames (${stream.frames.length})`);
  console.log(`stream: ${stream.frames.length} frames, ${stream.stats?.path}`);

  console.log('iOS Safari suite passed');
} finally {
  if (sessionId) await deleteSession(sessionId);
  await closeServer(server);
}
