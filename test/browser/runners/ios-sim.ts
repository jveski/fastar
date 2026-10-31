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

function pickSimulator(): { udid: string; name: string; version: string } {
  const out = execFileSync('xcrun', ['simctl', 'list', 'devices', 'available', '-j'], { encoding: 'utf8' });
  const data = JSON.parse(out) as { devices: Record<string, Array<{ udid: string; name: string; state: string }>> };
  const runtimes = Object.keys(data.devices)
    .filter((k) => /iOS[-.]/.test(k))
    .sort()
    .reverse();
  for (const runtime of runtimes) {
    const phone = data.devices[runtime].find((d) => d.name.startsWith('iPhone'));
    if (phone) {
      const match = runtime.match(/iOS-(\d+)-(\d+)/);
      const version = match ? `${match[1]}.${match[2]}` : '26.0';
      return { udid: phone.udid, name: phone.name, version };
    }
  }
  throw new Error('no available iPhone simulator runtime; install one in Xcode');
}

/** Boot the simulator and wait for SpringBoard, with a hard ceiling. */
function bootSimulator(udid: string, erase: boolean): void {
  try { execFileSync('xcrun', ['simctl', 'shutdown', 'all'], { stdio: 'ignore' }); } catch { /* none booted */ }
  if (erase) {
    try { execFileSync('xcrun', ['simctl', 'erase', udid], { stdio: 'ignore' }); } catch { /* best effort */ }
  }
  try { execFileSync('xcrun', ['simctl', 'boot', udid], { stdio: 'ignore' }); } catch { /* already booted */ }
  execFileSync('xcrun', ['simctl', 'bootstatus', udid, '-b'], { stdio: 'inherit', timeout: 600_000 });
}

async function startSession(sim: { udid: string; version: string }): Promise<string> {
  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      return await createSession({
        platformName: 'iOS',
        browserName: 'Safari',
        'appium:automationName': 'XCUITest',
        'appium:udid': sim.udid,
        'appium:platformVersion': sim.version,
        'appium:autoAcceptAlerts': true,
        // Force a real tab: a freshly wiped simulator can sit on Safari's
        // welcome screen, where the remote debugger has no web application.
        'appium:safariInitialUrl': `${BASE}/`,
        'appium:webviewConnectTimeout': 60_000,
        'appium:newCommandTimeout': 300,
        'appium:simulatorStartupTimeout': 600_000,
        'appium:wdaLaunchTimeout': 600_000,
        'appium:wdaConnectionTimeout': 600_000,
        'appium:showXcodeLog': true,
      });
    } catch (e) {
      const message = (e as Error).message;
      console.log(`iOS session attempt ${attempt} failed: ${message.slice(0, 500)}`);
      if (attempt === 2) throw e;
      // A corrupt boot needs a wipe; a Safari/remote-debugger miss only needs
      // a restart (erasing would put Safari back on its welcome screen).
      bootSimulator(sim.udid, /did not finish booting/i.test(message));
    }
  }
  throw new Error('unreachable');
}

const server = startServer();
let sessionId: string | null = null;

try {
  await waitForServer();
  await waitAppium();
  const sim = pickSimulator();
  console.log(`iOS simulator: ${sim.name} ${sim.udid} (${sim.version})`);
  bootSimulator(sim.udid, false);

  sessionId = await startSession(sim);
  console.log('iOS Appium session started');

  // Simulator Safari cannot ask for motion permission, and there is no camera.
  // The simulator's media clock is unreliable under automation, so use the
  // rAF fallback; the truth is still mapped by mediaTime.
  const sweep = await navigateAndRun(sessionId, `${BASE}/?mode=replay&scenario=sweep&useImu=0&norvfc=1`);
  checkReplay('sweep', sweep, { p95: 8, minFrames: 8, settle: 3, minErrors: 3 });

  const video = await navigateAndRun(sessionId, `${BASE}/?mode=replay&scenario=video&target=targets-video.bin&useImu=0&norvfc=1`);
  console.log(`video: ${video.frames.length} frames, ${JSON.stringify(video.stats)}`);
  checkVideo(video, { minFrames: 3, locked: 0.3 });

  // Canvas captureStream is optional in Mobile Safari; replay is the required
  // tracker check here. Exercise it when the runtime exposes it.
  const stream = await navigateAndRun(sessionId, `${BASE}/?mode=stream&scenario=static&useImu=0&norvfc=1`);
  console.log(`stream: ${stream.frames.length} frames, ${JSON.stringify(stream.stats)}`);
  if (stream.error) {
    console.log(`stream skipped: ${stream.error}`);
  } else {
    assert.ok(stream.frames.length > 3, `stream: too few frames (${stream.frames.length})`);
  }

  console.log('iOS Safari suite passed');
} finally {
  if (sessionId) await deleteSession(sessionId);
  await closeServer(server);
}
