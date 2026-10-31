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
  // Create a webview before Appium connects; the remote debugger occasionally
  // finds Safari's welcome screen and then reports no web applications.
  try { execFileSync('xcrun', ['simctl', 'openurl', udid, `${BASE}/`], { stdio: 'ignore' }); } catch { /* best effort */ }
}

async function startSession(sim: { udid: string; version: string }): Promise<string> {
  const maxAttempts = 3;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
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
        'appium:webviewConnectTimeout': 120_000,
        'appium:newCommandTimeout': 300,
        'appium:simulatorStartupTimeout': 600_000,
        'appium:wdaLaunchTimeout': 600_000,
        'appium:wdaConnectionTimeout': 600_000,
        'appium:showXcodeLog': true,
      });
    } catch (e) {
      const message = (e as Error).message;
      console.log(`iOS session attempt ${attempt} failed: ${message.slice(0, 500)}`);
      if (attempt === maxAttempts) throw e;
      // A simulator LaunchServices hiccup can leave the freshly installed
      // WebDriverAgent app unregistered ("Unknown application display
      // identifier"). Remove it so Appium reinstalls cleanly, then reboot.
      if (/WebDriverAgent|Unknown application display identifier|xcodebuild failed/i.test(message)) {
        try {
          execFileSync('xcrun', ['simctl', 'uninstall', sim.udid, 'com.facebook.WebDriverAgentRunner.xctrunner'], { stdio: 'ignore' });
        } catch { /* not installed */ }
      }
      // A corrupt boot needs a wipe; a Safari/remote-debugger miss only needs
      // a restart (erasing would put Safari back on its welcome screen).
      bootSimulator(sim.udid, /did not finish booting/i.test(message));
      // Give SpringBoard/LaunchServices a moment after the reboot.
      await new Promise((r) => setTimeout(r, 3000));
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
  //
  // The simulator renders WebKit at only a few frames per second, so run the
  // clips at half speed: acquisition gets more wall time while mediaTime (and
  // therefore the ground truth) stays in sync.
  //
  // The rAF fallback stamps a frame with the video time a display frame or two
  // after the pixels it grabbed, and at ~3 fps a moving clip loses lock between
  // frames. Motion overlay and sweep tracking are asserted by desktop and
  // Playwright's mobile WebKit; here the static scene is the accuracy gate.
  const still = await navigateAndRun(sessionId, `${BASE}/?mode=replay&scenario=static&useImu=0&norvfc=1&rate=0.5&runMs=8000`);
  checkReplay('static', still, { p95: 8, minFrames: 8, settle: 2, minErrors: 2 });

  const video = await navigateAndRun(sessionId, `${BASE}/?mode=replay&scenario=video&target=targets-video.bin&useImu=0&norvfc=1&rate=0.5&runMs=6000`);
  console.log(`video: ${video.frames.length} frames, ${JSON.stringify(video.stats)}`);
  checkVideo(video, { minFrames: 3, locked: 0.3 });

  // Canvas captureStream is optional in Mobile Safari; replay is the required
  // tracker check here. Exercise it when the runtime exposes it, but do not
  // fail the suite when Safari never starts the synthetic stream.
  try {
    const stream = await navigateAndRun(sessionId, `${BASE}/?mode=stream&scenario=static&useImu=0&norvfc=1`);
    console.log(`stream: ${stream.frames.length} frames, ${JSON.stringify(stream.stats)}`);
    if (stream.error) {
      console.log(`stream skipped: ${stream.error}`);
    } else {
      assert.ok(stream.frames.length > 3, `stream: too few frames (${stream.frames.length})`);
    }
  } catch (e) {
    const message = (e as Error).message;
    if (!/harness did not (load|finish)/.test(message)) throw e;
    console.log(`stream skipped: ${message}`);
  }

  console.log('iOS Safari suite passed');
} finally {
  if (sessionId) await deleteSession(sessionId);
  await closeServer(server);
}
