// Real Mobile Safari on an iOS Simulator, driven through Appium/XCUITest.
// Simulators have no camera or motion hardware, so this checks replay and
// synthetic-stream paths in Safari's actual WebKit engine.
//
// Prerequisites (the workflow installs these): appium + the xcuitest driver,
// an Appium server on :4723, and Xcode with an iOS runtime.
import { execFileSync } from 'node:child_process';
import { strict as assert } from 'node:assert';
import { BASE, checkReplay, checkVideo, closeServer, startServer, waitForServer } from './shared';

const APPIUM = process.env.APPIUM_URL ?? 'http://127.0.0.1:4723';

async function wd(method: string, path: string, body?: unknown): Promise<any> {
  const res = await fetch(`${APPIUM}${path}`, {
    method,
    headers: { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`${method} ${path}: HTTP ${res.status} ${JSON.stringify(json).slice(0, 800)}`);
  return json.value;
}

async function waitAppium(): Promise<void> {
  for (let i = 0; i < 120; i++) {
    try {
      const res = await fetch(`${APPIUM}/status`);
      if (res.ok) return;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`Appium did not start at ${APPIUM}`);
}

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

  const caps = {
    capabilities: {
      alwaysMatch: {
        platformName: 'iOS',
        browserName: 'Safari',
        'appium:automationName': 'XCUITest',
        'appium:udid': udid,
        'appium:autoAcceptAlerts': true,
        'appium:newCommandTimeout': 300,
        'appium:wdaLaunchTimeout': 240_000,
        'appium:wdaConnectionTimeout': 240_000,
        'appium:showXcodeLog': true,
      },
      firstMatch: [{}],
    },
  };
  sessionId = (await wd('POST', '/session', caps))?.sessionId;
  assert.ok(sessionId, 'Appium did not return a session id');

  const sid = sessionId;
  const script = async (body: string): Promise<any> => wd('POST', `/session/${sid}/execute/sync`, { script: body, args: [] });

  async function loadHarness(params: Record<string, string>): Promise<any> {
    const q = new URLSearchParams(params);
    await wd('POST', `/session/${sid}/url`, { url: `${BASE}/?${q.toString()}` });
    let ready = false;
    for (let i = 0; i < 120 && !ready; i++) {
      try {
        ready = await script('return typeof window.__fastarTest !== "undefined" && !window.__fastarTest.done');
      } catch { /* Safari is still swapping execution contexts */ }
      if (!ready) await new Promise((r) => setTimeout(r, 500));
    }
    if (!ready) throw new Error('harness did not load in Safari');

    // Prefer a real element tap; fall back to a scripted click if the remote
    // debugger's element lookup is still settling after navigation.
    try {
      const el = await wd('POST', `/session/${sid}/element`, { using: 'css selector', value: '#start' });
      await wd('POST', `/session/${sid}/element/${el['element-6066-11e4-a52e-4f735466cecf']}/click`);
    } catch {
      await script('document.getElementById("start").click()');
    }
    for (let i = 0; i < 220; i++) {
      try {
        const r = await script('return window.__fastarTest');
        if (r?.done) return r;
      } catch { /* page is still loading its module */ }
      await new Promise((r2) => setTimeout(r2, 500));
    }
    throw new Error('harness did not finish in Safari');
  }

  // Simulator Safari cannot ask for motion permission, and there is no camera.
  const sweep = await loadHarness({ mode: 'replay', scenario: 'sweep', useImu: '0' });
  checkReplay('sweep', sweep, { p95: 6 });

  const video = await loadHarness({ mode: 'replay', scenario: 'video', target: 'targets-video.bin', useImu: '0' });
  checkVideo(video);

  const stream = await loadHarness({ mode: 'stream', scenario: 'static', useImu: '0' });
  if (stream.error) throw new Error(`stream: ${stream.error}`);
  assert.ok(stream.frames.length > 20, `stream: too few frames (${stream.frames.length})`);
  console.log(`stream: ${stream.frames.length} frames, ${stream.stats?.path}`);

  console.log('iOS Safari suite passed');
} finally {
  if (sessionId) await wd('DELETE', `/session/${sessionId}`).catch(() => {});
  await closeServer(server);
}
