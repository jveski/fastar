// Minimal W3C WebDriver client for the Appium jobs. Keeping this small avoids
// pulling webdriverio into the project just for two CI-only runners.
//
// Node's fetch has a fixed 5-minute headers timeout, and WDA builds routinely
// exceed that. Use node:http so session creation can take 20 minutes.
import { strict as assert } from 'node:assert';
import { request } from 'node:http';

export const APPIUM = process.env.APPIUM_URL ?? 'http://127.0.0.1:4723';

export function wd(method: string, path: string, body?: unknown): Promise<any> {
  const url = new URL(`${APPIUM}${path}`);
  const payload = body === undefined ? undefined : JSON.stringify(body);
  return new Promise((resolve, reject) => {
    const req = request({
      hostname: url.hostname,
      port: url.port,
      path: `${url.pathname}${url.search}`,
      method,
      timeout: 20 * 60_000,
      headers: payload
        ? { 'content-type': 'application/json', 'content-length': Buffer.byteLength(payload) }
        : {},
    }, (res) => {
      let data = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        let json: any = {};
        try { json = JSON.parse(data); } catch { /* non-JSON error page */ }
        const code = res.statusCode ?? 0;
        if (code < 200 || code >= 300) {
          reject(new Error(`${method} ${path}: HTTP ${code} ${JSON.stringify(json).slice(0, 1200)}`));
        } else {
          resolve(json.value);
        }
      });
    });
    req.on('timeout', () => req.destroy(new Error(`${method} ${path}: timed out`)));
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

export async function waitAppium(): Promise<void> {
  for (let i = 0; i < 120; i++) {
    try {
      const res = await fetch(`${APPIUM}/status`);
      if (res.ok) return;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`Appium did not start at ${APPIUM}`);
}

export async function createSession(capabilities: Record<string, unknown>): Promise<string> {
  const value = await wd('POST', '/session', {
    capabilities: { alwaysMatch: capabilities, firstMatch: [{}] },
  });
  assert.ok(value?.sessionId, 'Appium did not return a session id');
  return value.sessionId as string;
}

export async function deleteSession(sid: string): Promise<void> {
  await wd('DELETE', `/session/${sid}`).catch(() => {});
}

/**
 * Navigate to a harness URL, tap start, and poll until the harness result is
 * done. Safari swaps execution contexts during navigation, so every script
 * call is retried.
 */
export async function navigateAndRun(sid: string, url: string): Promise<any> {
  const script = async (body: string): Promise<any> =>
    wd('POST', `/session/${sid}/execute/sync`, { script: body, args: [] });

  await wd('POST', `/session/${sid}/url`, { url });
  let ready = false;
  for (let i = 0; i < 120 && !ready; i++) {
    try {
      ready = await script('return typeof window.__fastarTest !== "undefined" && !window.__fastarTest.done');
    } catch { /* execution context is still swapping */ }
    if (!ready) await new Promise((r) => setTimeout(r, 500));
  }
  if (!ready) throw new Error(`harness did not load: ${url}`);

  try {
    const el = await wd('POST', `/session/${sid}/element`, { using: 'css selector', value: '#start' });
    await wd('POST', `/session/${sid}/element/${el['element-6066-11e4-a52e-4f735466cecf']}/click`);
  } catch {
    try { await script('document.getElementById("start").click()'); } catch { /* poll below */ }
  }

  for (let i = 0; i < 240; i++) {
    try {
      const r = await script('return window.__fastarTest');
      if (r?.done) return r;
    } catch { /* page is still loading its module */ }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`harness did not finish: ${url}`);
}
