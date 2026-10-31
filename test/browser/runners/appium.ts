// Minimal W3C WebDriver client for the Appium jobs. Keeping this small avoids
// pulling webdriverio into the project just for two CI-only runners.
import { strict as assert } from 'node:assert';

export const APPIUM = process.env.APPIUM_URL ?? 'http://127.0.0.1:4723';

export async function wd(method: string, path: string, body?: unknown): Promise<any> {
  const res = await fetch(`${APPIUM}${path}`, {
    method,
    headers: { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`${method} ${path}: HTTP ${res.status} ${JSON.stringify(json).slice(0, 1200)}`);
  return json.value;
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
