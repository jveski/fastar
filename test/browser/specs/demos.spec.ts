import { expect, test } from '@playwright/test';

const demos = ['index.html', 'video.html', 'paper.html', 'multi.html', 'timecode.html', 'performance.html'];

test('demo catalog links to every demo and its canonical source', async ({ page }) => {
  await page.goto('/site/demo/demos.html');
  for (const demo of demos) {
    const href = demo === 'index.html' ? './' : demo;
    await expect(page.locator(`main a[href="${href}"]`)).toBeVisible();
    await expect(page.locator(`main a[href="https://github.com/jveski/fastar/blob/main/site/demo/${demo}"]`)).toBeVisible();
  }
});

for (const demo of demos) {
  test(`${demo} loads and recovers from denied media access`, async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));
    // Denial is deterministic on every browser, including fake-camera projects.
    await page.addInitScript(() => {
      navigator.mediaDevices.getUserMedia = async () => {
        throw new DOMException('Demo test: permission denied', 'NotAllowedError');
      };
    });
    await page.goto(`/site/demo/${demo}`);
    await expect(page.locator('#start')).toBeEnabled();
    await page.locator('#start').click();
    await expect(page.locator('#error')).toContainText(/denied|not allowed/i);
    await expect(page.locator('body')).not.toHaveClass(/running/);
    await page.locator('#start').click();
    await expect(page.locator('body')).not.toHaveClass(/running/);
    expect(errors).toEqual([]);
  });
}

for (const demo of demos.filter((d) => d !== 'timecode.html')) {
  test(`${demo} starts, renders and stops`, async ({ page, context }, testInfo) => {
    test.skip(testInfo.project.name !== 'camera-chromium', 'Uses the fake camera');
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));
    await context.grantPermissions(['camera']);
    await page.goto(`/site/demo/${demo}`);
    await page.locator('#start').click();
    await expect(page.locator('#stop')).toBeEnabled();
    await expect(page.locator('#status')).not.toBeEmpty();
    await expect(page.locator('#status')).not.toContainText('NaN');
    await page.locator('#stop').click();
    await expect(page.locator('body')).not.toHaveClass(/running/);
    expect(errors).toEqual([]);
  });
}

test('timecode player loads without starting audio automatically', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));
  await page.goto('/site/demo/timecode/screen.html?key=demo-test');
  await expect(page.locator('#play')).toBeEnabled();
  await expect(page.locator('#stop')).toBeDisabled();
  await expect(page.locator('#receiver')).toHaveAttribute('href', '../timecode.html?key=demo-test');
  expect(errors).toEqual([]);
});

test('timecode receiver listens and stops with and without image tracking', async ({ page, context }, testInfo) => {
  test.skip(testInfo.project.name !== 'camera-chromium', 'Uses fake camera and microphone');
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));
  await context.grantPermissions(['camera', 'microphone']);
  await page.goto('/site/demo/timecode.html');
  for (const camera of [false, true]) {
    await page.locator('#camera').setChecked(camera);
    await page.locator('#start').click();
    await expect(page.locator('#stop')).toBeEnabled();
    await expect(page.locator('#status')).toContainText('Hz');
    await expect(page.locator('#status')).not.toContainText('NaN');
    await page.locator('#stop').click();
    await expect(page.locator('body')).not.toHaveClass(/running/);
  }
  expect(errors).toEqual([]);
});

test('performance demo accepts load and restarts with live display', async ({ page, context }, testInfo) => {
  test.skip(testInfo.project.name !== 'camera-chromium', 'Uses the fake camera');
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));
  await context.grantPermissions(['camera']);
  await page.goto('/site/demo/performance.html');
  await page.locator('#start').click();
  await expect(page.locator('#stop')).toBeEnabled();
  await page.locator('#load').fill('40');
  await expect(page.locator('#loadValue')).toHaveText('40 ms');
  await expect(page.locator('#status')).toContainText('Page busy');
  await page.locator('#stop').click();
  await page.locator('#sync').uncheck();
  await page.locator('#start').click();
  await expect(page.locator('#stop')).toBeEnabled();
  await expect(page.locator('#status')).toContainText('Display delay 0.0 ms');
  await page.locator('#stop').click();
  await expect(page.locator('body')).not.toHaveClass(/running/);
  expect(errors).toEqual([]);
});

test('timecode player generates audio and can stop and restart silently', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));
  await page.addInitScript(() => {
    // Exercise encoding/playback without sending ultrasonic audio to real speakers.
    AudioBufferSourceNode.prototype.connect = function () { return arguments[0]; };
  });
  await page.goto('/site/demo/timecode/screen.html');
  for (let i = 0; i < 2; i++) {
    await page.locator('#play').click();
    await expect(page.locator('#status')).toContainText('Playing');
    await page.locator('#stop').click();
    await expect(page.locator('#status')).toHaveText('Stopped');
    await expect(page.locator('#play')).toBeEnabled();
  }
  expect(errors).toEqual([]);
});
