import { expect, test } from '@playwright/test';
import { expectNoErrors, runHarness, visibleRatio } from '../metrics';

test.describe('session recorder round trip', () => {
  test.beforeEach(async ({}, testInfo) => {
    test.skip(testInfo.project.name !== 'chromium-desktop', 'MediaRecorder round trip runs on Chromium');
  });

  test('records a live stream and replays the produced session', async ({ page }) => {
    const r = await runHarness(page, { mode: 'record', scenario: 'static', useImu: '0' });
    expectNoErrors(r);
    expect(r.frames.length).toBeGreaterThan(20);
    expect(r.replay).not.toBeNull();
    expect(r.replay?.error).toBeNull();
    expect(r.replay?.frames.length).toBeGreaterThan(15);
    expect(r.replay?.stats?.fps).toBeGreaterThan(5);
    expect(visibleRatio({ ...r, frames: r.replay!.frames })).toBeGreaterThan(0.5);
  });
});
