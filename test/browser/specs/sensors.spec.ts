import { expect, test } from '@playwright/test';
import { expectNoErrors, runHarness, visibleRatio } from '../metrics';

function webkitLinux(testInfo: import('@playwright/test').TestInfo): boolean {
  const name = testInfo.project.name;
  return process.platform === 'linux' && (name.includes('webkit') || name.includes('safari'));
}

test.describe('motion sensors', () => {
  test('feeds synthetic devicemotion into the tracker', async ({ page }, testInfo) => {
    test.skip(webkitLinux(testInfo), 'WebKitGTK does not deliver canvas capture frames');
    const r = await runHarness(page, { mode: 'sensors', scenario: 'sweep' });
    test.skip(!r.diagnostics.deviceMotionEvent, 'DeviceMotionEvent is not available in this browser');
    expectNoErrors(r);
    expect(r.stats?.sensors).not.toBe('none');
    expect(r.stats?.imuSamples).toBeGreaterThan(20);
    expect(visibleRatio(r)).toBeGreaterThan(0.5);
  });

  test('runs vision-only when motion permission is denied', async ({ page }) => {
    const r = await runHarness(page, { mode: 'stream', scenario: 'static', useImu: '0', deny: '1' });
    test.skip(!r.diagnostics.needsMotionPermission, 'this browser does not gate motion access behind permission');
    expectNoErrors(r);
    expect(r.stats?.sensors).toBe('none');
    expect(r.frames.length).toBeGreaterThan(10);
  });
});
