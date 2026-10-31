import { expect, test } from '@playwright/test';
import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { expectNoErrors, runHarness, visibleRatio } from '../metrics';

const GOLDEN = join(process.cwd(), 'test/browser/fixtures/golden');
const sessions = existsSync(GOLDEN)
  ? readdirSync(GOLDEN).filter((f) => f.endsWith('.session.json')).map((f) => f.slice(0, -'.session.json'.length))
  : [];

test.describe('real-device golden sessions', () => {
  for (const name of sessions) {
    test(`replays ${name}`, async ({ page }) => {
      test.setTimeout(120_000);
      const r = await runHarness(page, {
        mode: 'replay', scenario: `golden/${name}`, truth: '0', target: `golden/${name}.bin`,
      });
      expectNoErrors(r);
      expect(r.frames.length).toBeGreaterThan(20);
      expect(visibleRatio(r)).toBeGreaterThan(0.5);
      expect(r.stats?.fps).toBeGreaterThan(5);
    });
  }

  test('has at least one real-device session', async () => {
    test.skip(true, 'Record a session on a physical phone; see fixtures/golden/README.md');
    expect(sessions.length).toBeGreaterThan(0);
  });
});
