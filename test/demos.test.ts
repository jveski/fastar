import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { transformSync } from 'esbuild';

const ROOT = resolve('site');
function pages(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    return entry.isDirectory() ? pages(path) : entry.name.endsWith('.html') ? [path] : [];
  });
}

describe('canonical demos', () => {
  for (const path of pages(join(ROOT, 'demo'))) {
    it(`${path.slice(ROOT.length + 1)} has valid scripts and local asset links`, () => {
      const html = readFileSync(path, 'utf8');
      for (const [, script] of html.matchAll(/<script\s+type="module">([\s\S]*?)<\/script>/g)) {
        expect(() => transformSync(script, { loader: 'js', target: 'es2022' })).not.toThrow();
        for (const [, asset] of script.matchAll(/(?:from\s+|import\(|fetch\()\s*['"]([^'"]+)['"]/g)) {
          if (!asset.startsWith('.')) continue;
          const target = resolve(dirname(path), asset);
          if (target === join(ROOT, 'fastar.js')) continue; // Supplied by the build.
          expect(existsSync(target), `missing import/fetch ${asset}`).toBe(true);
        }
      }
      for (const [, asset] of html.matchAll(/(?:href|src)="([^"#?]+)"/g)) {
        if (/^(https?:|data:)/.test(asset)) continue;
        expect(existsSync(resolve(dirname(path), asset)), `missing link ${asset}`).toBe(true);
      }
    });
  }
});
