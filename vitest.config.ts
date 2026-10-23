import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [
    {
      // The real worker source is inlined by build.mjs; tests drive the worker cores directly.
      name: 'worker-source-stub',
      resolveId: (id) => (id === 'virtual:worker-source' ? '\0worker-source' : null),
      load: (id) => (id === '\0worker-source' ? 'export default "";' : null),
    },
  ],
  test: {
    include: ['test/**/*.test.ts'],
    testTimeout: 60000,
    pool: 'threads',
  },
});
