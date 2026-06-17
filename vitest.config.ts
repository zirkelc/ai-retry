import { coverageConfigDefaults, defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    testTimeout: 120_000,
    setupFiles: ['./src/test-setup.ts'],
    typecheck: {
      enabled: true,
    },
  },
});
