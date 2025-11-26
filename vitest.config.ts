import { coverageConfigDefaults, defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    testTimeout: 120_000,
    typecheck: {
      enabled: true,
    },
  },
});
