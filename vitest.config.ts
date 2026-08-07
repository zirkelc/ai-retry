import { coverageConfigDefaults, defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    testTimeout: 120_000,
    setupFiles: ['./src/test-setup.ts'],
    typecheck: {
      enabled: true,
    },
    coverage: {
      include: ['src/**'],
      exclude: [
        ...coverageConfigDefaults.exclude,
        /**
         * Type tests. The defaults cover `*.test.ts` but not `*.test-d.ts`,
         * which has no runtime to measure.
         */
        '**/*.test-d.ts',
        /**
         * Type-only modules. They emit no runtime, so a coverage provider
         * reports them as 0% however thoroughly the types are exercised —
         * noise rather than a gap. The `.test-d.ts` files are what cover them.
         */
        'src/types.ts',
        'src/call/types.ts',
        'src/call/inputs.ts',
        /**
         * Re-export-only barrels. `src/index.ts` is not among them — it
         * carries a real declaration and is covered by `src/index.test.ts`.
         */
        'src/*-model/index.ts',
        'src/retryables/index.ts',
        'src/experimental/*/index.ts',
        /** Fixtures and setup, not subjects. */
        'src/internal/test-utils.ts',
        'src/test-setup.ts',
      ],
      /**
       * Gates against a real regression without being brittle: comfortably
       * below where the suite sits (97/93/96/98 overall, 99/96/98/100 for
       * `src/call`), so a genuine drop fails while ordinary churn does not.
       */
      thresholds: {
        statements: 90,
        branches: 85,
        functions: 90,
        lines: 90,
      },
    },
  },
});
