import { defineConfig } from 'tsdown';

export default defineConfig({
  /**
   * Run arethetypeswrong after bundling.
   * Requires @arethetypeswrong/core to be installed.
   */
  attw: {
    profile: 'esmOnly',
  },
  publint: true,
  exports: true,
  entry: 'src/**/index.ts',
  format: ['esm'],
});
