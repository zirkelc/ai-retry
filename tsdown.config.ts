import { defineConfig } from 'tsdown';

export default defineConfig({
  /**
   * Run arethetypeswrong after bundling.
   * Requires @arethetypeswrong/core to be installed.
   */
  attw: {
    profile: 'esm-only',
  },
  publint: true,
  /**
   * The export map is written by hand in package.json, because it deliberately
   * does not mirror the source layout: `src/` splits at the top into the two
   * retry layers (`model/`, `call/`), and neither segment appears in the
   * published paths. Generating the map would publish `./call/generate-text`
   * and rename the existing `./language-model` to `./model/language-model`.
   *
   * `publint` and `attw` still check every listed entry resolves; that a built
   * entry point is actually listed is checked by `src/index.test.ts`.
   */
  exports: false,
  entry: 'src/**/index.ts',
  format: ['esm'],
});
