import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { retryableEmbed } from './call/embed/index.js';
import { retryableEmbedMany } from './call/embed-many/index.js';
import { retryableGenerateImage } from './call/generate-image/index.js';
import { retryableGenerateText } from './call/generate-text/index.js';
import { retryableStreamText } from './call/stream-text/index.js';
import {
  createRetryable,
  isErrorAttempt,
  isResultAttempt,
  retryableGenerateText as rootGenerateText,
} from './index.js';
import { createRetryableModel } from './internal/create-retryable-model.js';
import {
  Embedding,
  MockEmbeddingModel,
  MockLanguageModel,
  mockResultText,
} from './internal/test-utils.js';

/**
 * The published surface, imported the way a consumer imports it.
 *
 * A barrel is exactly the kind of file unit tests never touch — every other
 * test reaches into the module it is about — so a broken re-export path stays
 * invisible until someone installs the package. These assertions are shallow on
 * purpose: what they check is that the names resolve at all.
 */

const packageJson = JSON.parse(
  readFileSync(fileURLToPath(new URL('../package.json', import.meta.url)), {
    encoding: 'utf8',
  }),
) as { exports: Record<string, string> };

describe('the per-entry-point barrels', () => {
  it('should export each call-level function from its own path', () => {
    // Assert
    expect(typeof retryableGenerateText).toBe('function');
    expect(typeof retryableStreamText).toBe('function');
    expect(typeof retryableEmbed).toBe('function');
    expect(typeof retryableEmbedMany).toBe('function');
    expect(typeof retryableGenerateImage).toBe('function');
  });

  it('should reach a real call through a published name', async () => {
    // Arrange
    const model = MockLanguageModel.from(mockResultText);

    // Act
    const result = await retryableGenerateText({ model, prompt: 'Hello!' });

    // Assert
    expect(result.text).toBe(mockResultText);
  });
});

describe('the root entry point', () => {
  it('should still re-export the call-level functions, deprecated', () => {
    // Assert — the same function object, not a second copy of it.
    expect(rootGenerateText).toBe(retryableGenerateText);
  });

  it('should export the attempt guards', () => {
    // Assert
    expect(typeof isErrorAttempt).toBe('function');
    expect(typeof isResultAttempt).toBe('function');
  });

  it('should alias createRetryable to the auto-detecting factory', () => {
    // Assert — deprecated, but still the shape published today.
    expect(createRetryable).toBe(createRetryableModel);
  });

  it('should build a retryable model through the published alias', async () => {
    // Arrange
    const model = MockEmbeddingModel.from([Embedding.vector(3)]);

    // Act
    const wrapped = createRetryable({ model, retries: [] });

    // Assert
    expect(wrapped.specificationVersion).toBe('v4');
  });
});

/**
 * Every `index.ts` under `src`, as the path tsdown will emit it to. The build
 * entry is `src/**\/index.ts`, so this is exactly the set of built entry points.
 */
function builtEntryPoints(): Array<string> {
  const root = fileURLToPath(new URL('.', import.meta.url));
  const walk = (dir: string, prefix: string): Array<string> =>
    readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
      entry.isDirectory()
        ? walk(`${dir}/${entry.name}`, `${prefix}${entry.name}/`)
        : entry.name === 'index.ts'
          ? [`./dist/${prefix}index.mjs`]
          : [],
    );
  return walk(root, '').sort();
}

describe('the export map', () => {
  /**
   * Hand-written in package.json, because the published paths deliberately do
   * not mirror the source layout. `publint` checks that every listed entry
   * resolves; nothing checks the other direction, so without this a new entry
   * point would build and quietly not be published.
   */
  it('should list every built entry point', () => {
    // Arrange
    const published = new Set(Object.values(packageJson.exports));

    // Act
    const unpublished = builtEntryPoints().filter(
      (path) => !published.has(path),
    );

    // Assert
    expect(unpublished).toEqual([]);
  });

  it('should not list an entry point that is not built', () => {
    // Arrange
    const built = new Set(builtEntryPoints());

    // Act
    const dangling = Object.entries(packageJson.exports)
      .filter(([key]) => key !== './package.json')
      .map(([, path]) => path)
      .filter((path) => !built.has(path));

    // Assert
    expect(dangling).toEqual([]);
  });
});
