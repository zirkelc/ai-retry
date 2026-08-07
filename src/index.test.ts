import { describe, expect, it } from 'vitest';
import {
  createRetryable,
  isEmbedManyResult,
  isEmbedResult,
  isErrorAttempt,
  isGenerateImageResult,
  isGenerateTextResult,
  isResultAttempt,
  isStreamTextResult,
  retryableEmbed,
  retryableEmbedMany,
  retryableGenerateImage,
  retryableGenerateText,
  retryableStreamText,
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

describe('the root entry point', () => {
  it('should export the five call-level functions', () => {
    // Assert
    expect(typeof retryableGenerateText).toBe('function');
    expect(typeof retryableStreamText).toBe('function');
    expect(typeof retryableEmbed).toBe('function');
    expect(typeof retryableEmbedMany).toBe('function');
    expect(typeof retryableGenerateImage).toBe('function');
  });

  it('should export the result guards', () => {
    // Assert
    expect(typeof isGenerateTextResult).toBe('function');
    expect(typeof isStreamTextResult).toBe('function');
    expect(typeof isEmbedResult).toBe('function');
    expect(typeof isEmbedManyResult).toBe('function');
    expect(typeof isGenerateImageResult).toBe('function');
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

  it('should reach a real call through the published names', async () => {
    // Arrange
    const model = MockLanguageModel.from(mockResultText);

    // Act
    const result = await retryableGenerateText({ model, prompt: 'Hello!' });

    // Assert
    expect(result.text).toBe(mockResultText);
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
