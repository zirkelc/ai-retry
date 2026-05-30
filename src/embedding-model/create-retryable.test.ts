import { describe, expect, it } from 'vitest';
import { RetryableEmbeddingModel } from '../internal/retryable-embedding-model.js';
import { MockEmbeddingModel } from '../internal/test-utils.js';
import { createRetryable } from './create-retryable.js';

describe('createRetryable', () => {
  it('should create a retryable model from an embedding model instance', () => {
    // Arrange
    const model = new MockEmbeddingModel();
    const fallbackModel = new MockEmbeddingModel();

    // Act
    const retryable = createRetryable({
      model,
      retries: [fallbackModel],
    });

    // Assert
    expect(retryable).toBeInstanceOf(RetryableEmbeddingModel);
    expect(retryable.provider).toBe(model.provider);
    expect(retryable.modelId).toBe(model.modelId);
    expect(retryable.specificationVersion).toBe('v3');
  });

  it('should resolve a gateway string base model to a gateway embedding model', () => {
    // Arrange + Act
    const retryable = createRetryable({
      model: 'openai/text-embedding-3-large',
      retries: ['openai/text-embedding-3-small'],
    });

    // Assert
    expect(retryable).toBeInstanceOf(RetryableEmbeddingModel);
    expect(retryable.provider).toBe('gateway');
    expect(retryable.modelId).toBe('openai/text-embedding-3-large');
  });
});
