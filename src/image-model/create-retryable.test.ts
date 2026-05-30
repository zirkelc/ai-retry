import { describe, expect, it } from 'vitest';
import { RetryableImageModel } from '../internal/retryable-image-model.js';
import { MockImageModel } from '../internal/test-utils.js';
import { createRetryable } from './create-retryable.js';

describe('createRetryable', () => {
  it('should create a retryable model from an image model instance', () => {
    // Arrange
    const model = new MockImageModel();
    const fallbackModel = new MockImageModel();

    // Act
    const retryable = createRetryable({
      model,
      retries: [fallbackModel],
    });

    // Assert
    expect(retryable).toBeInstanceOf(RetryableImageModel);
    expect(retryable.provider).toBe(model.provider);
    expect(retryable.modelId).toBe(model.modelId);
    expect(retryable.specificationVersion).toBe('v3');
  });

  it('should resolve a gateway string base model to a gateway image model', () => {
    // Arrange + Act
    const retryable = createRetryable({
      model: 'google/imagen-4.0-generate-001',
      retries: ['google/imagen-4.0-fast-generate-001'],
    });

    // Assert
    expect(retryable).toBeInstanceOf(RetryableImageModel);
    expect(retryable.provider).toBe('gateway');
    expect(retryable.modelId).toBe('google/imagen-4.0-generate-001');
  });
});
