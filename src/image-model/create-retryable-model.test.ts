import { describe, expect, it } from 'vitest';
import { RetryableImageModel } from '../internal/retryable-image-model.js';
import { MockImageModel } from '../internal/test-utils.js';
import { createRetryableModel } from './create-retryable-model.js';

describe('createRetryableModel', () => {
  it('should create a retryable model from an image model instance', () => {
    // Arrange
    const model = MockImageModel.from();
    const fallbackModel = MockImageModel.from();

    // Act
    const retryable = createRetryableModel({
      model,
      retries: [fallbackModel],
    });

    // Assert
    expect(retryable).toBeInstanceOf(RetryableImageModel);
    expect(retryable.provider).toBe(model.provider);
    expect(retryable.modelId).toBe(model.modelId);
    expect(retryable.specificationVersion).toBe('v4');
  });

  it('should resolve a gateway string base model to a gateway image model', () => {
    // Arrange + Act
    const retryable = createRetryableModel({
      model: 'google/imagen-4.0-generate-001',
      retries: ['google/imagen-4.0-fast-generate-001'],
    });

    // Assert
    expect(retryable).toBeInstanceOf(RetryableImageModel);
    expect(retryable.provider).toBe('gateway');
    expect(retryable.modelId).toBe('google/imagen-4.0-generate-001');
  });
});
