import { describe, expect, it } from 'vitest';
import { RetryableLanguageModel } from '../internal/retryable-language-model.js';
import { MockLanguageModel } from '../internal/test-utils.js';
import { createRetryableModel } from './create-retryable-model.js';

describe('createRetryableModel', () => {
  it('should create a retryable model from a language model instance', () => {
    // Arrange
    const model = MockLanguageModel.from();
    const fallbackModel = MockLanguageModel.from();

    // Act
    const retryable = createRetryableModel({
      model,
      retries: [fallbackModel],
    });

    // Assert
    expect(retryable).toBeInstanceOf(RetryableLanguageModel);
    expect(retryable.provider).toBe(model.provider);
    expect(retryable.modelId).toBe(model.modelId);
    expect(retryable.specificationVersion).toBe('v4');
  });

  it('should resolve a gateway string base model to a gateway language model', () => {
    // Arrange + Act
    const retryable = createRetryableModel({
      model: 'openai/gpt-4.1',
      retries: ['anthropic/claude-sonnet-4'],
    });

    // Assert
    expect(retryable).toBeInstanceOf(RetryableLanguageModel);
    expect(retryable.provider).toBe('gateway');
    expect(retryable.modelId).toBe('openai/gpt-4.1');
  });
});
