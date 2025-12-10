import { assertType, describe, expect, it } from 'vitest';
import { createRetryable } from './create-retryable-model.js';
import { RetryableEmbeddingModel } from './retryable-embedding-model.js';
import { RetryableLanguageModel } from './retryable-language-model.js';
import { MockEmbeddingModel, MockLanguageModel } from './test-utils.js';
import type { EmbeddingModel, LanguageModel } from './types.js';

describe('createRetryable', () => {
  describe('language model', () => {
    it('should create retryable model from language model instance', () => {
      const model = new MockLanguageModel();
      const fallbackModel = new MockLanguageModel();

      const retryable = createRetryable({
        model,
        retries: [fallbackModel],
      });

      expect(retryable).toBeDefined();
      expect(retryable).toBeInstanceOf(RetryableLanguageModel);
      expect(retryable.provider).toBe(model.provider);
      expect(retryable.modelId).toBe(model.modelId);
      expect(retryable.specificationVersion).toBe('v2');
    });

    it('should create retryable model from gateway string', () => {
      const fallbackModel = new MockLanguageModel();

      const retryable = createRetryable({
        model: 'openai/gpt-4.1',
        retries: [fallbackModel],
      });

      expect(retryable).toBeDefined();
      expect(retryable).toBeInstanceOf(RetryableLanguageModel);
      expect(retryable.provider).toBe('gateway');
      expect(retryable.modelId).toBe('openai/gpt-4.1');
      expect(retryable.specificationVersion).toBe('v2');
    });
  });

  describe('embedding model', () => {
    it('should create retryable model from embedding model instance', () => {
      const model = new MockEmbeddingModel();
      const fallbackModel = new MockEmbeddingModel();

      const retryable = createRetryable({
        model,
        retries: [fallbackModel],
      });

      expect(retryable).toBeDefined();
      expect(retryable).toBeInstanceOf(RetryableEmbeddingModel);
      expect(retryable.provider).toBe(model.provider);
      expect(retryable.modelId).toBe(model.modelId);
      expect(retryable.specificationVersion).toBe('v2');
    });
  });
});

describe('createRetryable type tests', () => {
  it('should return LanguageModel type for language model instance', () => {
    const model = new MockLanguageModel();
    const fallbackModel = new MockLanguageModel();

    const retryable = createRetryable({
      model,
      retries: [fallbackModel],
    });

    assertType<LanguageModel>(retryable);
  });

  it('should return LanguageModel type for gateway string', () => {
    const fallbackModel = new MockLanguageModel();

    const retryable = createRetryable({
      model: 'openai:gpt-4',
      retries: [fallbackModel],
    });

    assertType<LanguageModel>(retryable);
  });

  it('should return EmbeddingModel type for embedding model instance', () => {
    const model = new MockEmbeddingModel();
    const fallbackModel = new MockEmbeddingModel();

    const retryable = createRetryable({
      model,
      retries: [fallbackModel],
    });

    assertType<EmbeddingModel>(retryable);
  });
});
