import { describe, expectTypeOf, it } from 'vitest';
import {
  createRetryableModel,
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
} from '../internal/test-utils.js';
import type { EmbeddingModel, LanguageModel, Retryable } from '../types.js';
import { retryAfterDelay } from './retry-after-delay.js';

describe('retryAfterDelay types', () => {
  it('should accept language model with options', () => {
    const model = MockLanguageModel.from();
    const retryable = retryAfterDelay<MockLanguageModel>({ maxAttempts: 3 });

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockLanguageModel>>();
  });

  it('should accept embedding model with options', () => {
    const model = MockEmbeddingModel.from();
    const retryable = retryAfterDelay<MockEmbeddingModel>({ maxAttempts: 3 });

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockEmbeddingModel>>();
  });

  it('should accept model string', () => {
    const retryable = retryAfterDelay<MockLanguageModel>({});

    const retryableModel = createRetryableModel({
      model: 'openai/gpt-4.1',
      retries: [retryable],
    });

    // expectTypeOf(retryable).toEqualTypeOf<Retryable<LanguageModel>>();
  });

  it('should accept image model with options', () => {
    const model = MockImageModel.from();
    const retryable = retryAfterDelay<MockImageModel>({ maxAttempts: 3 });

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockImageModel>>();
  });
});
