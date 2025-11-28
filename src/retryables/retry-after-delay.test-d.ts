import { describe, expectTypeOf, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import { MockEmbeddingModel, MockLanguageModel } from '../test-utils.js';
import type { EmbeddingModel, LanguageModel, Retryable } from '../types.js';
import { retryAfterDelay } from './retry-after-delay.js';

describe('retryAfterDelay types', () => {
  it('should accept language model with options', () => {
    const model = new MockLanguageModel();
    const retryable = retryAfterDelay<MockLanguageModel>({ maxAttempts: 3 });

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockLanguageModel>>();
  });

  it('should accept embedding model with options', () => {
    const model = new MockEmbeddingModel();
    const retryable = retryAfterDelay<MockEmbeddingModel>({ maxAttempts: 3 });

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockEmbeddingModel>>();
  });

  it('should accept model string', () => {
    const retryable = retryAfterDelay<MockLanguageModel>({});

    const retryableModel = createRetryable({
      model: 'openai/gpt-4.1',
      retries: [retryable],
    });

    // expectTypeOf(retryable).toEqualTypeOf<Retryable<LanguageModel>>();
  });
});
