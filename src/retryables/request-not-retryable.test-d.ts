import { describe, expectTypeOf, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import { MockEmbeddingModel, MockLanguageModel } from '../test-utils.js';
import type { Retryable } from '../types.js';
import { requestNotRetryable } from './request-not-retryable.js';

describe('requestNotRetryable', () => {
  it('should accept language model instance', () => {
    const model = new MockLanguageModel();
    const retryable = requestNotRetryable(model);

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockLanguageModel>>();
  });

  it('should accept embedding model instance', () => {
    const model = new MockEmbeddingModel();
    const retryable = requestNotRetryable(model);

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockEmbeddingModel>>();
  });

  it('should accept language model with options', () => {
    const model = new MockLanguageModel();
    const retryable = requestNotRetryable(model, { maxAttempts: 3 });

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockLanguageModel>>();
  });

  it('should accept embedding model with options', () => {
    const model = new MockEmbeddingModel();
    const retryable = requestNotRetryable(model, { maxAttempts: 3 });

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockEmbeddingModel>>();
  });

  it('should accept model string', () => {
    const retryable = requestNotRetryable('openai/gpt-4.1');

    const retryableModel = createRetryable({
      model: 'openai/gpt-4.1',
      retries: [retryable],
    });

    // expectTypeOf(retryable).toEqualTypeOf<Retryable<LanguageModel>>();
  });
});
