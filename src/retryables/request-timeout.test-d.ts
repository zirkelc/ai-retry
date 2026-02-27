import { describe, expectTypeOf, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import {
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
} from '../test-utils.js';
import type { Retryable } from '../types.js';
import { requestTimeout } from './request-timeout.js';

describe('requestTimeout', () => {
  it('should accept language model instance', () => {
    const model = new MockLanguageModel();
    const retryable = requestTimeout(model);

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockLanguageModel>>();
  });

  it('should accept embedding model instance', () => {
    const model = new MockEmbeddingModel();
    const retryable = requestTimeout(model);

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockEmbeddingModel>>();
  });

  it('should accept language model with options', () => {
    const model = new MockLanguageModel();
    const retryable = requestTimeout(model, { maxAttempts: 3 });

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockLanguageModel>>();
  });

  it('should accept embedding model with options', () => {
    const model = new MockEmbeddingModel();
    const retryable = requestTimeout(model, { maxAttempts: 3 });

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockEmbeddingModel>>();
  });

  it('should accept model string', () => {
    const retryable = requestTimeout('openai/gpt-4.1');

    const retryableModel = createRetryable({
      model: 'openai/gpt-4.1',
      retries: [retryable],
    });

    // expectTypeOf(retryable).toEqualTypeOf<Retryable<LanguageModel>>();
  });

  it('should accept image model instance', () => {
    const model = new MockImageModel();
    const retryable = requestTimeout(model);

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockImageModel>>();
  });

  it('should accept image model with options', () => {
    const model = new MockImageModel();
    const retryable = requestTimeout(model, { maxAttempts: 3 });

    const retryableModel = createRetryable({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockImageModel>>();
  });
});
