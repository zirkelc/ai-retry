import { describe, expectTypeOf, it } from 'vitest';
import {
  createRetryableModel,
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
} from '../internal/test-utils.js';
import type { EmbeddingModel, LanguageModel, Retryable } from '../types.js';
import { serviceUnavailable } from './service-unavailable.js';

describe('serviceUnavailable', () => {
  it('should accept language model instance', () => {
    const model = new MockLanguageModel();
    const retryable = serviceUnavailable(model);

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockLanguageModel>>();
  });

  it('should accept embedding model instance', () => {
    const model = new MockEmbeddingModel();
    const retryable = serviceUnavailable(model);

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockEmbeddingModel>>();
  });

  it('should accept language model with options', () => {
    const model = new MockLanguageModel();
    const retryable = serviceUnavailable(model, { maxAttempts: 3 });

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockLanguageModel>>();
  });

  it('should accept embedding model with options', () => {
    const model = new MockEmbeddingModel();
    const retryable = serviceUnavailable(model, { maxAttempts: 3 });

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockEmbeddingModel>>();
  });

  it('should accept model string', () => {
    const retryable = serviceUnavailable('openai/gpt-4.1');

    const retryableModel = createRetryableModel({
      model: 'openai/gpt-4.1',
      retries: [retryable],
    });

    // expectTypeOf(retryable).toEqualTypeOf<Retryable<LanguageModel>>();
  });

  it('should accept image model instance', () => {
    const model = new MockImageModel();
    const retryable = serviceUnavailable(model);

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockImageModel>>();
  });

  it('should accept image model with options', () => {
    const model = new MockImageModel();
    const retryable = serviceUnavailable(model, { maxAttempts: 3 });

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<Retryable<MockImageModel>>();
  });
});
