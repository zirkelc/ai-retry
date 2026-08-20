import { describe, expectTypeOf, it } from 'vitest';
import {
  createRetryableModel,
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
} from '../internal/test-utils.js';
import type { ModelRetryable } from '../types.js';
import { requestNotRetryable } from './request-not-retryable.js';

describe('requestNotRetryable', () => {
  it('should accept language model instance', () => {
    const model = MockLanguageModel.from();
    const retryable = requestNotRetryable(model);

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<ModelRetryable<MockLanguageModel>>();
  });

  it('should accept embedding model instance', () => {
    const model = MockEmbeddingModel.from();
    const retryable = requestNotRetryable(model);

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<ModelRetryable<MockEmbeddingModel>>();
  });

  it('should accept language model with options', () => {
    const model = MockLanguageModel.from();
    const retryable = requestNotRetryable(model, { maxAttempts: 3 });

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<ModelRetryable<MockLanguageModel>>();
  });

  it('should accept embedding model with options', () => {
    const model = MockEmbeddingModel.from();
    const retryable = requestNotRetryable(model, { maxAttempts: 3 });

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<ModelRetryable<MockEmbeddingModel>>();
  });

  it('should accept model string', () => {
    const retryable = requestNotRetryable('openai/gpt-4.1');

    const retryableModel = createRetryableModel({
      model: 'openai/gpt-4.1',
      retries: [retryable],
    });

    // expectTypeOf(retryable).toEqualTypeOf<ModelRetryable<LanguageModel>>();
  });

  it('should accept image model instance', () => {
    const model = MockImageModel.from();
    const retryable = requestNotRetryable(model);

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<ModelRetryable<MockImageModel>>();
  });

  it('should accept image model with options', () => {
    const model = MockImageModel.from();
    const retryable = requestNotRetryable(model, { maxAttempts: 3 });

    const retryableModel = createRetryableModel({
      model,
      retries: [retryable],
    });

    expectTypeOf(retryable).toEqualTypeOf<ModelRetryable<MockImageModel>>();
  });
});
