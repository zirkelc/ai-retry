import { describe, expectTypeOf, it } from 'vitest';
import { findRetryModel } from './find-retry-model.js';
import { MockEmbeddingModel, MockLanguageModel } from './test-utils.js';
import type {
  EmbeddingModel,
  LanguageModel,
  ResolvableLanguageModel,
  ResolvedModel,
  Retries,
  Retry,
  Retryable,
  RetryContext,
} from './types.js';

describe('findRetryModel', () => {
  it('should accept LanguageModel retries', async () => {
    const model = new MockLanguageModel();
    const retries: Retries<LanguageModel> = [model];
    const context: RetryContext<LanguageModel> = {
      current: { type: 'error', error: new Error(), model },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should accept EmbeddingModel retries', async () => {
    const model = new MockEmbeddingModel();
    const retries: Retries<EmbeddingModel> = [model];
    const context: RetryContext<EmbeddingModel> = {
      current: { type: 'error', error: new Error(), model },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<EmbeddingModel>> | undefined
    >();
  });

  it('should accept string literal models in retries for LanguageModel', async () => {
    const model = new MockLanguageModel();
    const retries: Retries<LanguageModel> = [
      'openai/gpt-4o',
      'anthropic/claude-3.5-sonnet-20240620',
    ];
    const context: RetryContext<LanguageModel> = {
      current: { type: 'error', error: new Error(), model },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should accept Retryable functions', async () => {
    const model = new MockLanguageModel();
    const retryable: Retryable<LanguageModel> = () => ({
      model,
      maxAttempts: 1,
    });
    const retries: Retries<LanguageModel> = [retryable];
    const context: RetryContext<LanguageModel> = {
      current: { type: 'error', error: new Error(), model },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should accept Retryable functions with string models', async () => {
    const model = new MockLanguageModel();
    const retryable: Retryable<ResolvableLanguageModel> = () => ({
      model: 'openai/gpt-4o',
      maxAttempts: 1,
    });
    const retries: Retries<LanguageModel> = [retryable];
    const context: RetryContext<LanguageModel> = {
      current: { type: 'error', error: new Error(), model },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should accept Retry objects', async () => {
    const model = new MockLanguageModel();
    const retry: Retry<LanguageModel> = {
      model,
      maxAttempts: 3,
      delay: 1000,
    };
    const retries: Retries<LanguageModel> = [retry];
    const context: RetryContext<LanguageModel> = {
      current: { type: 'error', error: new Error(), model },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should accept Retry objects with string models', async () => {
    const model = new MockLanguageModel();
    const retry: Retry<ResolvableLanguageModel> = {
      model: 'anthropic/claude-3.5-sonnet-20240620',
      maxAttempts: 2,
    };
    const retries: Retries<LanguageModel> = [retry];
    const context: RetryContext<LanguageModel> = {
      current: { type: 'error', error: new Error(), model },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should accept mixed retry types', async () => {
    const model = new MockLanguageModel();
    const fallback = new MockLanguageModel();
    const retryable: Retryable<LanguageModel> = () => ({
      model: fallback,
      maxAttempts: 1,
    });
    const retry: Retry<ResolvableLanguageModel> = {
      model: 'openai/gpt-4o-mini',
      maxAttempts: 2,
    };
    const retries: Retries<LanguageModel> = [
      retryable,
      retry,
      fallback,
      'anthropic/claude-3.5-haiku',
    ];
    const context: RetryContext<LanguageModel> = {
      current: { type: 'error', error: new Error(), model },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should resolve string models to LanguageModel', async () => {
    const model = new MockLanguageModel();
    const retries: Retries<LanguageModel> = ['openai/gpt-4o'];
    const context: RetryContext<LanguageModel> = {
      current: { type: 'error', error: new Error(), model },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    if (result) {
      expectTypeOf(result.model).toEqualTypeOf<LanguageModel>();
    }
  });
});
