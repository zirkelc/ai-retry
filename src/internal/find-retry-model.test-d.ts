import { describe, expectTypeOf, it } from 'vitest';
import { findRetryModel } from './find-retry-model.js';
import { MockEmbeddingModel, MockLanguageModel } from './test-utils.js';
import type {
  EmbeddingModel,
  EmbeddingModelCallOptions,
  LanguageModel,
  LanguageModelCallOptions,
  ResolvableLanguageModel,
  ResolvedModel,
  ModelRetries,
  Retry,
  ModelRetryable,
  ModelRetryContext,
} from '../types.js';

const languageModelOptions: LanguageModelCallOptions = {
  prompt: [
    { role: 'user', content: [{ type: 'text', text: 'Hello, world!' }] },
  ],
};
const embeddingModelOptions: EmbeddingModelCallOptions = {
  values: ['Hello, world!'],
};

describe('findRetryModel', () => {
  it('should accept LanguageModel retries', async () => {
    const model = MockLanguageModel.from();
    const retries: ModelRetries<LanguageModel> = [model];
    const context: ModelRetryContext<LanguageModel> = {
      current: {
        type: 'error',
        error: new Error(),
        model,
        options: languageModelOptions,
      },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should accept EmbeddingModel retries', async () => {
    const model = MockEmbeddingModel.from();
    const retries: ModelRetries<EmbeddingModel> = [model];
    const context: ModelRetryContext<EmbeddingModel> = {
      current: {
        type: 'error',
        error: new Error(),
        model,
        options: embeddingModelOptions,
      },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<EmbeddingModel>> | undefined
    >();
  });

  it('should accept string literal models in retries for LanguageModel', async () => {
    const model = MockLanguageModel.from();
    const retries: ModelRetries<LanguageModel> = [
      'openai/gpt-4o',
      'anthropic/claude-sonnet-4',
    ];
    const context: ModelRetryContext<LanguageModel> = {
      current: {
        type: 'error',
        error: new Error(),
        model,
        options: languageModelOptions,
      },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should accept ModelRetryable functions', async () => {
    const model = MockLanguageModel.from();
    const retryable: ModelRetryable<LanguageModel> = () => ({
      model,
      maxAttempts: 1,
    });
    const retries: ModelRetries<LanguageModel> = [retryable];
    const context: ModelRetryContext<LanguageModel> = {
      current: {
        type: 'error',
        error: new Error(),
        model,
        options: languageModelOptions,
      },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should accept ModelRetryable functions with string models', async () => {
    const model = MockLanguageModel.from();
    const retryable: ModelRetryable<ResolvableLanguageModel> = () => ({
      model: 'openai/gpt-4o',
      maxAttempts: 1,
    });
    const retries: ModelRetries<LanguageModel> = [retryable];
    const context: ModelRetryContext<LanguageModel> = {
      current: {
        type: 'error',
        error: new Error(),
        model,
        options: languageModelOptions,
      },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should accept Retry objects', async () => {
    const model = MockLanguageModel.from();
    const retry: Retry<LanguageModel> = {
      model,
      maxAttempts: 3,
      delay: 1000,
    };
    const retries: ModelRetries<LanguageModel> = [retry];
    const context: ModelRetryContext<LanguageModel> = {
      current: {
        type: 'error',
        error: new Error(),
        model,
        options: languageModelOptions,
      },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should accept Retry objects with string models', async () => {
    const model = MockLanguageModel.from();
    const retry: Retry<ResolvableLanguageModel> = {
      model: 'anthropic/claude-sonnet-4',
      maxAttempts: 2,
    };
    const retries: ModelRetries<LanguageModel> = [retry];
    const context: ModelRetryContext<LanguageModel> = {
      current: {
        type: 'error',
        error: new Error(),
        model,
        options: languageModelOptions,
      },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should accept mixed retry types', async () => {
    const model = MockLanguageModel.from();
    const fallback = MockLanguageModel.from();
    const retryable: ModelRetryable<LanguageModel> = () => ({
      model: fallback,
      maxAttempts: 1,
    });
    const retry: Retry<ResolvableLanguageModel> = {
      model: 'openai/gpt-4o-mini',
      maxAttempts: 2,
    };
    const retries: ModelRetries<LanguageModel> = [
      retryable,
      retry,
      fallback,
      'anthropic/claude-haiku-4.5',
    ];
    const context: ModelRetryContext<LanguageModel> = {
      current: {
        type: 'error',
        error: new Error(),
        model,
        options: languageModelOptions,
      },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    expectTypeOf(result).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should resolve string models to LanguageModel', async () => {
    const model = MockLanguageModel.from();
    const retries: ModelRetries<LanguageModel> = ['openai/gpt-4o'];
    const context: ModelRetryContext<LanguageModel> = {
      current: {
        type: 'error',
        error: new Error(),
        model,
        options: languageModelOptions,
      },
      attempts: [],
    };

    const result = await findRetryModel(retries, context);

    if (result) {
      expectTypeOf(result.model).toEqualTypeOf<LanguageModel>();
    }
  });
});
