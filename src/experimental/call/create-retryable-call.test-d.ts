import { describe, expectTypeOf, it } from 'vitest';
import {
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
} from '../../internal/test-utils.js';
import type {
  EmbeddingModel,
  ImageModel,
  LanguageModel,
  RetryCallOptions,
} from '../../types.js';
import {
  createRetryableCall,
  type RetryCallAttempt,
} from './create-retryable-call.js';

describe('createRetryableCall types', () => {
  it('should default to LanguageModel with no type argument', () => {
    // Act
    const run = createRetryableCall({
      model: MockLanguageModel.from(),
      retries: [],
    });

    // Assert — the attempt is language-model-shaped.
    run((attempt) => {
      expectTypeOf(attempt).toEqualTypeOf<RetryCallAttempt<LanguageModel>>();
      expectTypeOf(attempt.model).toEqualTypeOf<LanguageModel>();
      return Promise.resolve('ok');
    });
  });

  it('should generalize to EmbeddingModel', () => {
    // Act
    const run = createRetryableCall<EmbeddingModel>({
      model: MockEmbeddingModel.from(),
      retries: [],
    });

    // Assert
    run((attempt) => {
      expectTypeOf(attempt.model).toEqualTypeOf<EmbeddingModel>();
      return Promise.resolve(0);
    });
  });

  it('should type the commit context with an opaque result', () => {
    // Act
    createRetryableCall({
      model: MockLanguageModel.from(),
      retries: [],
      onCommit: (context) => {
        // Assert — the driver never inspects the result, hence `unknown`.
        expectTypeOf(context.current.type).toEqualTypeOf<'commit'>();
        expectTypeOf(context.current.model).toEqualTypeOf<LanguageModel>();
        expectTypeOf(context.current.result).toEqualTypeOf<unknown>();
        expectTypeOf(context.current.options).toEqualTypeOf<
          RetryCallOptions<LanguageModel>
        >();
      },
    });
  });

  it('should type the failure context with the final error attempt', () => {
    // Act
    createRetryableCall({
      model: MockLanguageModel.from(),
      retries: [],
      onFailure: (context) => {
        // Assert
        expectTypeOf(context.current.type).toEqualTypeOf<'error'>();
        expectTypeOf(context.current.model).toEqualTypeOf<LanguageModel>();
        expectTypeOf(context.error).toEqualTypeOf<unknown>();
      },
    });
  });

  it('should generalize to ImageModel', () => {
    // Act
    const run = createRetryableCall<ImageModel>({
      model: MockImageModel.from(),
      retries: [],
    });

    // Assert
    run((attempt) => {
      expectTypeOf(attempt.model).toEqualTypeOf<ImageModel>();
      return Promise.resolve(0);
    });
  });
});
