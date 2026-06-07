import { describe, expectTypeOf, it } from 'vitest';
import {
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
} from '../../internal/test-utils.js';
import type { EmbeddingModel, ImageModel, LanguageModel } from '../../types.js';
import {
  createRetryableCall,
  type RetryCallAttempt,
} from './create-retryable-call.js';

describe('createRetryableCall types', () => {
  it('should default to LanguageModel with no type argument', () => {
    // Act
    const run = createRetryableCall({
      model: new MockLanguageModel(),
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
      model: new MockEmbeddingModel(),
      retries: [],
    });

    // Assert
    run((attempt) => {
      expectTypeOf(attempt.model).toEqualTypeOf<EmbeddingModel>();
      return Promise.resolve(0);
    });
  });

  it('should generalize to ImageModel', () => {
    // Act
    const run = createRetryableCall<ImageModel>({
      model: new MockImageModel(),
      retries: [],
    });

    // Assert
    run((attempt) => {
      expectTypeOf(attempt.model).toEqualTypeOf<ImageModel>();
      return Promise.resolve(0);
    });
  });
});
