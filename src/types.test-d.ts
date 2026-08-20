import { describe, expectTypeOf, it } from 'vitest';
import type {
  CallOptions,
  EmbeddingModel,
  FailureContext,
  FinishReason,
  ImageModel,
  LanguageModel,
  ModelCallOptions,
  ModelFailureContext,
  ModelFinishReason,
  ModelResult,
  ModelRetries,
  ModelRetryable,
  ModelRetryAttempt,
  ModelRetryCallOptions,
  ModelRetryContext,
  ModelRetryErrorAttempt,
  ModelRetryResultAttempt,
  ModelSuccessAttempt,
  ModelSuccessContext,
  Result,
  Retries,
  Retryable,
  RetryAttempt,
  RetryCallOptions,
  RetryContext,
  RetryErrorAttempt,
  RetryResultAttempt,
  SuccessAttempt,
  SuccessContext,
} from './types.js';

/**
 * The pre-`Model` names are kept as aliases for the whole deprecation window.
 * An alias that drifted from what it aliases would be a silent break for anyone
 * still on the old name, so each is pinned to be the *same* type rather than a
 * compatible one.
 */
describe('deprecated aliases', () => {
  it('should be identical to their Model-prefixed replacements', () => {
    // Arrange, Act & Assert
    expectTypeOf<FinishReason>().toEqualTypeOf<ModelFinishReason>();
    expectTypeOf<RetryCallOptions<LanguageModel>>().toEqualTypeOf<
      ModelRetryCallOptions<LanguageModel>
    >();
    expectTypeOf<CallOptions<EmbeddingModel>>().toEqualTypeOf<
      ModelCallOptions<EmbeddingModel>
    >();
    expectTypeOf<Result<ImageModel>>().toEqualTypeOf<ModelResult<ImageModel>>();
    expectTypeOf<RetryErrorAttempt<LanguageModel>>().toEqualTypeOf<
      ModelRetryErrorAttempt<LanguageModel>
    >();
    expectTypeOf<RetryResultAttempt>().toEqualTypeOf<ModelRetryResultAttempt>();
    expectTypeOf<RetryAttempt<LanguageModel>>().toEqualTypeOf<
      ModelRetryAttempt<LanguageModel>
    >();
    expectTypeOf<RetryContext<LanguageModel>>().toEqualTypeOf<
      ModelRetryContext<LanguageModel>
    >();
    expectTypeOf<SuccessAttempt<LanguageModel>>().toEqualTypeOf<
      ModelSuccessAttempt<LanguageModel>
    >();
    expectTypeOf<SuccessContext<LanguageModel>>().toEqualTypeOf<
      ModelSuccessContext<LanguageModel>
    >();
    expectTypeOf<FailureContext<LanguageModel>>().toEqualTypeOf<
      ModelFailureContext<LanguageModel>
    >();
    expectTypeOf<Retryable<LanguageModel>>().toEqualTypeOf<
      ModelRetryable<LanguageModel>
    >();
    expectTypeOf<Retries<LanguageModel>>().toEqualTypeOf<
      ModelRetries<LanguageModel>
    >();
  });

  it('should keep the defaulted generics of the two that have them', () => {
    // Arrange, Act & Assert — `Retryable`/`Retries` default `INPUT` to the
    // provider-level overrides, which is what keeps an existing annotation
    // like `Retryable<LanguageModel>` accepting `options`.
    expectTypeOf<Retryable<LanguageModel>>().toEqualTypeOf<
      ModelRetryable<LanguageModel, ModelRetryCallOptions<LanguageModel>>
    >();
    expectTypeOf<Retries<EmbeddingModel>>().toEqualTypeOf<
      ModelRetries<EmbeddingModel, ModelRetryCallOptions<EmbeddingModel>>
    >();
  });
});
