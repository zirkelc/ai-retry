import { describe, expectTypeOf, it } from 'vitest';
import type {
  CallRetries,
  CallRetryAttempt,
  CallRetryContext,
} from '../call/types.js';
import type {
  EmbeddingModel,
  LanguageModel,
  ModelRetries,
  ModelRetryAttempt,
  ModelRetryContext,
  ResolvedModel,
  Retry,
} from '../types.js';
import { countModelAttempts } from './count-model-attempts.js';
import { findRetryModel } from './find-retry-model.js';
import { resolveBackoffDelay } from './resolve-backoff-delay.js';

/**
 * The retry internals are shared by both layers, whose contexts and attempts
 * are deliberately unrelated types. These pin that the sharing is real — each
 * helper takes either layer — and that it did not come at the cost of `MODEL`
 * inference, which a structurally loosened parameter would have lost.
 */

declare const model: LanguageModel;
declare const embeddingModel: EmbeddingModel;

declare const modelRetries: ModelRetries<LanguageModel>;
declare const callRetries: CallRetries<LanguageModel, never>;
declare const modelContext: ModelRetryContext<LanguageModel>;
declare const callContext: CallRetryContext<LanguageModel>;

declare const modelAttempts: Array<ModelRetryAttempt<LanguageModel>>;
declare const callAttempts: Array<CallRetryAttempt<LanguageModel>>;
declare const modelEmbeddingAttempts: Array<ModelRetryAttempt<EmbeddingModel>>;
declare const callEmbeddingAttempts: Array<CallRetryAttempt<EmbeddingModel>>;

describe('findRetryModel', () => {
  it('should accept the model layer and still infer MODEL', async () => {
    // Arrange, Act & Assert
    const found = await findRetryModel(modelRetries, modelContext);
    expectTypeOf(found).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>> | undefined
    >();
  });

  it('should accept the call layer and still infer MODEL', async () => {
    // Arrange, Act & Assert
    const found = await findRetryModel(callRetries, callContext);
    expectTypeOf(found).toEqualTypeOf<
      Retry<ResolvedModel<LanguageModel>, never> | undefined
    >();
  });

  it('should take either layer context against either layer retries', async () => {
    // Arrange, Act & Assert — the context is read structurally, so it is not
    // what keeps the two layers apart; the `retries` element types are.
    await findRetryModel(modelRetries, callContext);
    await findRetryModel(callRetries, modelContext);
  });

  it('should reject something that is not a retries list', () => {
    // @ts-expect-error a bare object is not a retryable, a Retry or a model
    findRetryModel([{ nope: true }], modelContext);
  });
});

describe('countModelAttempts', () => {
  it('should accept either layer attempts', () => {
    // Arrange, Act & Assert
    expectTypeOf(
      countModelAttempts(model, modelAttempts),
    ).toEqualTypeOf<number>();
    expectTypeOf(
      countModelAttempts(model, callAttempts),
    ).toEqualTypeOf<number>();
    expectTypeOf(
      countModelAttempts(embeddingModel, modelEmbeddingAttempts),
    ).toEqualTypeOf<number>();
    expectTypeOf(
      countModelAttempts(embeddingModel, callEmbeddingAttempts),
    ).toEqualTypeOf<number>();
  });

  it('should reject an attempt-shaped object that is neither layer', () => {
    // @ts-expect-error carries a model but is not an attempt of either layer
    countModelAttempts(model, [{ model }]);
  });
});

describe('resolveBackoffDelay', () => {
  it('should accept either layer attempts', () => {
    // Arrange
    const retry: Retry<LanguageModel, unknown> = { model, delay: 100 };

    // Act & Assert
    expectTypeOf(resolveBackoffDelay(retry, modelAttempts)).toEqualTypeOf<
      number | undefined
    >();
    expectTypeOf(resolveBackoffDelay(retry, callAttempts)).toEqualTypeOf<
      number | undefined
    >();
  });
});
