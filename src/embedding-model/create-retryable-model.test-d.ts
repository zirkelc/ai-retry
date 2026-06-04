import { assertType, describe, expectTypeOf, it } from 'vitest';
import { MockEmbeddingModel } from '../internal/test-utils.js';
import type {
  EmbeddingModel,
  EmbeddingModelCallOptions,
  EmbeddingModelEmbed,
  SuccessContext,
} from '../types.js';
import { createRetryableModel } from './create-retryable-model.js';

describe('createRetryableModel', () => {
  it('should return EmbeddingModel for a model instance', () => {
    const retryable = createRetryableModel({
      model: new MockEmbeddingModel(),
      retries: [new MockEmbeddingModel(), { model: new MockEmbeddingModel() }],
    });

    assertType<EmbeddingModel>(retryable);
    expectTypeOf(retryable).toEqualTypeOf<EmbeddingModel>();
  });

  it('should return EmbeddingModel for a gateway string', () => {
    const retryable = createRetryableModel({
      model: 'openai/text-embedding-3-large',
      retries: ['openai/text-embedding-3-small'],
    });

    assertType<EmbeddingModel>(retryable);
    expectTypeOf(retryable).toEqualTypeOf<EmbeddingModel>();
  });

  it('should type SuccessContext correctly', () => {
    type Ctx = SuccessContext<EmbeddingModel>;

    expectTypeOf<Ctx['current']['model']>().toEqualTypeOf<EmbeddingModel>();
    expectTypeOf<
      Ctx['current']['result']
    >().toEqualTypeOf<EmbeddingModelEmbed>();
    expectTypeOf<
      Ctx['current']['options']
    >().toEqualTypeOf<EmbeddingModelCallOptions>();
  });
});
