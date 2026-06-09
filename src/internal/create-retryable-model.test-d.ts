import { assertType, describe, expectTypeOf, it } from 'vitest';
import { createRetryableModel } from './create-retryable-model.js';
import { MockEmbeddingModel, MockLanguageModel } from './test-utils.js';
import type {
  EmbeddingModel,
  EmbeddingModelCallOptions,
  EmbeddingModelEmbed,
  LanguageModel,
  LanguageModelCallOptions,
  LanguageModelGenerate,
  LanguageModelStream,
  SuccessContext,
} from '../types.js';

describe('createRetryableModel', () => {
  it('should return LanguageModel type for language model instance', () => {
    const model = new MockLanguageModel();
    const fallbackModel = new MockLanguageModel();

    const retryable = createRetryableModel({
      model,
      retries: [
        fallbackModel,
        {
          model: new MockLanguageModel(),
        },
      ],
    });

    assertType<LanguageModel>(retryable);
    expectTypeOf(retryable).toEqualTypeOf<LanguageModel>();
  });

  it('should return LanguageModel type for gateway string', () => {
    const fallbackModel = new MockLanguageModel();

    const retryable = createRetryableModel({
      model: 'openai/gpt-4.1',
      retries: [
        fallbackModel,
        'anthropic/claude-sonnet-4',
        { model: 'anthropic/claude-sonnet-4' },
      ],
    });

    assertType<LanguageModel>(retryable);
    expectTypeOf(retryable).toEqualTypeOf<LanguageModel>();
  });

  it('should type SuccessContext correctly for language models', () => {
    type Ctx = SuccessContext<LanguageModel>;

    expectTypeOf<Ctx['current']['model']>().toEqualTypeOf<LanguageModel>();
    expectTypeOf<Ctx['current']['result']>().toEqualTypeOf<
      LanguageModelGenerate | LanguageModelStream
    >();
    expectTypeOf<
      Ctx['current']['options']
    >().toEqualTypeOf<LanguageModelCallOptions>();
    expectTypeOf<Ctx['current']['type']>().toEqualTypeOf<'success'>();
  });

  it('should type SuccessContext correctly for embedding models', () => {
    type Ctx = SuccessContext<EmbeddingModel>;

    expectTypeOf<Ctx['current']['model']>().toEqualTypeOf<EmbeddingModel>();
    expectTypeOf<
      Ctx['current']['result']
    >().toEqualTypeOf<EmbeddingModelEmbed>();
    expectTypeOf<
      Ctx['current']['options']
    >().toEqualTypeOf<EmbeddingModelCallOptions>();
  });

  it('should return EmbeddingModel type for embedding model instance', () => {
    const model = new MockEmbeddingModel();
    const fallbackModel = new MockEmbeddingModel();

    const retryable = createRetryableModel({
      model,
      retries: [fallbackModel, { model: new MockEmbeddingModel() }],
    });

    assertType<EmbeddingModel>(retryable);
    expectTypeOf(retryable).toEqualTypeOf<EmbeddingModel>();
  });
});
