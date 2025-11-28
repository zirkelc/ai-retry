import { assertType, describe, expectTypeOf, it } from 'vitest';
import { createRetryable } from './create-retryable-model.js';
import { MockEmbeddingModel, MockLanguageModel } from './test-utils.js';
import type { EmbeddingModel, LanguageModel } from './types.js';

describe('createRetryable', () => {
  it('should return LanguageModel type for language model instance', () => {
    const model = new MockLanguageModel();
    const fallbackModel = new MockLanguageModel();

    const retryable = createRetryable({
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

    const retryable = createRetryable({
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

  it('should return EmbeddingModel type for embedding model instance', () => {
    const model = new MockEmbeddingModel();
    const fallbackModel = new MockEmbeddingModel();

    const retryable = createRetryable({
      model,
      retries: [fallbackModel, { model: new MockEmbeddingModel() }],
    });

    assertType<EmbeddingModel>(retryable);
    expectTypeOf(retryable).toEqualTypeOf<EmbeddingModel>();
  });
});
