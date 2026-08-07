import { embed } from 'ai';
import { describe, expectTypeOf, it } from 'vitest';
import {
  MockEmbeddingModel,
  MockImageModel,
} from '../../../internal/test-utils.js';
import { retryableEmbed } from './embed.js';

const embeddingModel = MockEmbeddingModel.from();

describe('retryableEmbed', () => {
  it('should keep the result type identical to a direct call', async () => {
    // Act
    const direct = await embed({ model: embeddingModel, value: 'hi' });
    const wrapped = await retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      retry: [MockEmbeddingModel.from()],
    });

    // Assert
    expectTypeOf(wrapped.embedding).toEqualTypeOf<typeof direct.embedding>();
    expectTypeOf(wrapped.usage).toEqualTypeOf<typeof direct.usage>();
    expectTypeOf(wrapped.value).toEqualTypeOf<typeof direct.value>();
  });

  it('should reject a fallback from the wrong model family', () => {
    // Assert
    retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      // @ts-expect-error an image model is not an embedding fallback
      retry: [MockImageModel.from()],
    });
  });

  it('should reject an unknown argument', () => {
    // Assert
    retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      // @ts-expect-error not an embed argument
      nonsense: true,
    });
  });

  it('should accept its own overrides but not embedMany ones', () => {
    // Assert
    retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      retry: [{ model: embeddingModel, options: { value: 'rephrased' } }],
    });

    retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      // @ts-expect-error `values` is an embedMany argument, not an embed one
      retry: [{ model: embeddingModel, options: { values: ['a', 'b'] } }],
    });
  });

  it('should accept the bare-array shorthand', async () => {
    // Act
    const direct = await embed({ model: embeddingModel, value: 'hi' });
    const wrapped = await retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      retry: [MockEmbeddingModel.from()],
    });

    // Assert — the shorthand does not disturb the entry point's own inference.
    expectTypeOf(wrapped.embedding).toEqualTypeOf<typeof direct.embedding>();
  });

  it('should accept the object form with hooks', () => {
    // Assert
    retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      retry: {
        retries: [MockEmbeddingModel.from()],
        disabled: false,
        onError: () => {},
        onRetry: () => {},
        onFailure: () => {},
      },
    });
  });

  it('should type onSuccess with the entry point result', async () => {
    // Act
    const direct = await embed({ model: embeddingModel, value: 'hi' });

    // Assert — the hook sees the same result the caller does.
    await retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      retry: {
        retries: [],
        onSuccess: (context) => {
          expectTypeOf(context.current.result.embedding).toEqualTypeOf<
            typeof direct.embedding
          >();
        },
      },
    });
  });
});
