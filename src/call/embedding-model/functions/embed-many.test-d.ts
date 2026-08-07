import { embedMany } from 'ai';
import { describe, expectTypeOf, it } from 'vitest';
import { MockEmbeddingModel } from '../../../internal/test-utils.js';
import { retryableEmbed } from './embed.js';
import { retryableEmbedMany } from './embed-many.js';

const embeddingModel = MockEmbeddingModel.from();

describe('retryableEmbedMany', () => {
  it('should keep the result type identical to a direct call', async () => {
    // Act
    const direct = await embedMany({
      model: embeddingModel,
      values: ['hi', 'there'],
    });
    const wrapped = await retryableEmbedMany({
      model: embeddingModel,
      values: ['hi', 'there'],
      retry: [MockEmbeddingModel.from()],
    });

    // Assert
    expectTypeOf(wrapped.embeddings).toEqualTypeOf<typeof direct.embeddings>();
    expectTypeOf(wrapped.values).toEqualTypeOf<typeof direct.values>();
  });

  it('should accept its own overrides but not embed ones', () => {
    // Assert
    retryableEmbedMany({
      model: embeddingModel,
      values: ['hi'],
      retry: [{ model: embeddingModel, options: { values: ['rephrased'] } }],
    });

    retryableEmbedMany({
      model: embeddingModel,
      values: ['hi'],
      // @ts-expect-error `value` is an embed argument, not an embedMany one
      retry: [{ model: embeddingModel, options: { value: 'a' } }],
    });
  });

  it('should share a retryable that sets no options with embed', () => {
    // Arrange — nothing here can tell TS which entry point it is destined for.
    const fallback = { model: embeddingModel, maxAttempts: 2 };

    // Assert — `options?: never` is assignable to every INPUT.
    retryableEmbed({ model: embeddingModel, value: 'hi', retry: [fallback] });
    retryableEmbedMany({
      model: embeddingModel,
      values: ['hi'],
      retry: [fallback],
    });
  });

  it('should accept the bare-array shorthand', async () => {
    // Act
    const direct = await embedMany({ model: embeddingModel, values: ['hi'] });
    const wrapped = await retryableEmbedMany({
      model: embeddingModel,
      values: ['hi'],
      retry: [MockEmbeddingModel.from()],
    });

    // Assert — the shorthand does not disturb the entry point's own inference.
    expectTypeOf(wrapped.embeddings).toEqualTypeOf<typeof direct.embeddings>();
  });

  it('should accept the object form with hooks', () => {
    // Assert
    retryableEmbedMany({
      model: embeddingModel,
      values: ['hi'],
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
    const direct = await embedMany({ model: embeddingModel, values: ['hi'] });

    // Assert — the hook sees the same result the caller does.
    await retryableEmbedMany({
      model: embeddingModel,
      values: ['hi'],
      retry: {
        retries: [],
        onSuccess: (context) => {
          expectTypeOf(context.current.result.embeddings).toEqualTypeOf<
            typeof direct.embeddings
          >();
        },
      },
    });
  });
});
