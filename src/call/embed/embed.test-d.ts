import { embed } from 'ai';
import { describe, expectTypeOf, it } from 'vitest';
import {
  MockEmbeddingModel,
  MockImageModel,
} from '../../internal/test-utils.js';
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
        onSettled: () => {},
      },
    });
  });

  it('should take a deadline as a number or a total budget', () => {
    // Assert
    retryableEmbed({ model: embeddingModel, value: 'hi', timeout: 5_000 });
    retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      timeout: { totalMs: 5_000 },
    });
  });

  it('should reject a deadline it could never enforce', () => {
    // Arrange — an embedding call has no steps and no stream, so these windows
    // describe stages that do not exist here.
    retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      // @ts-expect-error `stepMs` is not measurable around an embed call
      timeout: { stepMs: 5_000 },
    });

    retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      // @ts-expect-error nor is `chunkMs`
      timeout: { chunkMs: 100 },
    });
  });

  it('should reject a retry deadline it could never enforce', () => {
    // Assert — same rule for the retry's own deadline.
    retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      retry: [{ model: embeddingModel, timeout: 5_000 }],
    });

    retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      retry: [{ model: embeddingModel, timeout: { totalMs: 5_000 } }],
    });

    retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      // @ts-expect-error a stream window cannot bound an embed retry
      retry: [{ model: embeddingModel, timeout: { chunkMs: 100 } }],
    });
  });

  it('should type onSettled with the entry point result', async () => {
    // Act
    const direct = await embed({ model: embeddingModel, value: 'hi' });

    // Assert — the hook sees the same result the caller does.
    await retryableEmbed({
      model: embeddingModel,
      value: 'hi',
      retry: {
        retries: [],
        onSettled: (event) => {
          expectTypeOf(event.result?.embedding).toEqualTypeOf<
            typeof direct.embedding | undefined
          >();
          expectTypeOf(event.outcome).toEqualTypeOf<'success' | 'failure'>();
        },
      },
    });
  });
});
