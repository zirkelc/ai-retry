import { generateText, tool } from 'ai';
import { describe, expectTypeOf, it } from 'vitest';
import { z } from 'zod';
import {
  MockEmbeddingModel,
  MockLanguageModel,
} from '../../internal/test-utils.js';
import { retryableGenerateText } from './generate-text.js';

/**
 * Every claim is asserted against a *direct* SDK call rather than a literal
 * type. A wrapper that lost inference and an SDK that never had it look
 * identical otherwise, and only the direct-call baseline tells them apart.
 */

const model = MockLanguageModel.from();

const weather = tool({
  description: 'get the weather',
  inputSchema: z.object({ city: z.string() }),
  execute: async ({ city }) => ({ city, temp: 21 }),
});

describe('retryableGenerateText', () => {
  it('should keep the result type identical to a direct call', async () => {
    // Act
    const direct = await generateText({
      model,
      prompt: 'hi',
      tools: { weather },
    });
    const wrapped = await retryableGenerateText({
      model,
      prompt: 'hi',
      tools: { weather },
      retry: [MockLanguageModel.from()],
    });

    // Assert
    expectTypeOf(wrapped.toolResults).toEqualTypeOf<
      typeof direct.toolResults
    >();
    expectTypeOf(wrapped.toolCalls).toEqualTypeOf<typeof direct.toolCalls>();
    expectTypeOf(wrapped.steps).toEqualTypeOf<typeof direct.steps>();
    expectTypeOf(wrapped.content).toEqualTypeOf<typeof direct.content>();
    expectTypeOf(wrapped.text).toEqualTypeOf<typeof direct.text>();
  });

  it('should narrow activeTools against the tools map', () => {
    // Assert
    retryableGenerateText({
      model,
      prompt: 'hi',
      tools: { weather },
      // @ts-expect-error 'nope' is not a configured tool
      activeTools: ['nope'],
    });
  });

  it('should reject an unknown argument', () => {
    // Assert — excess property checking survives the intersection.
    retryableGenerateText({
      model,
      prompt: 'hi',
      // @ts-expect-error not a generateText argument
      nonsense: true,
    });
  });

  it('should allow omitting retry entirely', () => {
    // Assert
    retryableGenerateText({ model, prompt: 'hi' });
  });

  it('should reject a fallback from the wrong model family', () => {
    // Assert
    retryableGenerateText({
      model,
      prompt: 'hi',
      // @ts-expect-error an embedding model is not a language fallback
      retry: [MockEmbeddingModel.from()],
    });
  });

  it('should accept overrides the entry point actually takes', () => {
    // Assert
    retryableGenerateText({
      model,
      prompt: 'hi',
      retry: [{ model, options: { prompt: 'rephrased', temperature: 0 } }],
    });
  });

  it('should reject overrides belonging to another entry point', () => {
    // Assert
    retryableGenerateText({
      model,
      prompt: 'hi',
      // @ts-expect-error `values` is an embedMany argument, not a generateText one
      retry: [{ model, options: { values: ['a'] } }],
    });
  });

  it('should accept the bare-array shorthand', async () => {
    // Act
    const direct = await generateText({
      model,
      prompt: 'hi',
      tools: { weather },
    });
    const wrapped = await retryableGenerateText({
      model,
      prompt: 'hi',
      tools: { weather },
      retry: [MockLanguageModel.from()],
    });

    // Assert — the shorthand does not disturb the entry point's own inference.
    expectTypeOf(wrapped.toolResults).toEqualTypeOf<
      typeof direct.toolResults
    >();
  });

  it('should accept the object form with hooks', () => {
    // Assert
    retryableGenerateText({
      model,
      prompt: 'hi',
      tools: { weather },
      retry: {
        retries: [MockLanguageModel.from()],
        disabled: false,
        onError: () => {},
        onRetry: () => {},
        onSettled: () => {},
      },
    });
  });

  it('should type onSettled with the entry point result', async () => {
    // Act
    const direct = await generateText({
      model,
      prompt: 'hi',
      tools: { weather },
    });

    // Assert — the hook sees the same result the caller does.
    await retryableGenerateText({
      model,
      prompt: 'hi',
      tools: { weather },
      retry: {
        retries: [],
        onSettled: (event) => {
          expectTypeOf(event.result?.toolResults).toEqualTypeOf<
            typeof direct.toolResults | undefined
          >();
          expectTypeOf(event.outcome).toEqualTypeOf<'success' | 'failure'>();
        },
      },
    });
  });

  it('should take every deadline a stepped, non-streaming call can measure', () => {
    // Assert
    retryableGenerateText({ model, prompt: 'Hello!', timeout: 5_000 });
    retryableGenerateText({
      model,
      prompt: 'Hello!',
      timeout: { totalMs: 5_000, stepMs: 2_000, toolMs: 1_000 },
    });
    retryableGenerateText({
      model,
      prompt: 'Hello!',
      retry: [{ model, timeout: { totalMs: 5_000, stepMs: 2_000 } }],
    });
  });

  it('should reject a streaming-only deadline on a retry', () => {
    // Arrange — the SDK's `timeout` argument accepts these two on
    // `generateText` and then never reads them. A retry deadline of ours does
    // not repeat that.
    retryableGenerateText({
      model,
      prompt: 'Hello!',
      // @ts-expect-error `firstChunkMs` needs a stream to measure against
      retry: [{ model, timeout: { firstChunkMs: 2_000 } }],
    });

    retryableGenerateText({
      model,
      prompt: 'Hello!',
      // @ts-expect-error nor does `chunkMs` mean anything here
      retry: [{ model, timeout: { chunkMs: 100 } }],
    });
  });
});
