import { streamText, tool } from 'ai';
import { describe, expectTypeOf, it } from 'vitest';
import { z } from 'zod';
import { MockLanguageModel } from '../../internal/test-utils.js';
import type { CallFinishReason } from '../types.js';
import { retryableStreamText } from './stream-text.js';

const model = MockLanguageModel.from();

const weather = tool({
  description: 'get the weather',
  inputSchema: z.object({ city: z.string() }),
  execute: async ({ city }) => ({ city, temp: 21 }),
});

describe('retryableStreamText', () => {
  it('should keep the result type identical to a direct call', async () => {
    // Act
    const direct = streamText({ model, prompt: 'hi', tools: { weather } });
    const wrapped = await retryableStreamText({
      model,
      prompt: 'hi',
      tools: { weather },
      retry: [MockLanguageModel.from()],
    });

    // Assert
    expectTypeOf(await wrapped.toolResults).toEqualTypeOf<
      Awaited<typeof direct.toolResults>
    >();
    expectTypeOf(await wrapped.steps).toEqualTypeOf<
      Awaited<typeof direct.steps>
    >();
    expectTypeOf(await wrapped.content).toEqualTypeOf<
      Awaited<typeof direct.content>
    >();
    expectTypeOf(wrapped.textStream).toEqualTypeOf<typeof direct.textStream>();
  });

  it('should return a promise, unlike the synchronous original', () => {
    // Act
    const wrapped = retryableStreamText({ model, prompt: 'hi' });

    // Assert — the loop must know which attempt won before it can hand a
    // result back, so this is the one place the signature differs.
    expectTypeOf(wrapped).toEqualTypeOf<Promise<Awaited<typeof wrapped>>>();
  });

  it('should narrow activeTools against the tools map', () => {
    // Assert
    retryableStreamText({
      model,
      prompt: 'hi',
      tools: { weather },
      // @ts-expect-error 'nope' is not a configured tool
      activeTools: ['nope'],
    });
  });

  it('should accept the bare-array shorthand', async () => {
    // Act
    const direct = streamText({ model, prompt: 'hi', tools: { weather } });
    const wrapped = await retryableStreamText({
      model,
      prompt: 'hi',
      tools: { weather },
      retry: [MockLanguageModel.from()],
    });

    // Assert — the shorthand does not disturb the entry point's own inference.
    expectTypeOf(await wrapped.toolResults).toEqualTypeOf<
      Awaited<typeof direct.toolResults>
    >();
  });

  it('should accept the object form with hooks', () => {
    // Assert
    retryableStreamText({
      model,
      prompt: 'hi',
      tools: { weather },
      retry: {
        retries: [MockLanguageModel.from()],
        disabled: false,
        onError: () => {},
        onRetry: () => {},
        onFailure: () => {},
      },
    });
  });

  it('should type onSuccess with the entry point result', async () => {
    // Act
    const direct = streamText({ model, prompt: 'hi', tools: { weather } });

    // Assert — the hook sees the same result the caller does, and for a stream
    // that means the *unsettled* one: `onCommit` fires at the commit point,
    // so every field is still the promise a direct call hands back.
    await retryableStreamText({
      model,
      prompt: 'hi',
      tools: { weather },
      retry: {
        retries: [],
        onCommit: (context) => {
          expectTypeOf(context.current.result.toolResults).toEqualTypeOf<
            typeof direct.toolResults
          >();
        },
      },
    });
  });

  it('should take every deadline, streaming windows included', () => {
    // Assert — the counterpart to `generateText` rejecting these two.
    retryableStreamText({ model, prompt: 'Hello!', timeout: 5_000 });
    retryableStreamText({
      model,
      prompt: 'Hello!',
      timeout: { totalMs: 30_000, firstChunkMs: 2_000, chunkMs: 500 },
    });
    retryableStreamText({
      model,
      prompt: 'Hello!',
      retry: [{ model, timeout: { firstChunkMs: 2_000, chunkMs: 500 } }],
    });
    retryableStreamText({
      model,
      prompt: 'Hello!',
      retry: [{ model, timeout: { totalMs: 5_000, stepMs: 2_000 } }],
    });
  });

  it('should offer both ends of the stream, told apart by their contexts', () => {
    // Assert — `onCommit` when the attempt becomes the caller's, `onSuccess`
    // when it ends well. The contexts differ because the knowledge does.
    retryableStreamText({
      model,
      prompt: 'hi',
      retry: {
        retries: [],
        onCommit: (context) => {
          expectTypeOf(context.current.type).toEqualTypeOf<'commit'>();
          // @ts-expect-error nothing has finished yet, so there is no reason
          context.current.finishReason;
        },
        onSuccess: (context) => {
          expectTypeOf(context.current.type).toEqualTypeOf<'success'>();
          expectTypeOf(context.current.finishReason).toEqualTypeOf<
            CallFinishReason | undefined
          >();
        },
      },
    });
  });
});
