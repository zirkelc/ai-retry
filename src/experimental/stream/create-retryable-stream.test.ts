import { convertArrayToReadableStream } from '@ai-sdk/provider-utils/test';
import { streamText } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  MockLanguageModel,
  mockStreamChunks,
  mockStreamOptions,
} from '../../internal/test-utils.js';
import type { LanguageModelStreamPart } from '../../types.js';
import { createRetryableStream } from './create-retryable-stream.js';

const prompt = 'Hello!';

/** A result whose `fullStream` emits the given parts once. */
const streamOf = (parts: Array<unknown>) => ({
  fullStream: new ReadableStream<unknown>({
    start(controller) {
      for (const part of parts) controller.enqueue(part);
      controller.close();
    },
  }),
});

/** A model whose stream stalls (emits only `stream-start`) until aborted. */
const stallStreamModel = () =>
  new MockLanguageModel({
    doStream: async ({ abortSignal }) => ({
      stream: new ReadableStream<LanguageModelStreamPart>({
        start(controller) {
          controller.enqueue({ type: 'stream-start', warnings: [] });
          abortSignal?.addEventListener(
            'abort',
            () => controller.error(abortSignal.reason),
            { once: true },
          );
        },
      }),
    }),
  });

const okStreamModel = () =>
  new MockLanguageModel({
    doStream: { stream: convertArrayToReadableStream(mockStreamChunks) },
  });

describe('createRetryableStream', () => {
  it('should commit on the first content part (default classifier)', async () => {
    // Arrange
    const result = streamOf([
      { type: 'stream-start' },
      { type: 'text-delta', text: 'OK' },
    ]);
    const retryableStream = createRetryableStream({
      model: new MockLanguageModel(),
      retries: [],
    });

    // Act
    const committed = await retryableStream(() => result);

    // Assert
    expect(committed).toBe(result);
  });

  it('should fail over to the next model on a pre-content error part', async () => {
    // Arrange
    const primary = new MockLanguageModel();
    const fallback = new MockLanguageModel();
    const fallbackResult = streamOf([{ type: 'text-delta', text: 'OK' }]);
    const models: Array<unknown> = [];
    const retryableStream = createRetryableStream({
      model: primary,
      retries: [fallback],
    });

    // Act
    const committed = await retryableStream((attempt) => {
      models.push(attempt.model);
      return attempt.model === primary
        ? streamOf([{ type: 'error', error: new Error('boom') }])
        : fallbackResult;
    });

    // Assert
    expect(committed).toBe(fallbackResult);
    expect(models.length).toBe(2);
    expect(models[0]).toBe(primary);
    expect(models[1]).toBe(fallback);
  });

  it('should support a custom classifier for a non-streamText stream', async () => {
    // Arrange — a stream whose parts do not follow the AI SDK protocol.
    const result = streamOf([{ kind: 'meta' }, { kind: 'data' }]);
    const retryableStream = createRetryableStream({
      model: new MockLanguageModel(),
      retries: [],
      classifyPart: (part) =>
        (part as { kind?: string }).kind === 'data' ? 'content' : 'preamble',
    });

    // Act
    const committed = await retryableStream(() => result);

    // Assert
    expect(committed).toBe(result);
  });

  it('should recover a streamText deadline end-to-end', async () => {
    // Arrange
    const primary = stallStreamModel();
    const fallback = okStreamModel();
    const retryableStream = createRetryableStream({
      model: primary,
      retries: [{ model: fallback, timeout: 5_000 }],
    });

    // Act
    const result = await retryableStream((attempt) =>
      streamText({
        model: attempt.model,
        prompt,
        maxRetries: 0,
        abortSignal: attempt.abortSignal,
        timeout: { chunkMs: 50 },
        ...mockStreamOptions,
      }),
    );

    // Assert
    expect(await result.text).toBe('Hello, world!');
    expect(fallback.doStream).toHaveBeenCalledTimes(1);
  });
});
