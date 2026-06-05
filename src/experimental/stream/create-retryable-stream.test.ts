import { convertArrayToReadableStream } from '@ai-sdk/provider-utils/test';
import { APICallError, RetryError, streamText } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { createRetryable } from '../../create-retryable-model.js';
import {
  MockLanguageModel,
  mockStreamChunks,
  mockStreamOptions,
} from '../../internal/test-utils.js';
import { contentFilterTriggered } from '../../retryables/content-filter-triggered.js';
import type { LanguageModelStreamPart } from '../../types.js';
import {
  createRetryableStream,
  type RetryableStreamOptions,
} from './create-retryable-stream.js';

const prompt = 'Hello!';

/** A `finish` reporting `content-filter`, with no content emitted before it. */
const contentFilterChunks: Array<LanguageModelStreamPart> = [
  { type: 'stream-start', warnings: [] },
  { type: 'text-start', id: '0' },
  { type: 'text-end', id: '0' },
  {
    type: 'finish',
    finishReason: { unified: 'content-filter', raw: undefined },
    usage: {
      inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 20, text: 0, reasoning: 0 },
    },
  },
];

/** An Azure-style content-filter `APICallError` (code `content_filter`). */
const contentFilterError = new APICallError({
  message: 'The response was filtered due to the content management policy.',
  url: '',
  requestBodyValues: {},
  statusCode: 400,
  responseHeaders: {},
  responseBody: '',
  isRetryable: false,
  data: {
    error: {
      message:
        'The response was filtered due to the content management policy.',
      type: null,
      param: 'prompt',
      code: 'content_filter',
    },
  },
});

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
          if (abortSignal?.aborted) controller.error(abortSignal.reason);
          else
            abortSignal?.addEventListener(
              'abort',
              () => controller.error(abortSignal.reason),
              { once: true },
            );
        },
      }),
    }),
  });

/** A model that streams one content delta, then stalls until aborted. */
const partialThenStallStreamModel = () =>
  new MockLanguageModel({
    doStream: async ({ abortSignal }) => ({
      stream: new ReadableStream<LanguageModelStreamPart>({
        start(controller) {
          controller.enqueue({ type: 'stream-start', warnings: [] });
          controller.enqueue({ type: 'text-start', id: '1' });
          controller.enqueue({ type: 'text-delta', id: '1', delta: 'partial' });
          abortSignal?.addEventListener(
            'abort',
            () => controller.error(abortSignal.reason),
            { once: true },
          );
        },
      }),
    }),
  });

/** A model that emits `stream-start` then an `error` part before any content. */
const errorAtStartStreamModel = (error: unknown) =>
  new MockLanguageModel({
    doStream: {
      stream: convertArrayToReadableStream<LanguageModelStreamPart>([
        { type: 'stream-start', warnings: [] },
        { type: 'error', error },
      ]),
    },
  });

/** A model that streams one delta, then errors mid-stream after content. */
const errorAfterContentStreamModel = (error: unknown) =>
  new MockLanguageModel({
    doStream: {
      stream: convertArrayToReadableStream<LanguageModelStreamPart>([
        { type: 'stream-start', warnings: [] },
        { type: 'text-start', id: '1' },
        { type: 'text-delta', id: '1', delta: 'partial' },
        { type: 'error', error },
      ]),
    },
  });

/** A model that streams the full successful `mockStreamChunks` ("Hello, world!"). */
const okStreamModel = () =>
  new MockLanguageModel({
    doStream: { stream: convertArrayToReadableStream(mockStreamChunks) },
  });

/**
 * Inline `streamText` glue: re-run the whole `streamText` call per attempt with
 * the attempt's model and fresh deadline signal, deciding commit/fail-over from
 * `fullStream`. This is the shape a `streamText` drop-in built on
 * {@link createRetryableStream} takes — `prompt` and `messages` come from the
 * call `args`, so the attempt's low-level `options.prompt` is stripped.
 */
const retryableStreamText = (
  options: RetryableStreamOptions,
  args: Omit<Parameters<typeof streamText>[0], 'model'>,
) => {
  const retryableStream = createRetryableStream(options);
  return retryableStream(
    (attempt) => {
      const { prompt: _prompt, ...overrides } = attempt.options;
      return streamText({
        ...args,
        ...overrides,
        model: attempt.model,
        abortSignal: attempt.abortSignal,
        /**
         * Default `onError` to a no-op: this wrapper detects errors from
         * `fullStream` itself, so `streamText`'s default `console.error` would
         * just log every recovered attempt. A caller `onError` is respected.
         */
        onError: args.onError ?? (() => {}),
      } as Parameters<typeof streamText>[0]);
    },
    { abortSignal: args.abortSignal },
  );
};

describe('createRetryableStream', () => {
  describe('commit detection', () => {
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

    it('should keep reading past preamble parts until content', async () => {
      // Arrange — leading non-content parts, then a content part.
      const result = streamOf([
        { type: 'start' },
        { type: 'start-step' },
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
  });

  describe('streamText integration', () => {
    it('should return a usable result when the first attempt succeeds', async () => {
      // Arrange
      const primary = okStreamModel();

      // Act
      const result = await retryableStreamText(
        { model: primary, retries: [] },
        { prompt, ...mockStreamOptions },
      );

      // Assert
      expect(await result.text).toBe('Hello, world!');
      expect(primary.doStream).toHaveBeenCalledTimes(1);
    });

    it('should fall back to the next model when the stream fails before content', async () => {
      // Arrange
      const primary = new MockLanguageModel({
        doStream: new Error('creation failed'),
      });
      const fallback = okStreamModel();

      // Act
      const result = await retryableStreamText(
        { model: primary, retries: [fallback] },
        { prompt, ...mockStreamOptions },
      );

      // Assert
      expect(await result.text).toBe('Hello, world!');
      expect(primary.doStream).toHaveBeenCalledTimes(1);
      expect(fallback.doStream).toHaveBeenCalledTimes(1);
    });

    describe('error-based retries', () => {
      it('should fall back when the stream errors before any content', async () => {
        // Arrange
        const primary = errorAtStartStreamModel(
          new Error('stream-start failed'),
        );
        const fallback = okStreamModel();

        // Act
        const result = await retryableStreamText(
          { model: primary, retries: [fallback] },
          { prompt, ...mockStreamOptions },
        );

        // Assert
        expect(await result.text).toBe('Hello, world!');
        expect(primary.doStream).toHaveBeenCalledTimes(1);
        expect(fallback.doStream).toHaveBeenCalledTimes(1);
      });

      it('should fall back across consecutive errors', async () => {
        // Arrange
        const primary = errorAtStartStreamModel(new Error('first failed'));
        const second = errorAtStartStreamModel(new Error('second failed'));
        const third = okStreamModel();

        // Act
        const result = await retryableStreamText(
          { model: primary, retries: [second, third] },
          { prompt, ...mockStreamOptions },
        );

        // Assert
        expect(await result.text).toBe('Hello, world!');
        expect(primary.doStream).toHaveBeenCalledTimes(1);
        expect(second.doStream).toHaveBeenCalledTimes(1);
        expect(third.doStream).toHaveBeenCalledTimes(1);
      });

      it('should NOT fall back when the stream errors after content started', async () => {
        // Arrange
        const primary = errorAfterContentStreamModel(new Error('mid-stream'));
        const fallback = okStreamModel();

        // Act
        const result = await retryableStreamText(
          { model: primary, retries: [fallback] },
          { prompt, ...mockStreamOptions },
        );

        let text = '';
        try {
          for await (const part of result.fullStream) {
            if (part.type === 'text-delta') text += part.text ?? '';
          }
        } catch {
          /* mid-stream error after content */
        }

        // Assert — committed on the first content part, so no fail-over.
        expect(text).toBe('partial');
        expect(primary.doStream).toHaveBeenCalledTimes(1);
        expect(fallback.doStream).toHaveBeenCalledTimes(0);
      });

      it('should reject when no retryable matches a pre-content error', async () => {
        // Arrange
        const primary = errorAtStartStreamModel(new Error('boom'));

        // Act
        const result = retryableStreamText(
          { model: primary, retries: [] },
          { prompt, ...mockStreamOptions },
        );

        // Assert
        await expect(result).rejects.toThrow();
        expect(primary.doStream).toHaveBeenCalledTimes(1);
      });

      it('should call onError and onRetry around a pre-content fail-over', async () => {
        // Arrange
        const primary = errorAtStartStreamModel(new Error('boom'));
        const fallback = okStreamModel();
        const onError = vi.fn();
        const onRetry = vi.fn();

        // Act
        await retryableStreamText(
          { model: primary, retries: [fallback], onError, onRetry },
          { prompt, ...mockStreamOptions },
        );

        // Assert
        expect(onError.mock.calls.length).toBe(1);
        expect(onRetry.mock.calls.length).toBe(1);
      });
    });

    describe('streamText-level deadlines', () => {
      it('should recover a timeout.chunkMs deadline', async () => {
        // Arrange
        const primary = stallStreamModel();
        const fallback = okStreamModel();

        // Act
        const result = await retryableStreamText(
          { model: primary, retries: [fallback] },
          { prompt, timeout: { chunkMs: 50 }, ...mockStreamOptions },
        );

        // Assert
        expect(await result.text).toBe('Hello, world!');
        expect(fallback.doStream).toHaveBeenCalledTimes(1);
      });

      it('should recover a timeout.stepMs deadline', async () => {
        // Arrange
        const primary = stallStreamModel();
        const fallback = okStreamModel();

        // Act
        const result = await retryableStreamText(
          { model: primary, retries: [fallback] },
          { prompt, timeout: { stepMs: 50 }, ...mockStreamOptions },
        );

        // Assert
        expect(await result.text).toBe('Hello, world!');
        expect(fallback.doStream).toHaveBeenCalledTimes(1);
      });

      it('should recover a timeout.totalMs deadline', async () => {
        // Arrange
        const primary = stallStreamModel();
        const fallback = okStreamModel();

        // Act
        const result = await retryableStreamText(
          { model: primary, retries: [fallback] },
          { prompt, timeout: { totalMs: 50 }, ...mockStreamOptions },
        );

        // Assert
        expect(await result.text).toBe('Hello, world!');
        expect(fallback.doStream).toHaveBeenCalledTimes(1);
      });

      it('should recover an inbound abortSignal deadline with a per-attempt timeout', async () => {
        // Arrange
        const primary = stallStreamModel();
        const fallback = okStreamModel();

        // Act
        const result = await retryableStreamText(
          { model: primary, retries: [{ model: fallback, timeout: 5_000 }] },
          {
            prompt,
            abortSignal: AbortSignal.timeout(50),
            ...mockStreamOptions,
          },
        );

        // Assert
        expect(await result.text).toBe('Hello, world!');
        expect(fallback.doStream).toHaveBeenCalledTimes(1);
      });

      it('should give each attempt a fresh deadline signal', async () => {
        // Arrange
        const signals: Array<AbortSignal | undefined> = [];
        const primary = new MockLanguageModel({
          doStream: async ({ abortSignal }) => {
            signals.push(abortSignal);
            return {
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
            };
          },
        });
        const fallback = new MockLanguageModel({
          doStream: async ({ abortSignal }) => {
            signals.push(abortSignal);
            return { stream: convertArrayToReadableStream(mockStreamChunks) };
          },
        });

        // Act
        const result = await retryableStreamText(
          { model: primary, retries: [fallback] },
          { prompt, timeout: { chunkMs: 50 }, ...mockStreamOptions },
        );
        await result.text;

        // Assert
        expect(signals.length).toBe(2);
        expect(signals[0]).not.toBe(signals[1]);
        expect(signals[0]!.aborted).toBe(true);
        expect(signals[1]!.aborted).toBe(false);
      });

      it('should NOT recover a deadline that fires after content started', async () => {
        // Arrange
        const primary = partialThenStallStreamModel();
        const fallback = okStreamModel();

        // Act
        const result = await retryableStreamText(
          { model: primary, retries: [fallback] },
          { prompt, timeout: { chunkMs: 50 }, ...mockStreamOptions },
        );

        // Drain tolerantly: the post-content deadline surfaces an abort.
        let text = '';
        try {
          for await (const part of result.fullStream) {
            if (part.type === 'text-delta') text += part.text ?? '';
          }
        } catch {
          /* deadline abort after content */
        }

        // Assert — committed on the first delta, so no fail-over.
        expect(text).toBe('partial');
        expect(fallback.doStream).toHaveBeenCalledTimes(0);
      });

      it('should NOT retry a genuine caller cancellation', async () => {
        // Arrange
        const primary = stallStreamModel();
        const fallback = okStreamModel();
        const controller = new AbortController();
        controller.abort(
          Object.assign(new Error('user cancelled'), { name: 'AbortError' }),
        );

        // Act
        const result = retryableStreamText(
          { model: primary, retries: [fallback] },
          { prompt, abortSignal: controller.signal, ...mockStreamOptions },
        );

        // Assert
        await expect(result).rejects.toThrow();
        expect(fallback.doStream).toHaveBeenCalledTimes(0);
      });
    });

    describe('content-filter (parity with the model-layer retryable)', () => {
      it('should fail over when the stream throws a content-filter error', async () => {
        // Arrange
        const primary = new MockLanguageModel({ doStream: contentFilterError });
        const fallback = okStreamModel();

        // Act
        const result = await retryableStreamText(
          { model: primary, retries: [contentFilterTriggered(fallback)] },
          { prompt, ...mockStreamOptions },
        );

        // Assert
        expect(await result.text).toBe('Hello, world!');
        expect(primary.doStream).toHaveBeenCalledTimes(1);
        expect(fallback.doStream).toHaveBeenCalledTimes(1);
      });

      it('should fail over when the finish part reports content-filter before any content', async () => {
        // Arrange
        const primary = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(contentFilterChunks),
          },
        });
        const fallback = okStreamModel();

        // Act
        const result = await retryableStreamText(
          { model: primary, retries: [contentFilterTriggered(fallback)] },
          { prompt, ...mockStreamOptions },
        );

        // Assert
        expect(await result.text).toBe('Hello, world!');
        expect(primary.doStream).toHaveBeenCalledTimes(1);
        expect(fallback.doStream).toHaveBeenCalledTimes(1);
      });

      it('should return the content-filter result terminally when no retryable matches', async () => {
        // Arrange — content-filter finish, but no retryable handles it.
        const primary = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(contentFilterChunks),
          },
        });
        const fallback = okStreamModel();

        // Act
        const result = await retryableStreamText(
          { model: primary, retries: [fallback] },
          { prompt, ...mockStreamOptions },
        );

        // Assert — resolves to the primary's (empty) result, no fail-over.
        expect(await result.text).toBe('');
        expect(primary.doStream).toHaveBeenCalledTimes(1);
        expect(fallback.doStream).toHaveBeenCalledTimes(0);
      });

      it('should call onRetry but NOT onError for a result-based fail-over', async () => {
        // Arrange
        const primary = new MockLanguageModel({
          doStream: {
            stream: convertArrayToReadableStream(contentFilterChunks),
          },
        });
        const fallback = okStreamModel();
        const onError = vi.fn();
        const onRetry = vi.fn();

        // Act
        await retryableStreamText(
          {
            model: primary,
            retries: [contentFilterTriggered(fallback)],
            onError,
            onRetry,
          },
          { prompt, ...mockStreamOptions },
        );

        // Assert — result attempts are not errors (parity with the model layer).
        expect(onRetry.mock.calls.length).toBe(1);
        expect(onError.mock.calls.length).toBe(0);
      });
    });

    it('should throw a RetryError after all attempts are exhausted', async () => {
      // Arrange
      const primary = new MockLanguageModel({ doStream: new Error('first') });
      const fallback = new MockLanguageModel({ doStream: new Error('second') });

      // Act
      const result = retryableStreamText(
        { model: primary, retries: [fallback] },
        { prompt, ...mockStreamOptions },
      );

      // Assert
      await expect(result).rejects.toThrow();
      await result.catch((e) => expect(RetryError.isInstance(e)).toBe(true));
    });

    it('should bypass retries when disabled', async () => {
      // Arrange
      const primary = new MockLanguageModel({ doStream: new Error('boom') });
      const fallback = okStreamModel();

      // Act
      const result = retryableStreamText(
        { model: primary, retries: [fallback], disabled: true },
        { prompt, ...mockStreamOptions },
      );

      // Assert
      await expect(result).rejects.toThrow();
      expect(primary.doStream).toHaveBeenCalledTimes(1);
      expect(fallback.doStream).toHaveBeenCalledTimes(0);
    });

    describe('deferred consumption', () => {
      it('should let the caller drive the body via toUIMessageStreamResponse', async () => {
        // Arrange — fail over before content, then let the caller consume.
        const primary = new MockLanguageModel({ doStream: new Error('boom') });
        const fallback = okStreamModel();

        // Act
        const result = await retryableStreamText(
          { model: primary, retries: [fallback] },
          { prompt, ...mockStreamOptions },
        );
        const response = result.toUIMessageStreamResponse();
        const body = await response.text();

        // Assert — the fallback body streams out through the caller's mechanism.
        expect(response.status).toBe(200);
        expect(body).toContain('Hello');
        expect(body).toContain('world!');
      });
    });

    describe('user callbacks', () => {
      it('should forward onChunk and onFinish on a successful stream', async () => {
        // Arrange
        const onChunk = vi.fn();
        const onFinish = vi.fn();

        // Act
        const result = await retryableStreamText(
          { model: okStreamModel(), retries: [] },
          { prompt, onChunk, onFinish, ...mockStreamOptions },
        );
        await result.text;

        // Assert
        expect(onChunk.mock.calls.length).toBeGreaterThan(0);
        expect(onFinish.mock.calls.length).toBe(1);
      });

      it('should forward a post-commit error to the caller onError', async () => {
        // Arrange
        const onError = vi.fn();

        // Act
        const result = await retryableStreamText(
          {
            model: errorAfterContentStreamModel(new Error('mid-stream')),
            retries: [okStreamModel()],
          },
          { prompt, onError, ...mockStreamOptions },
        );
        await result.text;

        // Assert — committed on the first delta, so the error reaches the caller.
        expect(onError.mock.calls.length).toBe(1);
      });
    });

    describe('contrast with the model-layer wrapper', () => {
      it('model-layer createRetryable cannot recover a streamText deadline', async () => {
        // Arrange — same stalling primary + fallback, but the retry lives BELOW
        // streamText (wrapping doStream) instead of around the whole call.
        const primary = stallStreamModel();
        const fallback = okStreamModel();
        const result = streamText({
          model: createRetryable({ model: primary, retries: [fallback] }),
          prompt,
          maxRetries: 0,
          timeout: { totalMs: 50 },
          onError: () => {},
          ...mockStreamOptions,
        });

        // Act — bound the drain: the aborted stream may never cleanly settle,
        // which is itself a symptom of the discarded fallback (see issue #50).
        let text = '';
        const drain = (async () => {
          for await (const part of result.fullStream) {
            if (part.type === 'text-delta') text += part.text ?? '';
          }
        })();
        await Promise.race([
          drain.catch(() => {}),
          new Promise((resolve) => setTimeout(resolve, 500)),
        ]);

        // Assert — the fallback output never reaches the consumer.
        expect(text).not.toBe('Hello, world!');
      }, 10_000);
    });
  });
});
