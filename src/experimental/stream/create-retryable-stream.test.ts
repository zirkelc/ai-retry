import { RetryError, streamText } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { createRetryable } from '../../create-retryable-model.js';
import {
  contentFilterError,
  contentFilterStreamChunks,
  errorStreamChunks,
  MockLanguageModel,
  mockStream,
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

/** A result whose `fullStream` emits the given parts once. */
const streamOf = (parts: Array<unknown>) => ({
  fullStream: new ReadableStream<unknown>({
    start(controller) {
      for (const part of parts) controller.enqueue(part);
      controller.close();
    },
  }),
});

/** A model that streams the full successful `mockStreamChunks` ("Hello, world!"). */
const okStreamModel = () =>
  new MockLanguageModel({ doStream: mockStream(mockStreamChunks) });

/** A model that emits `stream-start` then an `error` part before any content. */
const errorAtStartStreamModel = (error: unknown) =>
  new MockLanguageModel({ doStream: mockStream(errorStreamChunks(error)) });

/** A model that streams one delta, then errors mid-stream after content. */
const errorAfterContentStreamModel = (error: unknown) =>
  new MockLanguageModel({
    doStream: mockStream([
      { type: 'stream-start', warnings: [] },
      { type: 'text-start', id: '1' },
      { type: 'text-delta', id: '1', delta: 'partial' },
      { type: 'error', error },
    ]),
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

/** A model that finishes with `content-filter` before any content (result-based). */
const contentFilterFinishModel = () =>
  new MockLanguageModel({ doStream: mockStream(contentFilterStreamChunks) });

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
    it('should commit on the first content part', async () => {
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

    it('should commit on a stream that produces no content', async () => {
      // Arrange — preamble only, then end-of-stream (e.g. an empty completion).
      const result = streamOf([{ type: 'start' }, { type: 'start-step' }]);
      const retryableStream = createRetryableStream({
        model: new MockLanguageModel(),
        retries: [],
      });

      // Act
      const committed = await retryableStream(() => result);

      // Assert
      expect(committed).toBe(result);
    });

    it('should fail over on a pre-content error part', async () => {
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

    it('should fail over on a pre-content abort part', async () => {
      // Arrange
      const primary = new MockLanguageModel();
      const fallback = new MockLanguageModel();
      const fallbackResult = streamOf([{ type: 'text-delta', text: 'OK' }]);
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [fallback],
      });

      // Act
      const committed = await retryableStream((attempt) =>
        attempt.model === primary
          ? streamOf([{ type: 'abort' }])
          : fallbackResult,
      );

      // Assert
      expect(committed).toBe(fallbackResult);
    });

    it('should NOT fail over once content has started', async () => {
      // Arrange — an error after the first content part must not fail over.
      const result = streamOf([
        { type: 'text-delta', text: 'OK' },
        { type: 'error', error: new Error('mid-stream') },
      ]);
      const fallback = new MockLanguageModel();
      const retryableStream = createRetryableStream({
        model: new MockLanguageModel(),
        retries: [fallback],
      });

      // Act
      const committed = await retryableStream(() => result);

      // Assert — committed on the first content part; the trailing error is the
      // caller's to handle.
      expect(committed).toBe(result);
      expect(fallback.doStream).toHaveBeenCalledTimes(0);
    });
  });

  describe('retries', () => {
    it('should throw a RetryError after all attempts are exhausted', async () => {
      // Arrange
      const primary = new MockLanguageModel();
      const fallback = new MockLanguageModel();
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [fallback],
      });

      // Act
      const result = retryableStream(() =>
        streamOf([{ type: 'error', error: new Error('boom') }]),
      );

      // Assert
      await expect(result).rejects.toThrow();
      await result.catch((e) => expect(RetryError.isInstance(e)).toBe(true));
    });
  });

  describe('disabled', () => {
    it('should bypass retries when disabled', async () => {
      // Arrange
      const primary = new MockLanguageModel();
      const fallback = new MockLanguageModel();
      const error = new Error('boom');
      const models: Array<unknown> = [];
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [fallback],
        disabled: true,
      });

      // Act
      const result = retryableStream((attempt) => {
        models.push(attempt.model);
        return streamOf([{ type: 'error', error }]);
      });

      // Assert
      await expect(result).rejects.toThrow();
      await result.catch((e) => expect(e).toBe(error));
      expect(models.length).toBe(1);
    });
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

  describe('error-based retries', () => {
    it('should fall back when stream creation fails', async () => {
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

    it('should fall back when the stream errors before any content', async () => {
      // Arrange
      const primary = errorAtStartStreamModel(new Error('stream-start failed'));
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

    it('should fall back on a content-filter error part', async () => {
      // Arrange — content-filter surfaces as an error (not a finish) here.
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
      expect(onError).toHaveBeenCalledTimes(1);
      expect(onRetry).toHaveBeenCalledTimes(1);
    });
  });

  describe('result-based conditions', () => {
    it('should ignore a content-filter finish without failing over', async () => {
      // Arrange — a content-filter *finish* (no content) is result-based; the
      // error-only call layer never sees it, so it streams through unchanged.
      const primary = contentFilterFinishModel();
      const fallback = okStreamModel();

      // Act
      const result = await retryableStreamText(
        { model: primary, retries: [contentFilterTriggered(fallback)] },
        { prompt, ...mockStreamOptions },
      );

      // Assert — no fail-over, no side effects.
      expect(await result.finishReason).toBe('content-filter');
      expect(primary.doStream).toHaveBeenCalledTimes(1);
      expect(fallback.doStream).toHaveBeenCalledTimes(0);
    });
  });

  describe('deadlines', () => {
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
          return mockStream(mockStreamChunks);
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

  describe('disabled', () => {
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
      expect(onChunk).toHaveBeenCalled();
      expect(onFinish).toHaveBeenCalledTimes(1);
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
      expect(onError).toHaveBeenCalledTimes(1);
    });
  });
});

describe('streamText integration with a retryable model', () => {
  it('should recover a content-filter finish at the model layer', async () => {
    // Arrange — the inner createRetryable handles the content-filter finish
    // BELOW streamText; the outer call layer never fails over.
    const primary = contentFilterFinishModel();
    const modelFallback = okStreamModel();
    const callFallback = okStreamModel();
    const inner = createRetryable({
      model: primary,
      retries: [contentFilterTriggered(modelFallback)],
    });

    // Act
    const result = await retryableStreamText(
      { model: inner, retries: [callFallback] },
      { prompt, ...mockStreamOptions },
    );

    // Assert — recovered inside the model layer.
    expect(await result.text).toBe('Hello, world!');
    expect(primary.doStream).toHaveBeenCalledTimes(1);
    expect(modelFallback.doStream).toHaveBeenCalledTimes(1);
    expect(callFallback.doStream).toHaveBeenCalledTimes(0);
  });

  it('should recover a streamText deadline at the call layer', async () => {
    // Arrange — a streamText deadline tears the stream down below the inner
    // model-layer retry, which cannot recover it (#50); the outer call layer
    // re-runs the whole call instead. The inner retry has its own fallback to
    // prove the deadline bypasses the model layer entirely.
    const primary = stallStreamModel();
    const modelFallback = okStreamModel();
    const callFallback = okStreamModel();
    const inner = createRetryable({ model: primary, retries: [modelFallback] });

    // Act
    const result = await retryableStreamText(
      { model: inner, retries: [callFallback] },
      { prompt, timeout: { totalMs: 50 }, ...mockStreamOptions },
    );

    // Assert — recovered by the call layer; the model layer never saw it.
    expect(await result.text).toBe('Hello, world!');
    expect(modelFallback.doStream).toHaveBeenCalledTimes(0);
    expect(callFallback.doStream).toHaveBeenCalledTimes(1);
  });

  describe('contrast', () => {
    it('a retryable model alone cannot recover a streamText deadline', async () => {
      // Arrange — the retry lives BELOW streamText (wrapping doStream); a
      // deadline tears the stream down before the fallback can be consumed.
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

    it('a retryable stream alone cannot recover a content-filter finish', async () => {
      // Arrange — content-filter is result-based; the error-only call layer
      // streams the filtered result through instead of failing over.
      const primary = contentFilterFinishModel();
      const fallback = okStreamModel();

      // Act
      const result = await retryableStreamText(
        { model: primary, retries: [contentFilterTriggered(fallback)] },
        { prompt, ...mockStreamOptions },
      );

      // Assert
      expect(await result.finishReason).toBe('content-filter');
      expect(fallback.doStream).toHaveBeenCalledTimes(0);
    });
  });
});
