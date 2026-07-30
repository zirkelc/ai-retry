import {
  createUIMessageStreamResponse,
  RetryError,
  stepCountIs,
  streamText,
  tool,
  toUIMessageStream,
} from 'ai';
import { z } from 'zod';
import { describe, expect, it, vi } from 'vitest';
import {
  aborted,
  error,
  finishReason,
  timeout,
} from '../../language-model/conditions/index.js';
import {
  contentFilterError,
  contentFilterStreamChunks,
  createRetryableModel,
  errorStreamChunks,
  Language,
  MockLanguageModel,
  mockStreamChunks,
  Streams,
} from '../../internal/test-utils.js';
import type {
  LanguageModelCallOptions,
  LanguageModelStreamPart,
} from '../../types.js';
import {
  createRetryableStream,
  type RetryableStreamOptions,
} from './create-retryable-stream.js';

const prompt = 'Hello!';

/**
 * These shapes mimic a `streamText` *result* (what `streamFn` returns), whose
 * `stream` yields `streamText`-level `TextStreamPart`s. `Streams.from` wraps the
 * parts into the `ReadableStream`. `error` parts use `Language.streamError`
 * (that part is identical at both layers), but `text-delta` (`.text`) and
 * `abort` are written as literals: ai-test-kit's `Language.*` build the
 * *provider*-level parts a model's `doStream` returns (one layer below), and it
 * has no `streamText`-level text-delta or `abort` builder. This is also why
 * `streamOf` is not `Language.streamResult()` — that builds a `doStream` result.
 */

/** A v7 `streamText`-shaped result: parts on `stream`. */
const streamOf = (parts: Array<unknown>) => ({ stream: Streams.from(parts) });

/** A `streamObject`-shaped result: parts on `fullStream` only. */
const fullStreamOf = (parts: Array<unknown>) => ({
  fullStream: Streams.from(parts),
});

/** A model that streams the full successful `mockStreamChunks` ("Hello, world!"). */
const okStreamModel = () =>
  MockLanguageModel.from({ doStream: mockStreamChunks });

/** A model that emits `stream-start` then an `error` part before any content. */
const errorAtStartStreamModel = (error: unknown) =>
  MockLanguageModel.from({ doStream: errorStreamChunks(error) });

/** A model that streams one delta, then errors mid-stream after content. */
const errorAfterContentStreamModel = (error: unknown) =>
  MockLanguageModel.from({
    doStream: [
      Language.streamStart(),
      ...Language.streamText(['partial'], { id: '1' }),
      Language.streamError(error),
    ],
  });

/**
 * A model whose stream stalls after emitting `preamble` parts, erroring only
 * once its `abortSignal` fires. Used to exercise `streamText`-level deadlines.
 */
const stallStreamModel = (
  preamble: Array<LanguageModelStreamPart> = [Language.streamStart()],
) =>
  MockLanguageModel.from({
    doStream: async ({ abortSignal }: LanguageModelCallOptions) => ({
      stream: new ReadableStream<LanguageModelStreamPart>({
        start(controller) {
          for (const part of preamble) controller.enqueue(part);
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
  stallStreamModel([
    Language.streamStart(),
    ...Language.streamText(['partial'], { id: '1' }),
  ]);

/** A model that finishes with `content-filter` before any content (result-based). */
const contentFilterFinishModel = () =>
  MockLanguageModel.from({ doStream: contentFilterStreamChunks });

/**
 * Inline `streamText` glue: re-run the whole `streamText` call per attempt with
 * the attempt's model and fresh deadline signal, deciding commit/fail-over from
 * the result's part stream. This is the shape a `streamText` drop-in built on
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
         * Default `onError` to a no-op: this wrapper detects errors from the
         * part stream itself, so `streamText`'s default `console.error` would
         * just log every recovered attempt. A caller `onError` is respected.
         */
        onError: args.onError ?? (() => {}),
      } as Parameters<typeof streamText>[0]);
    },
    { abortSignal: args.abortSignal },
  );
};

/**
 * Unit suite: drive `createRetryableStream` with synthetic stream results so
 * commit/fail-over is decided from the parts alone, no real `streamText`.
 */
describe('createRetryableStream', () => {
  describe('part stream resolution', () => {
    it('should read parts from `stream` (v7 streamText shape)', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fallbackResult = streamOf([{ type: 'text-delta', text: 'OK' }]);
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [fallback],
      });

      // Act — an error on `stream` before content must fail over.
      const committed = await retryableStream((attempt) =>
        attempt.model === primary
          ? streamOf([Language.streamError(new Error('x'))])
          : fallbackResult,
      );

      // Assert
      expect(committed).toBe(fallbackResult);
    });

    it('should fall back to `fullStream` (streamObject shape)', async () => {
      // Arrange — a streamObject-style result exposes parts only on fullStream.
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fallbackResult = fullStreamOf([{ type: 'text-delta', text: 'OK' }]);
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [fallback],
      });

      // Act
      const committed = await retryableStream((attempt) =>
        attempt.model === primary
          ? fullStreamOf([Language.streamError(new Error('x'))])
          : fallbackResult,
      );

      // Assert
      expect(committed).toBe(fallbackResult);
    });

    it('should prefer `stream` over `fullStream` when both are present', async () => {
      // Arrange — a v7 streamText result carries both; only `stream` is read, so
      // an error there fails over even though `fullStream` holds content.
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fallbackResult = streamOf([{ type: 'text-delta', text: 'OK' }]);
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [fallback],
      });

      // Act
      const committed = await retryableStream((attempt) =>
        attempt.model === primary
          ? {
              stream: Streams.from([Language.streamError(new Error('x'))]),
              fullStream: Streams.from([{ type: 'text-delta', text: 'nope' }]),
            }
          : fallbackResult,
      );

      // Assert
      expect(committed).toBe(fallbackResult);
    });
  });

  describe('commit detection', () => {
    it('should commit on the first content part', async () => {
      // Arrange
      const result = streamOf([
        { type: 'stream-start' },
        { type: 'text-delta', text: 'OK' },
      ]);
      const retryableStream = createRetryableStream({
        model: MockLanguageModel.from(),
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
        model: MockLanguageModel.from(),
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
        model: MockLanguageModel.from(),
        retries: [],
      });

      // Act
      const committed = await retryableStream(() => result);

      // Assert
      expect(committed).toBe(result);
    });

    it('should NOT fail over once content has started', async () => {
      // Arrange — an error after the first content part must not fail over.
      const result = streamOf([
        { type: 'text-delta', text: 'OK' },
        Language.streamError(new Error('mid-stream')),
      ]);
      const fallback = MockLanguageModel.from();
      const retryableStream = createRetryableStream({
        model: MockLanguageModel.from(),
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

  describe('pre-content failure', () => {
    it('should fail over on an error part', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
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
          ? streamOf([Language.streamError(new Error('boom'))])
          : fallbackResult;
      });

      // Assert
      expect(committed).toBe(fallbackResult);
      expect(models).toEqual([primary, fallback]);
    });

    it('should fail over on a bare abort part', async () => {
      // Arrange — an unconditional fallback matches any error.
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
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

    it('should reconstruct a TimeoutError from an abort reason so timeout() matches', async () => {
      // Arrange — a streamText stepMs/chunkMs/totalMs deadline emits an `abort`
      // part whose `reason` is the serialized `"<name>: <message>"`, since it
      // aborts an internal controller, not the attempt's own signal.
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fallbackResult = streamOf([{ type: 'text-delta', text: 'OK' }]);
      const seen: Array<{ name?: string; message?: string }> = [];
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [timeout().switch({ model: fallback })],
        onError: (ctx) => {
          const e = (ctx.current as { error?: Error }).error;
          seen.push({ name: e?.name, message: e?.message });
        },
      });

      // Act
      const committed = await retryableStream((attempt) =>
        attempt.model === primary
          ? streamOf([
              {
                type: 'abort',
                reason: 'TimeoutError: Step timeout of 200ms exceeded',
              },
            ])
          : fallbackResult,
      );

      // Assert — the abort reason became a TimeoutError, so timeout() failed over.
      expect(committed).toBe(fallbackResult);
      expect(seen.length).toBe(1);
      expect(seen[0]!.name).toBe('TimeoutError');
      expect(seen[0]!.message).toBe('Step timeout of 200ms exceeded');
    });

    it('should reconstruct an AbortError from an abort reason so aborted() matches', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fallbackResult = streamOf([{ type: 'text-delta', text: 'OK' }]);
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [aborted().switch({ model: fallback })],
      });

      // Act
      const committed = await retryableStream((attempt) =>
        attempt.model === primary
          ? streamOf([
              {
                type: 'abort',
                reason: 'AbortError: This operation was aborted',
              },
            ])
          : fallbackResult,
      );

      // Assert
      expect(committed).toBe(fallbackResult);
    });

    it('should match an abort reason by message', async () => {
      // Arrange — a message-based condition matches the reconstructed message.
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fallbackResult = streamOf([{ type: 'text-delta', text: 'OK' }]);
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [error.message('Step timeout').switch({ model: fallback })],
      });

      // Act
      const committed = await retryableStream((attempt) =>
        attempt.model === primary
          ? streamOf([
              {
                type: 'abort',
                reason: 'TimeoutError: Step timeout of 200ms exceeded',
              },
            ])
          : fallbackResult,
      );

      // Assert
      expect(committed).toBe(fallbackResult);
    });

    it('should surface an unmatchable error for an abort part with no reason', async () => {
      // Arrange — no attempt-signal reason and no part reason: nothing to match.
      const retryableStream = createRetryableStream({
        model: MockLanguageModel.from(),
        retries: [timeout().switch({ model: MockLanguageModel.from() })],
      });

      // Act
      const result = retryableStream(() => streamOf([{ type: 'abort' }]));

      // Assert — timeout() cannot match a bare abort, so no fail-over.
      await expect(result).rejects.toThrow();
    });
  });

  describe('RetryError', () => {
    it('should throw a RetryError after all attempts are exhausted', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [fallback],
      });

      // Act
      const result = retryableStream(() =>
        streamOf([Language.streamError(new Error('boom'))]),
      );

      // Assert
      await expect(result).rejects.toThrow();
      await result.catch((e) => expect(RetryError.isInstance(e)).toBe(true));
    });
  });

  describe('outcome hooks', () => {
    it('should call onCommit with the model that committed', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fallbackResult = streamOf([{ type: 'text-delta', text: 'OK' }]);
      const onCommit = vi.fn();
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [fallback],
        onCommit,
      });

      // Act
      const committed = await retryableStream((attempt) =>
        attempt.model === primary
          ? streamOf([Language.streamError(new Error('boom'))])
          : fallbackResult,
      );

      // Assert — the result reaches the caller, not the hook.
      expect(committed).toBe(fallbackResult);
      expect(onCommit).toHaveBeenCalledTimes(1);
      expect(onCommit.mock.calls[0]![0].current.model).toBe(fallback);
      expect(onCommit.mock.calls[0]![0].attempts.length).toBe(1);
    });

    it('should call onCommit at the commit point, not at stream completion', async () => {
      // Arrange — content, then an error the caller inherits past the commit.
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const onCommit = vi.fn();
      const onFailure = vi.fn();
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [fallback],
        onCommit,
        onFailure,
      });

      // Act
      await retryableStream(() =>
        streamOf([
          { type: 'text-delta', text: 'OK' },
          Language.streamError(new Error('mid-stream')),
        ]),
      );

      // Assert — the post-commit error is the caller's, so it fires nothing.
      expect(onCommit).toHaveBeenCalledTimes(1);
      expect(onCommit.mock.calls[0]![0].current.model).toBe(primary);
      expect(onFailure).toHaveBeenCalledTimes(0);
    });

    it('should call onFailure when every attempt fails before content', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const onCommit = vi.fn();
      const onFailure = vi.fn();
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [fallback],
        onCommit,
        onFailure,
      });

      // Act
      await retryableStream(() =>
        streamOf([Language.streamError(new Error('boom'))]),
      ).catch(() => {});

      // Assert
      expect(onCommit).toHaveBeenCalledTimes(0);
      expect(onFailure).toHaveBeenCalledTimes(1);
      expect(RetryError.isInstance(onFailure.mock.calls[0]![0].error)).toBe(
        true,
      );
      expect(onFailure.mock.calls[0]![0].current.model).toBe(fallback);
      expect(onFailure.mock.calls[0]![0].attempts.length).toBe(2);
    });

    it('should call neither hook when retries are disabled', async () => {
      // Arrange
      const onCommit = vi.fn();
      const onFailure = vi.fn();
      const retryableStream = createRetryableStream({
        model: MockLanguageModel.from(),
        retries: [],
        disabled: true,
        onCommit,
        onFailure,
      });

      // Act
      await retryableStream(() =>
        streamOf([{ type: 'text-delta', text: 'OK' }]),
      );

      // Assert
      expect(onCommit).toHaveBeenCalledTimes(0);
      expect(onFailure).toHaveBeenCalledTimes(0);
    });
  });

  describe('disabled', () => {
    it('should bypass retries when disabled', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const boom = new Error('boom');
      const models: Array<unknown> = [];
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [fallback],
        disabled: true,
      });

      // Act
      const result = retryableStream((attempt) => {
        models.push(attempt.model);
        return streamOf([Language.streamError(boom)]);
      });

      // Assert
      await expect(result).rejects.toThrow();
      await result.catch((e) => expect(e).toBe(boom));
      expect(models.length).toBe(1);
    });
  });
});

/**
 * Integration suite: drive `createRetryableStream` over a real `streamText`
 * call per attempt, so commit/fail-over is decided from the actual SDK stream.
 */
describe('streamText integration', () => {
  it('should return a usable result when the first attempt succeeds', async () => {
    // Arrange
    const primary = okStreamModel();

    // Act
    const result = await retryableStreamText(
      { model: primary, retries: [] },
      { prompt },
    );

    // Assert
    expect(await result.text).toBe('Hello, world!');
    expect(primary.doStream).toHaveBeenCalledTimes(1);
  });

  describe('error-based retries', () => {
    it('should fall back when stream creation fails', async () => {
      // Arrange
      const primary = MockLanguageModel.from({
        doStream: new Error('creation failed'),
      });
      const fallback = okStreamModel();

      // Act
      const result = await retryableStreamText(
        { model: primary, retries: [fallback] },
        { prompt },
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
        { prompt },
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
        { prompt },
      );

      // Assert
      expect(await result.text).toBe('Hello, world!');
      expect(primary.doStream).toHaveBeenCalledTimes(1);
      expect(second.doStream).toHaveBeenCalledTimes(1);
      expect(third.doStream).toHaveBeenCalledTimes(1);
    });

    it('should fall back on a content-filter error part', async () => {
      // Arrange — content-filter surfaces as an error (not a finish) here.
      const primary = MockLanguageModel.from({ doStream: contentFilterError });
      const fallback = okStreamModel();

      // Act
      const result = await retryableStreamText(
        {
          model: primary,
          retries: [
            error.message('content management policy').switch({
              model: fallback,
            }),
          ],
        },
        { prompt },
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
        { prompt },
      );

      let text = '';
      try {
        for await (const part of result.stream) {
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
        { prompt },
      );

      // Assert
      await expect(result).rejects.toThrow();
      expect(primary.doStream).toHaveBeenCalledTimes(1);
    });

    it('should throw a RetryError after all attempts are exhausted', async () => {
      // Arrange
      const primary = MockLanguageModel.from({ doStream: new Error('first') });
      const fallback = MockLanguageModel.from({
        doStream: new Error('second'),
      });

      // Act
      const result = retryableStreamText(
        { model: primary, retries: [fallback] },
        { prompt },
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
        { prompt },
      );

      // Assert
      expect(onError).toHaveBeenCalledTimes(1);
      expect(onRetry).toHaveBeenCalledTimes(1);
    });

    it('should call onCommit with the model that recovered the stream', async () => {
      // Arrange
      const primary = errorAtStartStreamModel(new Error('boom'));
      const fallback = okStreamModel();
      const onCommit = vi.fn();
      const onFailure = vi.fn();

      // Act
      const result = await retryableStreamText(
        { model: primary, retries: [fallback], onCommit, onFailure },
        { prompt },
      );
      await result.text;

      // Assert
      expect(onCommit).toHaveBeenCalledTimes(1);
      expect(onCommit.mock.calls[0]![0].current.model).toBe(fallback);
      expect(onCommit.mock.calls[0]![0].attempts.length).toBe(1);
      expect(onFailure).toHaveBeenCalledTimes(0);
    });

    it('should call onFailure when no retryable matches a pre-content error', async () => {
      // Arrange
      const primary = errorAtStartStreamModel(new Error('boom'));
      const onCommit = vi.fn();
      const onFailure = vi.fn();

      // Act
      await retryableStreamText(
        { model: primary, retries: [], onCommit, onFailure },
        { prompt },
      ).catch(() => {});

      // Assert
      expect(onCommit).toHaveBeenCalledTimes(0);
      expect(onFailure).toHaveBeenCalledTimes(1);
      expect(onFailure.mock.calls[0]![0].current.model).toBe(primary);
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
        {
          model: primary,
          retries: [finishReason('content-filter').switch({ model: fallback })],
        },
        { prompt },
      );

      // Assert — no fail-over, no side effects.
      expect(await result.finishReason).toBe('content-filter');
      expect(primary.doStream).toHaveBeenCalledTimes(1);
      expect(fallback.doStream).toHaveBeenCalledTimes(0);
    });
  });

  describe('timeouts', () => {
    /**
     * `firstChunkMs`, `stepMs`, and `totalMs` all start counting from the step
     * (or call) start, before any content, so a stall before the first content
     * part trips them pre-commit and the call layer fails over. `chunkMs` is the
     * exception — it measures the gap *between* content chunks, so it never fires
     * without content and only truncates once the attempt has committed.
     */
    describe('pre-content deadlines fail over', () => {
      it('should recover a timeout.firstChunkMs deadline', async () => {
        // Arrange — no first content chunk within firstChunkMs of the step start.
        const primary = stallStreamModel();
        const fallback = okStreamModel();

        // Act
        const result = await retryableStreamText(
          { model: primary, retries: [fallback] },
          { prompt, timeout: { firstChunkMs: 50 } },
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
          { prompt, timeout: { stepMs: 50 } },
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
          { prompt, timeout: { totalMs: 50 } },
        );

        // Assert
        expect(await result.text).toBe('Hello, world!');
        expect(fallback.doStream).toHaveBeenCalledTimes(1);
      });
    });

    it('should NOT recover a timeout.chunkMs deadline that fires after content', async () => {
      // Arrange — chunkMs measures the gap between content chunks, so it only
      // fires once a content part has already committed the attempt.
      const primary = partialThenStallStreamModel();
      const fallback = okStreamModel();

      // Act
      const result = await retryableStreamText(
        { model: primary, retries: [fallback] },
        { prompt, timeout: { chunkMs: 50 } },
      );

      // Drain tolerantly: the post-content deadline surfaces an abort.
      let text = '';
      try {
        for await (const part of result.stream) {
          if (part.type === 'text-delta') text += part.text ?? '';
        }
      } catch {
        /* deadline abort after content */
      }

      // Assert — committed on the first delta, so no fail-over.
      expect(text).toBe('partial');
      expect(fallback.doStream).toHaveBeenCalledTimes(0);
    });

    it('should NOT recover a timeout.toolMs deadline that fires during tool execution', async () => {
      // Arrange — the model emits a tool-call (a content part that commits the
      // attempt), then the hanging tool's execution trips toolMs. The abort
      // lands after the commit point, so the call layer never fails over.
      const primary = MockLanguageModel.from({
        doStream: [
          Language.streamStart(),
          Language.toolCall({
            toolCallId: 'c1',
            toolName: 'wait',
            input: '{}',
          }),
          Language.streamFinish({ finishReason: 'tool-calls' }),
        ],
      });
      const fallback = okStreamModel();
      const wait = tool({
        description: 'waits until aborted',
        inputSchema: z.object({}),
        execute: (_input, { abortSignal }) =>
          new Promise((_resolve, reject) => {
            abortSignal?.addEventListener(
              'abort',
              () => reject(abortSignal.reason),
              { once: true },
            );
          }),
      });

      // Act
      const result = await retryableStreamText(
        { model: primary, retries: [fallback] },
        {
          prompt,
          tools: { wait },
          timeout: { toolMs: 50 },
          stopWhen: stepCountIs(1),
        },
      );

      // Drain tolerantly: the tool-execution timeout surfaces a tool-error.
      const types: Array<string> = [];
      try {
        for await (const part of result.stream) types.push(part.type);
      } catch {
        /* tool timeout after the committing tool-call */
      }

      // Assert — committed on the tool-call, so no fail-over.
      expect(types).toContain('tool-call');
      expect(fallback.doStream).toHaveBeenCalledTimes(0);
    });

    it('should NOT recover an inbound abortSignal deadline (a hard caller cancel)', async () => {
      // Arrange — an inbound abortSignal is the caller's own deadline, not a
      // per-attempt timeout: once it fires the whole call is cancelled and does
      // not fail over, even to a retry with its own timeout. For a per-attempt
      // deadline, use streamText's `timeout` instead (see the test above).
      const primary = stallStreamModel();
      const fallback = okStreamModel();

      // Act
      const result = retryableStreamText(
        { model: primary, retries: [{ model: fallback, timeout: 5_000 }] },
        { prompt, abortSignal: AbortSignal.timeout(50) },
      );

      // Assert
      await expect(result).rejects.toThrow();
      expect(fallback.doStream).toHaveBeenCalledTimes(0);
    });

    it('should give each attempt a fresh deadline signal', async () => {
      // Arrange
      const signals: Array<AbortSignal | undefined> = [];
      const primary = MockLanguageModel.from({
        doStream: async ({ abortSignal }: LanguageModelCallOptions) => {
          signals.push(abortSignal);
          return {
            stream: new ReadableStream<LanguageModelStreamPart>({
              start(controller) {
                controller.enqueue(Language.streamStart());
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
      const fallback = MockLanguageModel.from({
        doStream: async ({ abortSignal }: LanguageModelCallOptions) => {
          signals.push(abortSignal);
          return { stream: Streams.from(mockStreamChunks) };
        },
      });

      // Act
      const result = await retryableStreamText(
        { model: primary, retries: [fallback] },
        { prompt, timeout: { firstChunkMs: 50 } },
      );
      await result.text;

      // Assert
      expect(signals.length).toBe(2);
      expect(signals[0]).not.toBe(signals[1]);
      expect(signals[0]!.aborted).toBe(true);
      expect(signals[1]!.aborted).toBe(false);
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
        { prompt, abortSignal: controller.signal },
      );

      // Assert
      await expect(result).rejects.toThrow();
      expect(fallback.doStream).toHaveBeenCalledTimes(0);
    });
  });

  describe('disabled', () => {
    it('should bypass retries when disabled', async () => {
      // Arrange
      const primary = MockLanguageModel.from({ doStream: new Error('boom') });
      const fallback = okStreamModel();

      // Act
      const result = retryableStreamText(
        { model: primary, retries: [fallback], disabled: true },
        { prompt },
      );

      // Assert
      await expect(result).rejects.toThrow();
      expect(primary.doStream).toHaveBeenCalledTimes(1);
      expect(fallback.doStream).toHaveBeenCalledTimes(0);
    });
  });

  describe('deferred consumption', () => {
    it('should let the caller drive the body via a UI message stream response', async () => {
      // Arrange — fail over before content, then let the caller consume.
      const primary = MockLanguageModel.from({ doStream: new Error('boom') });
      const fallback = okStreamModel();

      // Act
      const result = await retryableStreamText(
        { model: primary, retries: [fallback] },
        { prompt },
      );
      const response = createUIMessageStreamResponse({
        stream: toUIMessageStream({ stream: result.stream }),
      });
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
        { prompt, onChunk, onFinish },
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
        { prompt, onError },
      );
      await result.text;

      // Assert — committed on the first delta, so the error reaches the caller.
      expect(onError).toHaveBeenCalledTimes(1);
    });
  });

  describe('composition with a retryable base model', () => {
    it('should recover a content-filter finish at the model layer', async () => {
      // Arrange — the inner createRetryableModel handles the content-filter finish
      // BELOW streamText; the outer call layer never fails over.
      const primary = contentFilterFinishModel();
      const modelFallback = okStreamModel();
      const callFallback = okStreamModel();
      const inner = createRetryableModel({
        model: primary,
        retries: [
          finishReason('content-filter').switch({ model: modelFallback }),
        ],
      });

      // Act
      const result = await retryableStreamText(
        { model: inner, retries: [callFallback] },
        { prompt },
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
      const inner = createRetryableModel({
        model: primary,
        retries: [modelFallback],
      });

      // Act
      const result = await retryableStreamText(
        { model: inner, retries: [callFallback] },
        { prompt, timeout: { totalMs: 50 } },
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
          model: createRetryableModel({ model: primary, retries: [fallback] }),
          prompt,
          maxRetries: 0,
          timeout: { totalMs: 50 },
          onError: () => {},
        });

        // Act — bound the drain: the aborted stream may never cleanly settle,
        // which is itself a symptom of the discarded fallback (see issue #50).
        let text = '';
        const drain = (async () => {
          for await (const part of result.stream) {
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
          {
            model: primary,
            retries: [
              finishReason('content-filter').switch({ model: fallback }),
            ],
          },
          { prompt },
        );

        // Assert
        expect(await result.finishReason).toBe('content-filter');
        expect(fallback.doStream).toHaveBeenCalledTimes(0);
      });
    });
  });
});
