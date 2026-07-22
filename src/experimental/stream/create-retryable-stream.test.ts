import { RetryError, streamText } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { createRetryable } from '../../index.js';
import {
  aborted,
  error,
  timeout,
} from '../../language-model/conditions/index.js';
import {
  contentFilterError,
  contentFilterStreamChunks,
  errorStreamChunks,
  Language,
  MockLanguageModel,
  mockStreamChunks,
  Streams,
  successStreamChunks,
} from '../../internal/test-utils.js';
import { contentFilterTriggered } from '../../retryables/content-filter-triggered.js';
import type {
  LanguageModelCallOptions,
  LanguageModelStreamPart,
} from '../../types.js';
import {
  createRetryableStream,
  type RetryableStreamOptions,
} from './create-retryable-stream.js';
import { RefusalError, refusalGate } from './refusal-gate.js';

const prompt = 'Hello!';
const REFUSAL = "I'm sorry, but I cannot assist";

/**
 * Result shapes carry `streamText`-level parts (`TextStreamPart`s: `text-delta`
 * with `.text`, plus `abort`/`error`), written as literals since ai-test-kit's
 * `Language.*` helpers build the provider-level parts a mock model's `doStream`
 * returns, one layer below. `Streams.from` wraps any array into a `ReadableStream`.
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
 * A model that streams a natural-language refusal and finishes with `stop` —
 * no error, no `content-filter` finish reason, so only a text-buffering gate at
 * the call layer can tell it apart from a real answer.
 */
const refusalStreamModel = (
  text = "I'm sorry, but I cannot assist with that request.",
) => MockLanguageModel.from({ doStream: successStreamChunks(text) });

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
          ? streamOf([{ type: 'error', error: new Error('x') }])
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
          ? fullStreamOf([{ type: 'error', error: new Error('x') }])
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
              stream: Streams.from([{ type: 'error', error: new Error('x') }]),
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
        { type: 'error', error: new Error('mid-stream') },
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
          ? streamOf([{ type: 'error', error: new Error('boom') }])
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

  describe('commit gate', () => {
    it('should fail over when the buffered text matches a refusal phrase', async () => {
      // Arrange — a refusal split across deltas, with finishReason `stop` (no
      // error/finish signal): the gate must buffer and detect it from the text.
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fallbackResult = streamOf([{ type: 'text-delta', text: 'answer' }]);
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [
          error((e) => e instanceof RefusalError).switch({ model: fallback }),
        ],
        commitGate: refusalGate([REFUSAL]),
      });

      // Act
      const committed = await retryableStream((attempt) =>
        attempt.model === primary
          ? streamOf([
              { type: 'text-delta', text: "I'm sorry, " },
              { type: 'text-delta', text: 'but I cannot assist' },
              { type: 'text-delta', text: ' with that.' },
            ])
          : fallbackResult,
      );

      // Assert
      expect(committed).toBe(fallbackResult);
    });

    it('should commit a real answer that shares a leading fragment', async () => {
      // Arrange — "I'm sorry to hear" diverges from the refusal at "to" vs "but".
      const fallback = MockLanguageModel.from();
      const result = streamOf([
        { type: 'text-delta', text: "I'm sorry " },
        { type: 'text-delta', text: 'to hear that! Here is how.' },
      ]);
      const retryableStream = createRetryableStream({
        model: MockLanguageModel.from(),
        retries: [fallback],
        commitGate: refusalGate([REFUSAL]),
      });

      // Act
      const committed = await retryableStream(() => result);

      // Assert — committed without a false fail-over.
      expect(committed).toBe(result);
      expect(fallback.doStream).toHaveBeenCalledTimes(0);
    });

    it('should commit when the stream ends on an inconclusive prefix', async () => {
      // Arrange — text stops while still a prefix of the phrase; never resolved
      // as a refusal, so it must commit rather than hang or fail over.
      const result = streamOf([{ type: 'text-delta', text: "I'm sorry" }]);
      const retryableStream = createRetryableStream({
        model: MockLanguageModel.from(),
        retries: [],
        commitGate: refusalGate([REFUSAL]),
      });

      // Act
      const committed = await retryableStream(() => result);

      // Assert
      expect(committed).toBe(result);
    });

    it('should ignore the gate for non-text content parts', async () => {
      // Arrange — a tool-call is not text, so it commits immediately even with
      // an active gate.
      const result = streamOf([{ type: 'tool-call', toolName: 'search' }]);
      const retryableStream = createRetryableStream({
        model: MockLanguageModel.from(),
        retries: [],
        commitGate: refusalGate([REFUSAL]),
      });

      // Act
      const committed = await retryableStream(() => result);

      // Assert
      expect(committed).toBe(result);
    });

    it('should fail over on a custom onRefusal error the conditions match', async () => {
      // Arrange — the gate throws a caller-supplied error, matched by message.
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fallbackResult = streamOf([{ type: 'text-delta', text: 'OK' }]);
      const retryableStream = createRetryableStream({
        model: primary,
        retries: [
          error.message('blocked by policy').switch({ model: fallback }),
        ],
        commitGate: refusalGate([REFUSAL], {
          onRefusal: () => new Error('blocked by policy'),
        }),
      });

      // Act
      const committed = await retryableStream((attempt) =>
        attempt.model === primary
          ? streamOf([{ type: 'text-delta', text: REFUSAL }])
          : fallbackResult,
      );

      // Assert
      expect(committed).toBe(fallbackResult);
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
        return streamOf([{ type: 'error', error: boom }]);
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
        { model: primary, retries: [contentFilterTriggered(fallback)] },
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
        { prompt },
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
        { prompt, timeout: { chunkMs: 50 } },
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

    it('should recover an inbound abortSignal deadline with a per-attempt timeout', async () => {
      // Arrange
      const primary = stallStreamModel();
      const fallback = okStreamModel();

      // Act
      const result = await retryableStreamText(
        { model: primary, retries: [{ model: fallback, timeout: 5_000 }] },
        { prompt, abortSignal: AbortSignal.timeout(50) },
      );

      // Assert
      expect(await result.text).toBe('Hello, world!');
      expect(fallback.doStream).toHaveBeenCalledTimes(1);
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
        { prompt, timeout: { chunkMs: 50 } },
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

  describe('commit gate', () => {
    it('should fail over from a canned refusal streamed by the model', async () => {
      // Arrange — the primary streams a natural-language refusal (finishReason
      // `stop`); only the gate can catch it, then the call layer fails over.
      const primary = refusalStreamModel();
      const fallback = okStreamModel();

      // Act
      const result = await retryableStreamText(
        {
          model: primary,
          retries: [
            error((e) => e instanceof RefusalError).switch({ model: fallback }),
          ],
          commitGate: refusalGate([REFUSAL]),
        },
        { prompt },
      );

      // Assert
      expect(await result.text).toBe('Hello, world!');
      expect(primary.doStream).toHaveBeenCalledTimes(1);
      expect(fallback.doStream).toHaveBeenCalledTimes(1);
    });

    it('should commit a real answer that shares a leading fragment', async () => {
      // Arrange — a genuine answer that opens like a refusal must not fail over.
      const answer = "I'm sorry to hear that. Here is what to do.";
      const primary = refusalStreamModel(answer);
      const fallback = okStreamModel();

      // Act
      const result = await retryableStreamText(
        {
          model: primary,
          retries: [
            error((e) => e instanceof RefusalError).switch({ model: fallback }),
          ],
          commitGate: refusalGate([REFUSAL]),
        },
        { prompt },
      );

      // Assert
      expect(await result.text).toBe(answer);
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
    it('should let the caller drive the body via toUIMessageStreamResponse', async () => {
      // Arrange — fail over before content, then let the caller consume.
      const primary = MockLanguageModel.from({ doStream: new Error('boom') });
      const fallback = okStreamModel();

      // Act
      const result = await retryableStreamText(
        { model: primary, retries: [fallback] },
        { prompt },
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
      const inner = createRetryable({
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

    describe('with a commit gate', () => {
      it('should not block the inner model-layer recovery of a clean answer', async () => {
        // Arrange — the inner retryable recovers a content-filter finish BELOW
        // streamText; the outer gate then sees only the clean fallback text,
        // which diverges from the refusal phrase and commits. They compose.
        const primary = contentFilterFinishModel();
        const modelFallback = okStreamModel();
        const callFallback = okStreamModel();
        const inner = createRetryable({
          model: primary,
          retries: [contentFilterTriggered(modelFallback)],
        });

        // Act
        const result = await retryableStreamText(
          {
            model: inner,
            retries: [callFallback],
            commitGate: refusalGate([REFUSAL]),
          },
          { prompt },
        );

        // Assert — recovered at the model layer; the gate did not block it.
        expect(await result.text).toBe('Hello, world!');
        expect(modelFallback.doStream).toHaveBeenCalledTimes(1);
        expect(callFallback.doStream).toHaveBeenCalledTimes(0);
      });

      it('should fail over at the call layer on a refusal the model layer ignores', async () => {
        // Arrange — the primary streams a natural-language refusal (finishReason
        // `stop`), which the inner content-filter retryable does NOT match, so
        // it passes through. The outer gate catches it and fails over. Proves
        // the retryable base does not swallow or block the call-layer fail-over.
        const primary = refusalStreamModel();
        const modelFallback = okStreamModel();
        const callFallback = okStreamModel();
        const inner = createRetryable({
          model: primary,
          retries: [contentFilterTriggered(modelFallback)],
        });

        // Act
        const result = await retryableStreamText(
          {
            model: inner,
            retries: [
              error((e) => e instanceof RefusalError).switch({
                model: callFallback,
              }),
            ],
            commitGate: refusalGate([REFUSAL]),
          },
          { prompt },
        );

        // Assert — the inner layer ignored the refusal; the outer recovered.
        expect(await result.text).toBe('Hello, world!');
        expect(primary.doStream).toHaveBeenCalledTimes(1);
        expect(modelFallback.doStream).toHaveBeenCalledTimes(0);
        expect(callFallback.doStream).toHaveBeenCalledTimes(1);
      });
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
          { model: primary, retries: [contentFilterTriggered(fallback)] },
          { prompt },
        );

        // Assert
        expect(await result.finishReason).toBe('content-filter');
        expect(fallback.doStream).toHaveBeenCalledTimes(0);
      });
    });
  });
});
