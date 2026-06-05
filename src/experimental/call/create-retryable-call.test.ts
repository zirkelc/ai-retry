import { RetryError } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { MockLanguageModel } from '../../internal/test-utils.js';
import { requestTimeout } from '../../retryables/request-timeout.js';
import type {
  LanguageModel,
  LanguageModelResult,
  Retryable,
} from '../../types.js';
import {
  createRetryableCall,
  type ResultRetry,
  type RetryCallAttempt,
} from './create-retryable-call.js';

/**
 * Rejects with the signal's abort reason once it aborts. Simulates a call that
 * stalls until its per-attempt deadline fires.
 */
const stallUntilAbort = (signal: AbortSignal | undefined): Promise<never> =>
  new Promise((_resolve, reject) => {
    if (signal?.aborted) reject(signal.reason);
    else
      signal?.addEventListener('abort', () => reject(signal.reason), {
        once: true,
      });
  });

/** A minimal synthetic result a stream call would flag for a result retry. */
const filteredResult = {
  content: [],
  finishReason: { unified: 'content-filter', raw: undefined },
} as unknown as LanguageModelResult;

/** Build a result-retry signal a call function throws to request re-evaluation. */
const resultRetry = <V>(value: V): ResultRetry<V> => ({
  type: 'result',
  result: filteredResult,
  value,
});

/** A function retryable that matches any result attempt. */
const retryOnResult =
  (model: LanguageModel): Retryable<LanguageModel> =>
  (context) =>
    context.current.type === 'result' ? { model } : undefined;

describe('createRetryableCall', () => {
  it('should return the result of the first attempt on success', async () => {
    // Arrange
    const primary = new MockLanguageModel();
    const fallback = new MockLanguageModel();
    const fn = vi.fn(async (_attempt: RetryCallAttempt) => 'OK');
    const run = createRetryableCall({ model: primary, retries: [fallback] });

    // Act
    const result = await run(fn);

    // Assert
    expect(result).toBe('OK');
    expect(fn.mock.calls.length).toBe(1);
    expect(fn.mock.calls[0]![0].model).toBe(primary);
    expect(fn.mock.calls[0]![0].attempt).toBe(1);
  });

  it('should fall back to the next model after an error', async () => {
    // Arrange
    const primary = new MockLanguageModel();
    const fallback = new MockLanguageModel();
    const fn = vi.fn(async ({ model }: RetryCallAttempt) => {
      if (model === primary) throw new Error('primary failed');
      return 'FALLBACK_OK';
    });
    const run = createRetryableCall({ model: primary, retries: [fallback] });

    // Act
    const result = await run(fn);

    // Assert
    expect(result).toBe('FALLBACK_OK');
    expect(fn.mock.calls.length).toBe(2);
    expect(fn.mock.calls[0]![0].model).toBe(primary);
    expect(fn.mock.calls[1]![0].model).toBe(fallback);
    expect(fn.mock.calls[1]![0].attempt).toBe(2);
  });

  it('should give each attempt a fresh deadline so a stalled attempt recovers', async () => {
    // Arrange
    const primary = new MockLanguageModel();
    const fallback = new MockLanguageModel();
    const signals: Array<AbortSignal | undefined> = [];
    const fn = vi.fn(async ({ model, abortSignal }: RetryCallAttempt) => {
      signals.push(abortSignal);
      if (model === primary) return stallUntilAbort(abortSignal);
      return 'FALLBACK_OK';
    });
    const run = createRetryableCall({
      model: primary,
      retries: [requestTimeout(fallback, { timeout: 1_000 })],
    });

    // Act
    const result = await run(fn, { timeout: 30 });

    // Assert
    expect(result).toBe('FALLBACK_OK');
    expect(signals.length).toBe(2);
    expect(signals[0]).toBeInstanceOf(AbortSignal);
    expect(signals[1]).toBeInstanceOf(AbortSignal);
    // Each attempt gets its own signal, not the spent one.
    expect(signals[0]).not.toBe(signals[1]);
    expect(signals[0]!.aborted).toBe(true);
    expect(signals[1]!.aborted).toBe(false);
  });

  it('should not retry when the inbound signal is already aborted and the retry has no fresh timeout', async () => {
    // Arrange
    const primary = new MockLanguageModel();
    const fallback = new MockLanguageModel();
    const controller = new AbortController();
    controller.abort(new Error('user cancelled'));
    const error = new Error('primary failed');
    const fn = vi.fn(async () => {
      throw error;
    });
    const run = createRetryableCall({ model: primary, retries: [fallback] });

    // Act
    const result = run(fn, { abortSignal: controller.signal });

    // Assert
    await expect(result).rejects.toThrow();
    await result.catch((e) => expect(e).toBe(error));
    expect(fn.mock.calls.length).toBe(1);
  });

  it('should throw a RetryError after all attempts are exhausted', async () => {
    // Arrange
    const primary = new MockLanguageModel();
    const fallback = new MockLanguageModel();
    const fn = vi.fn(async () => {
      throw new Error('always fails');
    });
    const run = createRetryableCall({ model: primary, retries: [fallback] });

    // Act
    const result = run(fn);

    // Assert
    await expect(result).rejects.toThrow();
    await result.catch((e) => expect(RetryError.isInstance(e)).toBe(true));
    expect(fn.mock.calls.length).toBe(2);
  });

  it('should apply onRetry option overrides to the next attempt', async () => {
    // Arrange
    const primary = new MockLanguageModel();
    const fallback = new MockLanguageModel();
    const fn = vi.fn(async ({ model }: RetryCallAttempt) => {
      if (model === primary) throw new Error('primary failed');
      return 'OK';
    });
    const run = createRetryableCall({
      model: primary,
      retries: [fallback],
      onRetry: () => ({ options: { temperature: 0.5 } }),
    });

    // Act
    await run(fn);

    // Assert
    expect(fn.mock.calls[1]![0].options.temperature).toBe(0.5);
  });

  it('should bypass retries entirely when disabled', async () => {
    // Arrange
    const primary = new MockLanguageModel();
    const fallback = new MockLanguageModel();
    const error = new Error('primary failed');
    const fn = vi.fn(async (_attempt: RetryCallAttempt) => {
      throw error;
    });
    const run = createRetryableCall({
      model: primary,
      retries: [fallback],
      disabled: true,
    });

    // Act
    const result = run(fn);

    // Assert
    await expect(result).rejects.toThrow();
    await result.catch((e) => expect(e).toBe(error));
    expect(fn.mock.calls.length).toBe(1);
    expect(fn.mock.calls[0]![0].model).toBe(primary);
  });

  it('should fail over on a ResultRetry when a function retryable matches', async () => {
    // Arrange
    const primary = new MockLanguageModel();
    const fallback = new MockLanguageModel();
    const fn = vi.fn(async ({ model }: RetryCallAttempt) => {
      if (model === primary) throw resultRetry('FILTERED');
      return 'FALLBACK_OK';
    });
    const run = createRetryableCall({
      model: primary,
      retries: [retryOnResult(fallback)],
    });

    // Act
    const result = await run(fn);

    // Assert
    expect(result).toBe('FALLBACK_OK');
    expect(fn.mock.calls.length).toBe(2);
    expect(fn.mock.calls[1]![0].model).toBe(fallback);
  });

  it('should return the ResultRetry value terminally when no retryable matches', async () => {
    // Arrange — a plain fallback model is skipped for result attempts.
    const primary = new MockLanguageModel();
    const fallback = new MockLanguageModel();
    const fn = vi.fn(async (_attempt: RetryCallAttempt) => {
      throw resultRetry('TERMINAL_RESULT');
    });
    const run = createRetryableCall({ model: primary, retries: [fallback] });

    // Act
    const result = await run(fn);

    // Assert — resolves to the flagged value, no fail-over.
    expect(result).toBe('TERMINAL_RESULT');
    expect(fn.mock.calls.length).toBe(1);
  });

  it('should return the ResultRetry value when retries are disabled', async () => {
    // Arrange
    const primary = new MockLanguageModel();
    const fallback = new MockLanguageModel();
    const fn = vi.fn(async (_attempt: RetryCallAttempt) => {
      throw resultRetry('TERMINAL_RESULT');
    });
    const run = createRetryableCall({
      model: primary,
      retries: [retryOnResult(fallback)],
      disabled: true,
    });

    // Act
    const result = await run(fn);

    // Assert
    expect(result).toBe('TERMINAL_RESULT');
    expect(fn.mock.calls.length).toBe(1);
  });

  it('should stick to the recovered model on the next run per the reset policy', async () => {
    // Arrange
    const primary = new MockLanguageModel();
    const fallback = new MockLanguageModel();
    const fn = vi.fn(async ({ model }: RetryCallAttempt) => {
      if (model === primary) throw new Error('primary failed');
      return 'OK';
    });
    const run = createRetryableCall({
      model: primary,
      retries: [fallback],
      reset: 'after-2-requests',
    });

    // Act
    await run(fn); // recovers on fallback, becomes sticky
    const secondRunModels: Array<MockLanguageModel> = [];
    await run(async ({ model }) => {
      secondRunModels.push(model as MockLanguageModel);
      return 'OK';
    });

    // Assert
    expect(secondRunModels.length).toBe(1);
    expect(secondRunModels[0]!).toBe(fallback);
  });
});
