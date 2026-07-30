import { RetryError } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { MockLanguageModel } from '../../internal/test-utils.js';
import { requestTimeout } from '../../retryables/request-timeout.js';
import {
  createRetryableCall,
  type RetryCallAttempt,
} from './create-retryable-call.js';

/** A call function that succeeds on any model except the given failing ones. */
const failOn = (
  failing: ReadonlyArray<MockLanguageModel>,
  result = 'OK',
  error: () => unknown = () => new Error('attempt failed'),
) =>
  vi.fn(async ({ model }: RetryCallAttempt) => {
    if (failing.includes(model as MockLanguageModel)) throw error();
    return result;
  });

describe('createRetryableCall', () => {
  describe('success', () => {
    it('should return the result of the first attempt', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fn = vi.fn(async (_attempt: RetryCallAttempt) => 'OK');
      const run = createRetryableCall({ model: primary, retries: [] });

      // Act
      const result = await run(fn);

      // Assert
      expect(result).toBe('OK');
      expect(fn).toHaveBeenCalledTimes(1);
    });

    it('should pass the result through unchanged (opaque to the driver)', async () => {
      // Arrange — the driver never inspects the result; a returned value is
      // terminal even if it looks like a failure.
      const primary = MockLanguageModel.from();
      const result = { ok: false };
      const run = createRetryableCall({ model: primary, retries: [] });

      // Act
      const returned = await run(async () => result);

      // Assert
      expect(returned).toBe(result);
    });
  });

  describe('retries', () => {
    it('should fall back to the next model after an error', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fn = failOn([primary], 'FALLBACK_OK');
      const run = createRetryableCall({ model: primary, retries: [fallback] });

      // Act
      const result = await run(fn);

      // Assert
      expect(result).toBe('FALLBACK_OK');
      expect(fn).toHaveBeenCalledTimes(2);
      expect(fn.mock.calls[1]![0].model).toBe(fallback);
    });

    it('should fall back across consecutive errors', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const second = MockLanguageModel.from();
      const third = MockLanguageModel.from();
      const fn = failOn([primary, second], 'THIRD_OK');
      const run = createRetryableCall({
        model: primary,
        retries: [second, third],
      });

      // Act
      const result = await run(fn);

      // Assert
      expect(result).toBe('THIRD_OK');
      expect(fn).toHaveBeenCalledTimes(3);
      expect(fn.mock.calls[2]![0].model).toBe(third);
    });

    it('should select a retry via a function retryable', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fn = failOn([primary]);
      const run = createRetryableCall({
        model: primary,
        retries: [() => ({ model: fallback, maxAttempts: 1 })],
      });

      // Act
      const result = await run(fn);

      // Assert
      expect(result).toBe('OK');
      expect(fn.mock.calls[1]![0].model).toBe(fallback);
    });
  });

  describe('disabled', () => {
    it('should bypass retries when disabled is true', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
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
      expect(fn).toHaveBeenCalledTimes(1);
      expect(fn.mock.calls[0]![0].model).toBe(primary);
    });

    it('should retry when disabled is false', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fn = failOn([primary]);
      const run = createRetryableCall({
        model: primary,
        retries: [fallback],
        disabled: false,
      });

      // Act
      const result = await run(fn);

      // Assert
      expect(result).toBe('OK');
      expect(fn).toHaveBeenCalledTimes(2);
    });

    it('should bypass retries when disabled returns true', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fn = vi.fn(async () => {
        throw new Error('primary failed');
      });
      const disabledFn = vi.fn(() => true);
      const run = createRetryableCall({
        model: primary,
        retries: [fallback],
        disabled: disabledFn,
      });

      // Act
      const result = run(fn);

      // Assert
      await expect(result).rejects.toThrow();
      expect(disabledFn).toHaveBeenCalledTimes(1);
      expect(fn).toHaveBeenCalledTimes(1);
    });

    it('should retry when disabled returns false', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fn = failOn([primary]);
      const disabledFn = vi.fn(() => false);
      const run = createRetryableCall({
        model: primary,
        retries: [fallback],
        disabled: disabledFn,
      });

      // Act
      const result = await run(fn);

      // Assert
      expect(result).toBe('OK');
      expect(disabledFn).toHaveBeenCalledTimes(1);
      expect(fn).toHaveBeenCalledTimes(2);
    });
  });

  describe('onError', () => {
    it('should call onError when an error occurs', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const error = new Error('primary failed');
      const onError = vi.fn();
      const fn = failOn([primary], 'OK', () => error);
      const run = createRetryableCall({
        model: primary,
        retries: [fallback],
        onError,
      });

      // Act
      await run(fn);

      // Assert
      expect(onError).toHaveBeenCalledTimes(1);
      expect(onError.mock.calls[0]![0].current.error).toBe(error);
      expect(onError.mock.calls[0]![0].current.model).toBe(primary);
      expect(onError.mock.calls[0]![0].attempts.length).toBe(1);
    });

    it('should call onError for each error across attempts', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const second = MockLanguageModel.from();
      const third = MockLanguageModel.from();
      const onError = vi.fn();
      const fn = failOn([primary, second]);
      const run = createRetryableCall({
        model: primary,
        retries: [second, third],
        onError,
      });

      // Act
      await run(fn);

      // Assert
      expect(onError).toHaveBeenCalledTimes(2);
      expect(onError.mock.calls[0]![0].current.model).toBe(primary);
      expect(onError.mock.calls[1]![0].current.model).toBe(second);
      expect(onError.mock.calls[1]![0].attempts.length).toBe(2);
    });

    it('should call onError before onRetry', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const onError = vi.fn();
      const onRetry = vi.fn();
      const fn = failOn([primary]);
      const run = createRetryableCall({
        model: primary,
        retries: [fallback],
        onError,
        onRetry,
      });

      // Act
      await run(fn);

      // Assert
      expect(onError).toHaveBeenCalledTimes(1);
      expect(onRetry).toHaveBeenCalledTimes(1);
      const errorOrder = onError.mock.invocationCallOrder[0]!;
      const retryOrder = onRetry.mock.invocationCallOrder[0]!;
      expect(errorOrder).toBeLessThan(retryOrder);
    });

    it('should expose the call options on the error context', async () => {
      // Arrange — the second (fallback) attempt carries the retry's options.
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const onError = vi.fn();
      const fn = vi.fn(async () => {
        throw new Error('failed');
      });
      const run = createRetryableCall({
        model: primary,
        retries: [{ model: fallback, options: { temperature: 0.5 } }],
        onError,
      });

      // Act
      await run(fn).catch(() => {});

      // Assert
      expect(onError).toHaveBeenCalledTimes(2);
      expect(
        onError.mock.calls[0]![0].current.options.temperature,
      ).toBeUndefined();
      expect(onError.mock.calls[1]![0].current.options.temperature).toBe(0.5);
    });
  });

  describe('onRetry', () => {
    it('should call onRetry for an error-based retry', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const onRetry = vi.fn();
      const fn = failOn([primary]);
      const run = createRetryableCall({
        model: primary,
        retries: [fallback],
        onRetry,
      });

      // Act
      await run(fn);

      // Assert — onRetry's current attempt names the next (fallback) model.
      expect(onRetry).toHaveBeenCalledTimes(1);
      expect(onRetry.mock.calls[0]![0].current.model).toBe(fallback);
      expect(onRetry.mock.calls[0]![0].attempts.length).toBe(1);
    });

    it('should call onRetry for each retry attempt', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const second = MockLanguageModel.from();
      const third = MockLanguageModel.from();
      const onRetry = vi.fn();
      const fn = failOn([primary, second]);
      const run = createRetryableCall({
        model: primary,
        retries: [second, third],
        onRetry,
      });

      // Act
      await run(fn);

      // Assert
      expect(onRetry).toHaveBeenCalledTimes(2);
      expect(onRetry.mock.calls[0]![0].current.model).toBe(second);
      expect(onRetry.mock.calls[1]![0].current.model).toBe(third);
      expect(onRetry.mock.calls[1]![0].attempts.length).toBe(2);
    });

    it('should NOT call onRetry on the first attempt', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const onRetry = vi.fn();
      const fn = vi.fn(async (_attempt: RetryCallAttempt) => 'OK');
      const run = createRetryableCall({
        model: primary,
        retries: [],
        onRetry,
      });

      // Act
      await run(fn);

      // Assert
      expect(onRetry).toHaveBeenCalledTimes(0);
    });

    describe('overrides', () => {
      it('should apply onRetry option overrides to the next attempt', async () => {
        // Arrange
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        const fn = failOn([primary]);
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

      it('should apply onRetry providerOptions overrides to the next attempt', async () => {
        // Arrange
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        const providerOptions = { openai: { store: true } };
        const fn = failOn([primary]);
        const run = createRetryableCall({
          model: primary,
          retries: [fallback],
          onRetry: () => ({ options: { providerOptions } }),
        });

        // Act
        await run(fn);

        // Assert
        expect(fn.mock.calls[1]![0].options.providerOptions).toEqual(
          providerOptions,
        );
      });

      it('should let onRetry overrides beat Retry.options', async () => {
        // Arrange
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        const fn = failOn([primary]);
        const run = createRetryableCall({
          model: primary,
          retries: [{ model: fallback, options: { temperature: 0.5 } }],
          onRetry: () => ({ options: { temperature: 0.1 } }),
        });

        // Act
        await run(fn);

        // Assert
        expect(fn.mock.calls[1]![0].options.temperature).toBe(0.1);
      });

      it('should fall back to Retry.options when onRetry returns undefined', async () => {
        // Arrange
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        const fn = failOn([primary]);
        const run = createRetryableCall({
          model: primary,
          retries: [{ model: fallback, options: { temperature: 0.5 } }],
          onRetry: () => undefined,
        });

        // Act
        await run(fn);

        // Assert
        expect(fn.mock.calls[1]![0].options.temperature).toBe(0.5);
      });

      it('should support async onRetry overrides', async () => {
        // Arrange
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        const fn = failOn([primary]);
        const run = createRetryableCall({
          model: primary,
          retries: [fallback],
          onRetry: async () => {
            await Promise.resolve();
            return { options: { temperature: 0.42 } };
          },
        });

        // Act
        await run(fn);

        // Assert
        expect(fn.mock.calls[1]![0].options.temperature).toBe(0.42);
      });
    });
  });

  describe('onComplete', () => {
    it('should call onComplete when the first attempt succeeds', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const onComplete = vi.fn();
      const fn = vi.fn(async (_attempt: RetryCallAttempt) => 'OK');
      const run = createRetryableCall({
        model: primary,
        retries: [],
        onComplete,
      });

      // Act
      const returned = await run(fn);

      // Assert — the result reaches the caller, not the hook.
      expect(returned).toBe('OK');
      expect(onComplete).toHaveBeenCalledTimes(1);
      expect(onComplete.mock.calls[0]![0].current.model).toBe(primary);
      expect('result' in onComplete.mock.calls[0]![0].current).toBe(false);
      expect(onComplete.mock.calls[0]![0].attempts.length).toBe(0);
    });

    it('should call onComplete with the model that recovered the call', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const onComplete = vi.fn();
      const fn = failOn([primary], 'FALLBACK_OK');
      const run = createRetryableCall({
        model: primary,
        retries: [fallback],
        onComplete,
      });

      // Act
      const returned = await run(fn);

      // Assert — the failed attempt precedes the successful one.
      expect(returned).toBe('FALLBACK_OK');
      expect(onComplete).toHaveBeenCalledTimes(1);
      expect(onComplete.mock.calls[0]![0].current.model).toBe(fallback);
      expect(onComplete.mock.calls[0]![0].attempts.length).toBe(1);
      expect(onComplete.mock.calls[0]![0].attempts[0].model).toBe(primary);
    });

    it('should expose the final attempt options on the context', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const onComplete = vi.fn();
      const fn = failOn([primary]);
      const run = createRetryableCall({
        model: primary,
        retries: [{ model: fallback, options: { temperature: 0.5 } }],
        onComplete,
      });

      // Act
      await run(fn);

      // Assert
      expect(onComplete.mock.calls[0]![0].current.options.temperature).toBe(
        0.5,
      );
    });

    it('should NOT call onComplete when every attempt fails', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const onComplete = vi.fn();
      const fn = vi.fn(async () => {
        throw new Error('always fails');
      });
      const run = createRetryableCall({
        model: primary,
        retries: [fallback],
        onComplete,
      });

      // Act
      await run(fn).catch(() => {});

      // Assert
      expect(onComplete).toHaveBeenCalledTimes(0);
    });

    it('should NOT call onComplete when retries are disabled', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const onComplete = vi.fn();
      const fn = vi.fn(async (_attempt: RetryCallAttempt) => 'OK');
      const run = createRetryableCall({
        model: primary,
        retries: [],
        disabled: true,
        onComplete,
      });

      // Act
      await run(fn);

      // Assert
      expect(onComplete).toHaveBeenCalledTimes(0);
    });
  });

  describe('onFailure', () => {
    it('should call onFailure with the original error when no retryable matches', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const error = new Error('boom');
      const onFailure = vi.fn();
      const fn = vi.fn(async () => {
        throw error;
      });
      const run = createRetryableCall({
        model: primary,
        retries: [],
        onFailure,
      });

      // Act
      await run(fn).catch(() => {});

      // Assert
      expect(onFailure).toHaveBeenCalledTimes(1);
      expect(onFailure.mock.calls[0]![0].error).toBe(error);
      expect(onFailure.mock.calls[0]![0].current.type).toBe('error');
      expect(onFailure.mock.calls[0]![0].current.error).toBe(error);
      expect(onFailure.mock.calls[0]![0].current.model).toBe(primary);
      expect(onFailure.mock.calls[0]![0].attempts.length).toBe(1);
    });

    it('should call onFailure with a RetryError once retries are exhausted', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const onFailure = vi.fn();
      const fn = vi.fn(async () => {
        throw new Error('always fails');
      });
      const run = createRetryableCall({
        model: primary,
        retries: [fallback],
        onFailure,
      });

      // Act
      await run(fn).catch(() => {});

      // Assert — the final attempt is the fallback's.
      expect(onFailure).toHaveBeenCalledTimes(1);
      expect(RetryError.isInstance(onFailure.mock.calls[0]![0].error)).toBe(
        true,
      );
      expect(onFailure.mock.calls[0]![0].current.model).toBe(fallback);
      expect(onFailure.mock.calls[0]![0].attempts.length).toBe(2);
    });

    it('should call onFailure when the caller aborts before the retry fires', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const controller = new AbortController();
      const error = new Error('primary failed');
      const onFailure = vi.fn();
      const fn = vi.fn(async () => {
        controller.abort();
        throw error;
      });
      const run = createRetryableCall({
        model: primary,
        retries: [fallback],
        onFailure,
      });

      // Act
      await run(fn, { abortSignal: controller.signal }).catch(() => {});

      // Assert — the retry is skipped, so the raw error is surfaced.
      expect(fn).toHaveBeenCalledTimes(1);
      expect(onFailure).toHaveBeenCalledTimes(1);
      expect(onFailure.mock.calls[0]![0].error).toBe(error);
    });

    it('should call onFailure when the caller aborts during the backoff delay', async () => {
      // Arrange — the abort lands while the retry is waiting out its delay,
      // after the failed attempt has already been recorded.
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const controller = new AbortController();
      const onComplete = vi.fn();
      const onFailure = vi.fn();
      const fn = vi.fn(async () => {
        setTimeout(() => controller.abort(), 10);
        throw new Error('primary failed');
      });
      const run = createRetryableCall({
        model: primary,
        retries: [{ model: fallback, delay: 200 }],
        onComplete,
        onFailure,
      });

      // Act
      const result = run(fn, { abortSignal: controller.signal });

      // Assert
      await expect(result).rejects.toThrow();
      expect(fn).toHaveBeenCalledTimes(1);
      expect(onComplete).toHaveBeenCalledTimes(0);
      expect(onFailure).toHaveBeenCalledTimes(1);
      expect(onFailure.mock.calls[0]![0].attempts.length).toBe(1);
    });

    it('should call onFailure when an onRetry handler throws', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const handlerError = new Error('onRetry blew up');
      const onFailure = vi.fn();
      const fn = failOn([primary]);
      const run = createRetryableCall({
        model: primary,
        retries: [fallback],
        onRetry: () => {
          throw handlerError;
        },
        onFailure,
      });

      // Act
      const result = run(fn);

      // Assert — the handler error escapes the loop and is still reported.
      await expect(result).rejects.toThrow();
      expect(onFailure).toHaveBeenCalledTimes(1);
      expect(onFailure.mock.calls[0]![0].error).toBe(handlerError);
    });

    it('should NOT re-run a completed call when an onComplete handler throws', async () => {
      // Arrange — a throwing hook must not look like a failed attempt, or the
      // driver would fail over and issue the call a second time.
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const handlerError = new Error('onComplete blew up');
      const onFailure = vi.fn();
      const fn = vi.fn(async (_attempt: RetryCallAttempt) => 'OK');
      const run = createRetryableCall({
        model: primary,
        retries: [fallback],
        onComplete: () => {
          throw handlerError;
        },
        onFailure,
      });

      // Act
      const result = run(fn);

      // Assert — one call only; the handler error surfaces unwrapped. No
      // onFailure: nothing about the attempt failed, so there is no failed
      // attempt to report as `current`.
      await expect(result).rejects.toThrow();
      await result.catch((e) => expect(e).toBe(handlerError));
      expect(fn).toHaveBeenCalledTimes(1);
      expect(onFailure).toHaveBeenCalledTimes(0);
    });

    it('should NOT call onFailure when a retryable throws on the first attempt', async () => {
      // Arrange — documents a coverage gap: the attempt is only recorded once
      // the retryables have been evaluated, so a retryable that throws leaves
      // nothing to report as the failed attempt.
      const primary = MockLanguageModel.from();
      const retryableError = new Error('retryable blew up');
      const onFailure = vi.fn();
      const fn = vi.fn(async () => {
        throw new Error('primary failed');
      });
      const run = createRetryableCall({
        model: primary,
        retries: [
          () => {
            throw retryableError;
          },
        ],
        onFailure,
      });

      // Act
      const result = run(fn);

      // Assert — the run still rejects with the retryable's error.
      await expect(result).rejects.toThrow();
      await result.catch((e) => expect(e).toBe(retryableError));
      expect(onFailure).toHaveBeenCalledTimes(0);
    });

    it('should NOT call onFailure when the call succeeds', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const onFailure = vi.fn();
      const fn = failOn([primary]);
      const run = createRetryableCall({
        model: primary,
        retries: [fallback],
        onFailure,
      });

      // Act
      await run(fn);

      // Assert
      expect(onFailure).toHaveBeenCalledTimes(0);
    });

    it('should NOT call onFailure when retries are disabled', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const onFailure = vi.fn();
      const fn = vi.fn(async () => {
        throw new Error('boom');
      });
      const run = createRetryableCall({
        model: primary,
        retries: [],
        disabled: true,
        onFailure,
      });

      // Act
      await run(fn).catch(() => {});

      // Assert
      expect(onFailure).toHaveBeenCalledTimes(0);
    });
  });

  describe('attempt', () => {
    it('should expose the model and a 1-based attempt number', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fn = failOn([primary]);
      const run = createRetryableCall({ model: primary, retries: [fallback] });

      // Act
      await run(fn);

      // Assert
      expect(fn.mock.calls[0]![0].model).toBe(primary);
      expect(fn.mock.calls[0]![0].attempt).toBe(1);
      expect(fn.mock.calls[1]![0].model).toBe(fallback);
      expect(fn.mock.calls[1]![0].attempt).toBe(2);
    });

    it('should pass the caller abortSignal through to the attempt', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const controller = new AbortController();
      const fn = vi.fn(async (_attempt: RetryCallAttempt) => 'OK');
      const run = createRetryableCall({ model: primary, retries: [] });

      // Act
      await run(fn, { abortSignal: controller.signal });

      // Assert — the attempt's signal IS the caller's, unchanged.
      expect(fn.mock.calls[0]![0].abortSignal).toBe(controller.signal);
    });

    it('should leave abortSignal undefined when the caller passes none', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fn = vi.fn(async (_attempt: RetryCallAttempt) => 'OK');
      const run = createRetryableCall({ model: primary, retries: [] });

      // Act
      await run(fn);

      // Assert
      expect(fn.mock.calls[0]![0].abortSignal).toBeUndefined();
    });

    it('should expose the per-attempt timeout as a number', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fn = vi.fn(async (_attempt: RetryCallAttempt) => 'OK');
      const run = createRetryableCall({ model: primary, retries: [] });

      // Act
      await run(fn, { timeout: 250 });

      // Assert
      expect(fn.mock.calls[0]![0].timeout).toBe(250);
    });

    it('should leave timeout undefined when none applies', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fn = vi.fn(async (_attempt: RetryCallAttempt) => 'OK');
      const run = createRetryableCall({ model: primary, retries: [] });

      // Act
      await run(fn);

      // Assert
      expect(fn.mock.calls[0]![0].timeout).toBeUndefined();
    });
  });

  describe('RetryableOptions', () => {
    describe('maxAttempts', () => {
      it('should try each model once by default', async () => {
        // Arrange
        const primary = MockLanguageModel.from();
        const fallback1 = MockLanguageModel.from();
        const fallback2 = MockLanguageModel.from();
        const finalModel = MockLanguageModel.from();
        const fn = failOn([primary, fallback1, fallback2]);
        const run = createRetryableCall({
          model: primary,
          retries: [
            fallback1,
            { model: fallback2 },
            async () => ({ model: finalModel }),
          ],
        });

        // Act
        await run(fn);

        // Assert
        expect(fn).toHaveBeenCalledTimes(4);
      });

      it('should try a model multiple times when maxAttempts is set', async () => {
        // Arrange
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        const finalModel = MockLanguageModel.from();
        const fn = failOn([primary, fallback]);
        const run = createRetryableCall({
          model: primary,
          retries: [{ model: fallback, maxAttempts: 2 }, finalModel],
        });

        // Act
        await run(fn);

        // Assert — primary(1) + fallback(2) + final(1)
        const fallbackAttempts = fn.mock.calls.filter(
          (c) => c[0]!.model === fallback,
        ).length;
        expect(fallbackAttempts).toBe(2);
        expect(fn).toHaveBeenCalledTimes(4);
      });
    });

    describe('options', () => {
      it('should expose Retry.options on the attempt', async () => {
        // Arrange
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        const fn = failOn([primary]);
        const run = createRetryableCall({
          model: primary,
          retries: [{ model: fallback, options: { temperature: 0.5 } }],
        });

        // Act
        await run(fn);

        // Assert
        expect(fn.mock.calls[1]![0].options.temperature).toBe(0.5);
      });
    });

    describe('providerOptions', () => {
      it('should expose Retry.providerOptions on the attempt', async () => {
        // Arrange — the deprecated top-level form surfaces on the attempt.
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        const providerOptions = { anthropic: { thinking: 'low' } };
        const fn = failOn([primary]);
        const run = createRetryableCall({
          model: primary,
          retries: [{ model: fallback, providerOptions }],
        });

        // Act
        await run(fn);

        // Assert
        expect(fn.mock.calls[1]![0].options.providerOptions).toEqual(
          providerOptions,
        );
      });
    });

    describe('delay', () => {
      it('should apply delay before retrying', async () => {
        // Arrange
        vi.useFakeTimers();
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        const fn = failOn([primary]);
        const run = createRetryableCall({
          model: primary,
          retries: [{ model: fallback, delay: 100 }],
        });

        // Act
        const promise = run(fn);
        await vi.runAllTimersAsync();
        const result = await promise;

        // Assert
        expect(result).toBe('OK');
        expect(fn).toHaveBeenCalledTimes(2);

        vi.useRealTimers();
      });

      it('should apply delays across multiple retries', async () => {
        // Arrange
        vi.useFakeTimers();
        const primary = MockLanguageModel.from();
        const second = MockLanguageModel.from();
        const third = MockLanguageModel.from();
        const fn = failOn([primary, second]);
        const run = createRetryableCall({
          model: primary,
          retries: [
            { model: second, delay: 50 },
            { model: third, delay: 50, backoffFactor: 2 },
          ],
        });

        // Act
        const promise = run(fn);
        await vi.runAllTimersAsync();
        const result = await promise;

        // Assert
        expect(result).toBe('OK');
        expect(fn).toHaveBeenCalledTimes(3);

        vi.useRealTimers();
      });

      it('should not delay when no delay is specified', async () => {
        // Arrange — no fake timers: a real delay would hang the test.
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        const fn = failOn([primary]);
        const run = createRetryableCall({
          model: primary,
          retries: [{ model: fallback }],
        });

        // Act
        const result = await run(fn);

        // Assert
        expect(result).toBe('OK');
        expect(fn).toHaveBeenCalledTimes(2);
      });
    });

    describe('timeout', () => {
      it('should surface a fresh per-attempt deadline as attempt.timeout', async () => {
        // Arrange — the primary "hits" its deadline (a TimeoutError); the
        // fallback answers. Each attempt receives its own timeout as a number:
        // the run timeout first, then the matched retry's own.
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        const timeouts: Array<number | undefined> = [];
        const fn = vi.fn(async ({ model, timeout }: RetryCallAttempt) => {
          timeouts.push(timeout);
          if (model === primary)
            throw new DOMException('The operation timed out', 'TimeoutError');
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
        expect(timeouts).toEqual([30, 1_000]);
      });

      it('should not retry when the caller signal is already aborted', async () => {
        // Arrange — a cancelled caller signal is a hard stop; even a retry with
        // its own timeout does not rescue it, since the dead signal is still
        // passed through to the next attempt.
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        const controller = new AbortController();
        controller.abort(new Error('user cancelled'));
        const error = new Error('primary failed');
        const fn = vi.fn(async () => {
          throw error;
        });
        const run = createRetryableCall({
          model: primary,
          retries: [{ model: fallback, timeout: 1_000 }],
        });

        // Act
        const result = run(fn, { abortSignal: controller.signal });

        // Assert
        await expect(result).rejects.toThrow();
        await result.catch((e) => expect(e).toBe(error));
        expect(fn).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe('RetryError', () => {
    it('should throw a RetryError after all attempts are exhausted', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const fallback = MockLanguageModel.from();
      const fn = vi.fn(async () => {
        throw new Error('always fails');
      });
      const run = createRetryableCall({ model: primary, retries: [fallback] });

      // Act
      const result = run(fn);

      // Assert
      await expect(result).rejects.toThrow();
      await result.catch((e) => expect(RetryError.isInstance(e)).toBe(true));
      expect(fn).toHaveBeenCalledTimes(2);
    });

    it('should throw the original error on the first attempt when no retryable matches', async () => {
      // Arrange
      const primary = MockLanguageModel.from();
      const error = new Error('boom');
      const fn = vi.fn(async () => {
        throw error;
      });
      const run = createRetryableCall({ model: primary, retries: [] });

      // Act
      const result = run(fn);

      // Assert
      await expect(result).rejects.toThrow();
      await result.catch((e) => expect(e).toBe(error));
      expect(fn).toHaveBeenCalledTimes(1);
    });
  });

  describe('reset', () => {
    describe('after-request (default)', () => {
      it('should reset to the base model on every request', async () => {
        // Arrange
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        let primaryFails = true;
        const fn = vi.fn(async ({ model }: RetryCallAttempt) => {
          if (model === primary) {
            if (primaryFails) throw new Error('primary failed');
            return 'PRIMARY_OK';
          }
          return 'FALLBACK_OK';
        });
        const run = createRetryableCall({
          model: primary,
          retries: [fallback],
        });

        // Act — first run fails over to the fallback, second starts at base.
        const first = await run(fn);
        primaryFails = false;
        const second = await run(fn);

        // Assert
        expect(first).toBe('FALLBACK_OK');
        expect(second).toBe('PRIMARY_OK');
      });
    });

    describe('after-N-requests', () => {
      it('should stick to the recovered model for N requests then reset', async () => {
        // Arrange
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        const models: Array<MockLanguageModel> = [];
        const fn = vi.fn(async ({ model }: RetryCallAttempt) => {
          models.push(model as MockLanguageModel);
          if (model === primary) throw new Error('primary failed');
          return 'OK';
        });
        const run = createRetryableCall({
          model: primary,
          retries: [fallback],
          reset: 'after-2-requests',
        });

        // Act — run 1 recovers on fallback (sticky), runs 2-3 reuse it directly,
        // run 4 resets to the base model.
        await run(fn); // primary, fallback
        await run(fn); // fallback (sticky)
        await run(fn); // fallback (sticky, last)
        await run(fn); // primary (reset), fallback

        // Assert
        expect(models).toEqual([
          primary,
          fallback,
          fallback,
          fallback,
          primary,
          fallback,
        ]);
      });
    });

    describe('after-N-seconds', () => {
      it('should stick to the recovered model within the window then reset', async () => {
        // Arrange
        vi.useFakeTimers();
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
        const models: Array<MockLanguageModel> = [];
        const fn = vi.fn(async ({ model }: RetryCallAttempt) => {
          models.push(model as MockLanguageModel);
          if (model === primary) throw new Error('primary failed');
          return 'OK';
        });
        const run = createRetryableCall({
          model: primary,
          retries: [fallback],
          reset: 'after-5-seconds',
        });

        // Act — run 1 recovers on fallback (sticky); run 2 within 5s reuses it;
        // run 3 past 5s resets to the base model.
        await run(fn); // primary, fallback
        vi.advanceTimersByTime(2_000);
        await run(fn); // fallback (sticky)
        vi.advanceTimersByTime(4_000);
        await run(fn); // primary (reset), fallback

        // Assert
        expect(models).toEqual([
          primary,
          fallback,
          fallback,
          primary,
          fallback,
        ]);

        vi.useRealTimers();
      });
    });
  });
});
