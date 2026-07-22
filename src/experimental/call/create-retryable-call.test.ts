import { RetryError } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { MockLanguageModel } from '../../internal/test-utils.js';
import { requestTimeout } from '../../retryables/request-timeout.js';
import {
  createRetryableCall,
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

    it('should leave abortSignal undefined when no deadline applies', async () => {
      // Arrange — no run timeout, no run signal, no retry timeout.
      const primary = MockLanguageModel.from();
      const fn = vi.fn(async (_attempt: RetryCallAttempt) => 'OK');
      const run = createRetryableCall({ model: primary, retries: [] });

      // Act
      await run(fn);

      // Assert
      expect(fn.mock.calls[0]![0].abortSignal).toBeUndefined();
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
      it('should give a stalled attempt a fresh deadline so it recovers', async () => {
        // Arrange
        const primary = MockLanguageModel.from();
        const fallback = MockLanguageModel.from();
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

        // Assert — each attempt gets its own signal; the spent one is aborted.
        expect(result).toBe('FALLBACK_OK');
        expect(signals.length).toBe(2);
        expect(signals[0]).toBeInstanceOf(AbortSignal);
        expect(signals[1]).toBeInstanceOf(AbortSignal);
        expect(signals[0]).not.toBe(signals[1]);
        expect(signals[0]!.aborted).toBe(true);
        expect(signals[1]!.aborted).toBe(false);
      });

      it('should NOT retry when the inbound signal is aborted and the retry has no timeout', async () => {
        // Arrange
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
          retries: [fallback],
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
