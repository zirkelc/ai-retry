import { describe, expect, it, vi } from 'vitest';
import {
  isCallAbort,
  isRetryCancelled,
  retryCancelSignal,
  waitBeforeRetry,
} from './retry-signal.js';

const abortError = () =>
  new DOMException('The operation was aborted.', 'AbortError');
const timeoutError = () =>
  new DOMException('The operation timed out.', 'TimeoutError');

describe('retryCancelSignal', () => {
  it('should return the base unchanged when the retry has no timeout', () => {
    // Arrange
    const base = AbortSignal.abort();

    // Act
    const result = retryCancelSignal(base, {});

    // Assert
    expect(result).toBe(base);
  });

  it('should return undefined when there is no base', () => {
    // Arrange
    const base = undefined;

    // Act
    const result = retryCancelSignal(base, { timeout: 1_000 });

    // Assert
    expect(result).toBe(undefined);
  });

  it('should drop a base that already timed out when the retry has a timeout', () => {
    // Arrange
    const base = AbortSignal.abort(timeoutError());

    // Act
    const result = retryCancelSignal(base, { timeout: 1_000 });

    // Assert
    expect(result).toBe(undefined);
  });

  it('should return a base that was already cancelled when the retry has a timeout', () => {
    // Arrange
    const base = AbortSignal.abort();

    // Act
    const result = retryCancelSignal(base, { timeout: 1_000 });

    // Assert
    expect(result).toBe(base);
  });

  it('should follow a later cancel of a live base but not a later timeout', () => {
    // Arrange
    const cancelled = new AbortController();
    const timedOut = new AbortController();

    // Act
    const followsCancel = retryCancelSignal(cancelled.signal, {
      timeout: 1_000,
    });
    const followsTimeout = retryCancelSignal(timedOut.signal, {
      timeout: 1_000,
    });
    cancelled.abort();
    timedOut.abort(timeoutError());

    // Assert
    expect(followsCancel?.aborted).toBe(true);
    expect(followsTimeout?.aborted).toBe(false);
  });
});

describe('isRetryCancelled', () => {
  it('should be true for a cancelled base, with or without a retry timeout', () => {
    // Arrange
    const base = AbortSignal.abort();

    // Act
    const withoutTimeout = isRetryCancelled(base, {});
    const withTimeout = isRetryCancelled(base, { timeout: 1_000 });

    // Assert
    expect(withoutTimeout).toBe(true);
    expect(withTimeout).toBe(true);
  });

  it('should be false for a timed-out base only when the retry has a timeout', () => {
    // Arrange
    const base = AbortSignal.abort(timeoutError());

    // Act
    const withoutTimeout = isRetryCancelled(base, {});
    const withTimeout = isRetryCancelled(base, { timeout: 1_000 });

    // Assert
    expect(withoutTimeout).toBe(true);
    expect(withTimeout).toBe(false);
  });

  it('should be false for a live base', () => {
    // Arrange
    const base = new AbortController().signal;

    // Act
    const result = isRetryCancelled(base, {});

    // Assert
    expect(result).toBe(false);
  });

  it('should be false when there is no base', () => {
    // Arrange
    const base = undefined;

    // Act
    const result = isRetryCancelled(base, {});

    // Assert
    expect(result).toBe(false);
  });

  it('should not listen on a live base', () => {
    // Arrange
    const base = new AbortController().signal;
    const addEventListener = vi.spyOn(base, 'addEventListener');

    // Act
    const result = isRetryCancelled(base, { timeout: 1_000 });

    // Assert
    expect(result).toBe(false);
    expect(addEventListener.mock.calls.length).toBe(0);
  });
});

describe('isCallAbort', () => {
  it('should be true for an abort error while the call is cancelled', () => {
    // Arrange
    const base = AbortSignal.abort();

    // Act
    const result = isCallAbort(abortError(), base);

    // Assert
    expect(result).toBe(true);
  });

  it('should be true for a timeout error while the call deadline has fired', () => {
    // Arrange
    const base = AbortSignal.abort(timeoutError());

    // Act
    const result = isCallAbort(timeoutError(), base);

    // Assert
    expect(result).toBe(true);
  });

  it('should be false for a retry that timed out on its own deadline after the call deadline fired', () => {
    // Arrange
    const base = AbortSignal.abort(timeoutError());

    // Act
    const result = isCallAbort(timeoutError(), base, { timeout: 1_000 });

    // Assert
    expect(result).toBe(false);
  });

  it('should be false while the call signal is live', () => {
    // Arrange
    const base = new AbortController().signal;

    // Act
    const result = isCallAbort(abortError(), base);

    // Assert
    expect(result).toBe(false);
  });

  it('should be false for an error that is not an abort', () => {
    // Arrange
    const base = AbortSignal.abort();

    // Act
    const result = isCallAbort(new Error('boom'), base);

    // Assert
    expect(result).toBe(false);
  });
});

describe('waitBeforeRetry', () => {
  it('should wait out the delay when the call deadline fired and the retry has a timeout', async () => {
    // Arrange
    const base = AbortSignal.abort(timeoutError());

    // Act
    const result = waitBeforeRetry(10, base, { timeout: 1_000 });

    // Assert
    await expect(result).resolves.toBe(undefined);
  });

  it('should reject when the call is cancelled during the delay', async () => {
    // Arrange
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 10);

    // Act
    const result = waitBeforeRetry(5_000, controller.signal, {
      timeout: 1_000,
    });

    // Assert
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('should remove every listener it added once the delay is over', async () => {
    // Arrange
    const base = new AbortController().signal;
    const addEventListener = vi.spyOn(base, 'addEventListener');
    const removeEventListener = vi.spyOn(base, 'removeEventListener');

    // Act
    await waitBeforeRetry(10, base, { timeout: 1_000 });

    // Assert
    expect(removeEventListener.mock.calls.length).toBe(
      addEventListener.mock.calls.length,
    );
  });

  it('should reject with the signal reason when it is an abort error', async () => {
    // Arrange
    const controller = new AbortController();
    setTimeout(() => controller.abort(timeoutError()), 10);

    // Act
    const result = waitBeforeRetry(5_000, controller.signal, undefined);

    // Assert
    await expect(result).rejects.toMatchObject({ name: 'TimeoutError' });
  });

  it('should reject with an abort error when the signal reason is not one', async () => {
    // Arrange
    const controller = new AbortController();
    setTimeout(() => controller.abort(new Error('custom stop')), 10);

    // Act
    const result = waitBeforeRetry(5_000, controller.signal, {
      timeout: 1_000,
    });

    // Assert
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('should not wait without a delay', async () => {
    // Arrange
    const base = AbortSignal.abort();

    // Act
    const result = waitBeforeRetry(undefined, base, {});

    // Assert
    await expect(result).resolves.toBe(undefined);
  });
});
