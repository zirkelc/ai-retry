import { describe, expect, it } from 'vitest';
import { retryDiesOnAbortedSignal } from './retry-dies-on-aborted-signal.js';

describe('retryDiesOnAbortedSignal', () => {
  it('should be true when the signal is aborted and the retry has no timeout', () => {
    // Arrange
    const signal = AbortSignal.abort();

    // Act
    const result = retryDiesOnAbortedSignal(signal, {});

    // Assert
    expect(result).toBe(true);
  });

  it('should be false when the retry supplies a fresh deadline', () => {
    // Arrange — aborted signal, but the retry mints its own timeout.
    const signal = AbortSignal.abort();

    // Act
    const result = retryDiesOnAbortedSignal(signal, { timeout: 1_000 });

    // Assert
    expect(result).toBe(false);
  });

  it('should be false when the signal is not aborted', () => {
    // Arrange
    const signal = new AbortController().signal;

    // Act
    const result = retryDiesOnAbortedSignal(signal, {});

    // Assert
    expect(result).toBe(false);
  });

  it('should be false when there is no inbound signal', () => {
    // Act
    const result = retryDiesOnAbortedSignal(undefined, {});

    // Assert
    expect(result).toBe(false);
  });
});
