import { RetryError } from 'ai';
import { describe, expect, it } from 'vitest';
import { prepareRetryError } from './prepare-retry-error.js';

const first = new Error('first');
const last = new Error('last');

describe('prepareRetryError', () => {
  it('should wrap every attempt error in order', () => {
    // Arrange
    const attempts = [
      { type: 'error', error: first },
      { type: 'error', error: last },
    ];

    // Act
    const retryError = prepareRetryError(last, attempts);

    // Assert
    expect(RetryError.isInstance(retryError)).toBe(true);
    expect(retryError.errors.length).toBe(2);
    expect(retryError.errors[0]).toBe(first);
    expect(retryError.errors[1]).toBe(last);
  });

  it('should report the attempt count and the last error', () => {
    // Arrange
    const attempts = [
      { type: 'error', error: first },
      { type: 'error', error: last },
    ];

    // Act
    const retryError = prepareRetryError(last, attempts);

    // Assert — the error is stringified rather than reduced to `.message`,
    // so an `Error` carries its class name into the summary.
    expect(retryError.message).toBe(
      'Failed after 2 attempts. Last error: Error: last',
    );
    expect(retryError.reason).toBe('maxRetriesExceeded');
  });

  it('should describe a model-layer result attempt by its finishReason', () => {
    // Arrange — below a model the reason sits on the attempt.
    const attempts = [
      { type: 'result', finishReason: 'content-filter' },
      { type: 'error', error: last },
    ];

    // Act
    const retryError = prepareRetryError(last, attempts);

    // Assert
    expect(retryError.errors[0]).toBe(
      'Result with finishReason: content-filter',
    );
  });

  it('should describe a call-layer result attempt by the result finishReason', () => {
    // Arrange — around a call the reason sits on the result instead.
    const attempts = [
      { type: 'result', result: { finishReason: 'length' } },
      { type: 'error', error: last },
    ];

    // Act
    const retryError = prepareRetryError(last, attempts);

    // Assert
    expect(retryError.errors[0]).toBe('Result with finishReason: length');
  });

  it('should describe a result attempt that has no finish reason', () => {
    // Arrange — an embedding or an image has none to report.
    const attempts = [
      { type: 'result', result: { embedding: [0, 0, 0] } },
      { type: 'error', error: last },
    ];

    // Act
    const retryError = prepareRetryError(last, attempts);

    // Assert
    expect(retryError.errors[0]).toBe('Result');
  });

  it('should prefer the attempt finishReason over the result one', () => {
    // Arrange — only one layer sets each, but the precedence is fixed.
    const attempts = [
      {
        type: 'result',
        finishReason: 'content-filter',
        result: { finishReason: 'length' },
      },
    ];

    // Act
    const retryError = prepareRetryError(last, attempts);

    // Assert
    expect(retryError.errors[0]).toBe(
      'Result with finishReason: content-filter',
    );
  });
});
