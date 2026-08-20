import { describe, expect, it, vi } from 'vitest';
import { MockLanguageModel } from '../internal/test-utils.js';
import { toCallRetryOptions } from './retry-arg.js';

const fallback = MockLanguageModel.from();

describe('toCallRetryOptions', () => {
  it('should turn the bare-array shorthand into the full options object', () => {
    // Arrange
    const retries = [fallback];

    // Act
    const options = toCallRetryOptions(retries);

    // Assert
    expect(options.retries).toBe(retries);
    expect(options.onRetry).toBeUndefined();
    expect(options.disabled).toBeUndefined();
  });

  it('should pass the object form through untouched', () => {
    // Arrange
    const onRetry = vi.fn();
    const retry = { retries: [fallback], disabled: true, onRetry };

    // Act
    const options = toCallRetryOptions(retry);

    // Assert — the same object, so a caller can rely on identity.
    expect(options).toBe(retry);
    expect(options.onRetry).toBe(onRetry);
    expect(options.disabled).toBe(true);
  });

  it('should default an absent retry to no retries at all', () => {
    // Act
    const options = toCallRetryOptions(undefined);

    // Assert
    expect(options.retries).toEqual([]);
  });

  it('should treat an empty array as the shorthand rather than as absent', () => {
    // Arrange — both produce no retries, but only one is the caller's array.
    const retries: [] = [];

    // Act
    const options = toCallRetryOptions(retries);

    // Assert
    expect(options.retries).toBe(retries);
  });
});
