import { RetryError } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { evaluateError } from './evaluate-error.js';
import { MockLanguageModel } from './test-utils.js';
import type {
  LanguageModel,
  LanguageModelCallOptions,
  RetryAttempt,
} from '../types.js';

describe('evaluateError', () => {
  const primary = MockLanguageModel.from();
  const fallback = MockLanguageModel.from();
  const options: LanguageModelCallOptions = { prompt: [] };

  it('should return the matched retry model and notify onError', async () => {
    // Arrange
    const error = new Error('boom');
    const onError = vi.fn();

    // Act
    const result = await evaluateError({
      error,
      model: primary,
      options,
      attempts: [],
      retries: [fallback],
      onError,
    });

    // Assert
    expect(result.retryModel?.model).toBe(fallback);
    expect(result.attempt.type).toBe('error');
    expect(result.attempt.error).toBe(error);
    expect(result.finalError).toBe(undefined);
    expect(onError.mock.calls.length).toBe(1);
  });

  it('should surface the original error on the first attempt when no retry matches', async () => {
    // Arrange
    const error = new Error('boom');

    // Act
    const result = await evaluateError({
      error,
      model: primary,
      options,
      attempts: [],
      retries: [],
    });

    // Assert
    expect(result.retryModel).toBe(undefined);
    expect(result.finalError).toBe(error);
  });

  it('should wrap in a RetryError when no retry matches after multiple attempts', async () => {
    // Arrange — one prior attempt already recorded.
    const error = new Error('second');
    const priorAttempts: Array<RetryAttempt<LanguageModel>> = [
      { type: 'error', error: new Error('first'), model: primary, options },
    ];

    // Act
    const result = await evaluateError({
      error,
      model: fallback,
      options,
      attempts: priorAttempts,
      retries: [],
    });

    // Assert
    expect(result.retryModel).toBe(undefined);
    expect(RetryError.isInstance(result.finalError)).toBe(true);
  });
});
