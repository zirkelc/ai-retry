import { describe, expect, it } from 'vitest';
import { resolveBackoffDelay } from './resolve-backoff-delay.js';
import { MockLanguageModel } from './test-utils.js';
import type {
  LanguageModel,
  LanguageModelCallOptions,
  Retry,
  RetryAttempt,
} from '../types.js';

describe('resolveBackoffDelay', () => {
  const model = MockLanguageModel.from();
  const other = MockLanguageModel.from();
  const options: LanguageModelCallOptions = { prompt: [] };

  /** Build N prior error attempts against the given model. */
  const attemptsFor = (
    m: LanguageModel,
    count: number,
  ): Array<RetryAttempt<LanguageModel>> =>
    Array.from({ length: count }, () => ({
      type: 'error' as const,
      error: new Error('boom'),
      model: m,
      options,
    }));

  it('should return undefined when the retry sets no delay', () => {
    // Arrange
    const retry: Retry<LanguageModel> = { model };

    // Act
    const result = resolveBackoffDelay(retry, []);

    // Assert
    expect(result).toBe(undefined);
  });

  it('should return the base delay on the first attempt against the model', () => {
    // Arrange
    const retry: Retry<LanguageModel> = { model, delay: 100 };

    // Act
    const result = resolveBackoffDelay(retry, []);

    // Assert
    expect(result).toBe(100);
  });

  it('should grow the delay by the backoff factor per prior attempt on that model', () => {
    // Arrange — two prior attempts already made against this model.
    const retry: Retry<LanguageModel> = {
      model,
      delay: 100,
      backoffFactor: 2,
    };

    // Act
    const result = resolveBackoffDelay(retry, attemptsFor(model, 2));

    // Assert — 100 * 2^2.
    expect(result).toBe(400);
  });

  it('should only count attempts against the retry model', () => {
    // Arrange — prior attempts belong to a different model.
    const retry: Retry<LanguageModel> = {
      model,
      delay: 100,
      backoffFactor: 2,
    };

    // Act
    const result = resolveBackoffDelay(retry, attemptsFor(other, 3));

    // Assert — 0 matching attempts → 100 * 2^0.
    expect(result).toBe(100);
  });
});
