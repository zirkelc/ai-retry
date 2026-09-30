import { RetryError } from 'ai';
import { describe, expect, it } from 'vitest';
import { AiRetryError } from './ai-retry-error.js';
import { countModelAttempts } from './count-model-attempts.js';
import { MockLanguageModel } from './test-utils.js';
import type { CallArgs, CallRetryAttempt } from '../call/types.js';
import type {
  LanguageModel,
  LanguageModelCallOptions,
  ModelRetryAttempt,
} from '../types.js';

describe('countModelAttempts', () => {
  const mockModel1 = MockLanguageModel.from();
  const mockModel2 = MockLanguageModel.from();
  const mockOptions: LanguageModelCallOptions = {
    prompt: [],
  };

  it('should return 0 when no attempts', () => {
    const attempts: Array<ModelRetryAttempt<LanguageModel>> = [];
    expect(countModelAttempts(mockModel1, attempts)).toBe(0);
  });

  it('should count single model attempts', () => {
    const attempts: Array<ModelRetryAttempt<LanguageModel>> = [
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel1,
        options: mockOptions,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel1,
        options: mockOptions,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel1,
        options: mockOptions,
      },
    ];
    expect(countModelAttempts(mockModel1, attempts)).toBe(3);
  });

  it('should count only matching model attempts', () => {
    const attempts: Array<ModelRetryAttempt<LanguageModel>> = [
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel1,
        options: mockOptions,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel2,
        options: mockOptions,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel1,
        options: mockOptions,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel2,
        options: mockOptions,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel1,
        options: mockOptions,
      },
    ];
    expect(countModelAttempts(mockModel1, attempts)).toBe(3);
    expect(countModelAttempts(mockModel2, attempts)).toBe(2);
  });

  it('should return 0 when no matching model', () => {
    const attempts: Array<ModelRetryAttempt<LanguageModel>> = [
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel2,
        options: mockOptions,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel2,
        options: mockOptions,
      },
    ];
    expect(countModelAttempts(mockModel1, attempts)).toBe(0);
  });

  it('should count call-layer attempts the same way', () => {
    // Arrange — the helper is shared, and a call attempt records the entry
    // point's arguments where a model attempt records provider call options.
    const attempts: Array<CallRetryAttempt<LanguageModel>> = [
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel1,
        options: {} as CallArgs<LanguageModel>,
      },
      {
        type: 'error',
        error: new Error('test'),
        model: mockModel2,
        options: {} as CallArgs<LanguageModel>,
      },
    ];

    // Act & Assert
    expect(countModelAttempts(mockModel1, attempts)).toBe(1);
    expect(countModelAttempts(mockModel2, attempts)).toBe(1);
  });

  describe('through a retryable model', () => {
    const primary = MockLanguageModel.from();
    const fallback = MockLanguageModel.from();
    const wrapper = MockLanguageModel.from();

    /** What a retryable model fails with after trying both models. */
    const retryableModelError = () =>
      new AiRetryError({
        message: 'Failed after 2 attempts.',
        reason: 'maxRetriesExceeded',
        errors: [new Error('first'), new Error('last')],
        attempts: [
          {
            type: 'error',
            error: new Error('first'),
            model: primary,
            options: mockOptions,
          },
          {
            type: 'error',
            error: new Error('last'),
            model: fallback,
            options: mockOptions,
          },
        ],
      });

    const attemptWith = (
      error: unknown,
    ): Array<ModelRetryAttempt<LanguageModel>> => [
      { type: 'error', error, model: wrapper, options: mockOptions },
    ];

    it('should count the models the retryable model tried, not the wrapper', () => {
      // Arrange
      const attempts = attemptWith(retryableModelError());

      // Act
      const counts = [primary, fallback, wrapper].map((m) =>
        countModelAttempts(m, attempts),
      );

      // Assert
      expect(counts).toEqual([1, 1, 0]);
    });

    it('should read one level down into the RetryError of the SDK retries', () => {
      // Arrange — the SDK ran the retryable model twice: once it failed with a
      // plain error, counted on the wrapper, once after trying both models.
      const attempts = attemptWith(
        new RetryError({
          message: 'Failed after 2 attempts.',
          reason: 'errorNotRetryable',
          errors: [new Error('plain'), retryableModelError()],
        }),
      );

      // Act
      const counts = [primary, fallback, wrapper].map((m) =>
        countModelAttempts(m, attempts),
      );

      // Assert
      expect(counts).toEqual([1, 1, 1]);
    });

    it('should read a retryable model error given as the cause', () => {
      // Arrange
      const attempts = attemptWith(
        new Error('wrapped', { cause: retryableModelError() }),
      );

      // Act
      const counts = [primary, fallback, wrapper].map((m) =>
        countModelAttempts(m, attempts),
      );

      // Assert
      expect(counts).toEqual([1, 1, 0]);
    });

    it('should count the wrapper for a plain RetryError of the SDK retries', () => {
      // Arrange
      const attempts = attemptWith(
        new RetryError({
          message: 'Failed after 2 attempts.',
          reason: 'maxRetriesExceeded',
          errors: [new Error('first'), new Error('last')],
        }),
      );

      // Act
      const count = countModelAttempts(wrapper, attempts);

      // Assert
      expect(count).toBe(1);
    });
  });
});
