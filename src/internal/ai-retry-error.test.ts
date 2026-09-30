import { embed, generateText, RetryError } from 'ai';
import { describe, expect, it } from 'vitest';
import { AiRetryError } from './ai-retry-error.js';
import {
  createRetryableModel,
  MockEmbeddingModel,
  MockLanguageModel,
  mockResultText,
  retryableError,
} from './test-utils.js';

const createError = () => {
  const model = MockLanguageModel.from();
  const first = new Error('first');
  const last = new Error('last');
  return new AiRetryError({
    message: 'Failed after 2 attempts.',
    reason: 'maxRetriesExceeded',
    errors: [first, last],
    attempts: [
      { type: 'error', error: first, model, options: { prompt: [] } },
      { type: 'error', error: last, model, options: { prompt: [] } },
    ],
  });
};

describe('AiRetryError', () => {
  it('should be a RetryError to every check a consumer makes', () => {
    // Arrange
    const error = createError();

    // Act
    const isRetryError = RetryError.isInstance(error);

    // Assert
    expect(isRetryError).toBe(true);
    expect(error.name).toBe('AI_RetryError');
    expect(error.reason).toBe('maxRetriesExceeded');
    expect(error.errors.length).toBe(2);
    expect(error.lastError).toBe(error.errors[1]);
  });

  it('should carry the attempts', () => {
    // Arrange
    const error = createError();

    // Act
    const attempts = error.attempts;

    // Assert
    expect(attempts.length).toBe(2);
    expect(attempts[1]!.error).toBe(error.lastError);
  });

  it('should keep the attempts out of its serialized form', () => {
    // Arrange — attempts hold prompts and results, which must not reach logs.
    const error = createError();

    // Act
    const keys = [
      ...Object.keys(error),
      ...Object.keys(JSON.parse(JSON.stringify(error))),
    ];

    // Assert
    expect(keys.includes('attempts')).toBe(false);
  });

  it('should detect its own instances', () => {
    // Arrange
    const error = createError();

    // Act
    const isAiRetryError = AiRetryError.isInstance(error);

    // Assert
    expect(isAiRetryError).toBe(true);
  });

  it('should not detect a plain RetryError', () => {
    // Arrange — what the SDK's own `maxRetries` fails with.
    const error = new RetryError({
      message: 'Failed after 2 attempts.',
      reason: 'maxRetriesExceeded',
      errors: [new Error('first'), new Error('last')],
    });

    // Act
    const isAiRetryError = AiRetryError.isInstance(error);

    // Assert
    expect(isAiRetryError).toBe(false);
  });

  it('should detect an instance from another copy of the package by its marker', () => {
    // Arrange — another copy has its own class, so only the marker is shared.
    const error = { [Symbol.for('ai-retry.error.AiRetryError')]: true };

    // Act
    const isAiRetryError = AiRetryError.isInstance(error);

    // Assert
    expect(isAiRetryError).toBe(true);
  });

  describe('thrown by a retryable model', () => {
    it('should carry every language model attempt', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from(retryableError);
      const model = createRetryableModel({
        model: primary,
        retries: [fallback],
      });

      // Act
      const error = await generateText({
        model,
        prompt: 'Hello!',
        maxRetries: 0,
      }).catch((e: unknown) => e);

      // Assert
      expect(AiRetryError.isInstance(error)).toBe(true);
      expect((error as AiRetryError).attempts.map((a) => a.model)).toEqual([
        primary,
        fallback,
      ]);
    });

    it('should carry every embedding model attempt', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from(retryableError);
      const fallback = MockEmbeddingModel.from(retryableError);
      const model = createRetryableModel({
        model: primary,
        retries: [fallback],
      });

      // Act
      const error = await embed({
        model,
        value: 'Hello!',
        maxRetries: 0,
      }).catch((e: unknown) => e);

      // Assert
      expect((error as AiRetryError).attempts.map((a) => a.model)).toEqual([
        primary,
        fallback,
      ]);
    });

    it('should not re-run a model a nested retryable model already tried', async () => {
      // Arrange
      const primary = MockLanguageModel.from(retryableError);
      const fallback = MockLanguageModel.from(retryableError);
      const other = MockLanguageModel.from(mockResultText);
      const inner = createRetryableModel({
        model: primary,
        retries: [fallback],
      });
      const outer = createRetryableModel({
        model: inner,
        retries: [fallback, other],
      });

      // Act
      const result = await generateText({
        model: outer,
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(result.text).toBe(mockResultText);
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });
  });
});
