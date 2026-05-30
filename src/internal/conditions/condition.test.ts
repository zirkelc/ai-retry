import { generateText } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  apiError,
  buildErrorContext,
  createRetryable,
  generateTextResult,
  MockLanguageModel,
} from '../test-utils.js';
import { Condition } from './condition.js';

describe('Condition', () => {
  describe('evaluate', () => {
    it(`should resolve true when predicate returns true`, async () => {
      // Arrange
      const cond = new Condition<MockLanguageModel>(() => true);
      const ctx = buildErrorContext(new Error('boom'));

      // Act
      const result = await cond.evaluate(ctx);

      // Assert
      expect(result).toBe(true);
    });

    it(`should resolve false when predicate returns false`, async () => {
      // Arrange
      const cond = new Condition<MockLanguageModel>(() => false);
      const ctx = buildErrorContext(new Error('boom'));

      // Act
      const result = await cond.evaluate(ctx);

      // Assert
      expect(result).toBe(false);
    });

    it(`should await an async predicate`, async () => {
      // Arrange
      const cond = new Condition<MockLanguageModel>(async () =>
        Promise.resolve(true),
      );
      const ctx = buildErrorContext(new Error('boom'));

      // Act
      const result = await cond.evaluate(ctx);

      // Assert
      expect(result).toBe(true);
    });
  });

  describe('switch', () => {
    it(`should return a Retry with the target model when predicate matches`, async () => {
      // Arrange
      const cond = new Condition<MockLanguageModel>(() => true);
      const target = new MockLanguageModel();
      const retryable = cond.switch({ model: target });
      const ctx = buildErrorContext(new Error('boom'));

      // Act
      const retry = await retryable(ctx);

      // Assert
      expect(retry).toEqual({ model: target, maxAttempts: 1 });
    });

    it(`should return undefined when predicate does not match`, async () => {
      // Arrange
      const cond = new Condition<MockLanguageModel>(() => false);
      const target = new MockLanguageModel();
      const retryable = cond.switch({ model: target });
      const ctx = buildErrorContext(new Error('boom'));

      // Act
      const retry = await retryable(ctx);

      // Assert
      expect(retry).toBeUndefined();
    });

    it(`should pass extra Retry options through`, async () => {
      // Arrange
      const cond = new Condition<MockLanguageModel>(() => true);
      const target = new MockLanguageModel();
      const retryable = cond.switch({
        model: target,
        maxAttempts: 3,
        delay: 500,
      });
      const ctx = buildErrorContext(new Error('boom'));

      // Act
      const retry = await retryable(ctx);

      // Assert
      expect(retry).toEqual({ model: target, maxAttempts: 3, delay: 500 });
    });
  });

  describe('retry', () => {
    it(`should reuse the current model when predicate matches`, async () => {
      // Arrange
      const cond = new Condition<MockLanguageModel>(() => true);
      const retryable = cond.retry({ delay: 100 });
      const ctx = buildErrorContext(new Error('boom'));

      // Act
      const retry = await retryable(ctx);

      // Assert
      expect(retry).toEqual({
        model: ctx.current.model,
        maxAttempts: 2,
        delay: 100,
      });
    });

    it(`should honor the Retry-After header when present (seconds)`, async () => {
      // Arrange
      const cond = new Condition<MockLanguageModel>(() => true);
      const retryable = cond.retry({ delay: 100, backoffFactor: 2 });
      const ctx = buildErrorContext(
        apiError({
          message: 'Rate limited',
          statusCode: 429,
          responseHeaders: { 'retry-after': '5' },
        }),
      );

      // Act
      const retry = await retryable(ctx);

      // Assert
      expect(retry).toEqual({
        model: ctx.current.model,
        maxAttempts: 2,
        delay: 5000,
        backoffFactor: 1,
      });
    });

    it(`should cap the Retry-After delay at MAX_RETRY_AFTER_MS`, async () => {
      // Arrange
      const cond = new Condition<MockLanguageModel>(() => true);
      const retryable = cond.retry();
      const ctx = buildErrorContext(
        apiError({
          message: 'Rate limited',
          statusCode: 429,
          responseHeaders: { 'retry-after': '600' }, // 600 seconds = 10 minutes
        }),
      );

      // Act
      const retry = await retryable(ctx);

      // Assert
      expect(retry).toEqual({
        model: ctx.current.model,
        maxAttempts: 2,
        delay: 60_000,
        backoffFactor: 1,
      });
    });

    it(`should fall back to provided delay when no Retry-After header`, async () => {
      // Arrange
      const cond = new Condition<MockLanguageModel>(() => true);
      const retryable = cond.retry({ delay: 250, backoffFactor: 2 });
      const ctx = buildErrorContext(
        apiError({ message: 'Rate limited', statusCode: 429 }),
      );

      // Act
      const retry = await retryable(ctx);

      // Assert
      expect(retry).toEqual({
        model: ctx.current.model,
        maxAttempts: 2,
        delay: 250,
        backoffFactor: 2,
      });
    });

    it(`should return undefined when predicate does not match`, async () => {
      // Arrange
      const cond = new Condition<MockLanguageModel>(() => false);
      const retryable = cond.retry();
      const ctx = buildErrorContext(new Error('boom'));

      // Act
      const retry = await retryable(ctx);

      // Assert
      expect(retry).toBeUndefined();
    });

    it(`should default maxAttempts to 2`, async () => {
      // Arrange
      const cond = new Condition<MockLanguageModel>(() => true);
      const retryable = cond.retry();
      const ctx = buildErrorContext(new Error('boom'));

      // Act
      const retry = await retryable(ctx);

      // Assert
      expect(retry).toEqual({ model: ctx.current.model, maxAttempts: 2 });
    });

    it(`should preserve a higher user-provided maxAttempts`, async () => {
      // Arrange
      const cond = new Condition<MockLanguageModel>(() => true);
      const retryable = cond.retry({ maxAttempts: 5 });
      const ctx = buildErrorContext(new Error('boom'));

      // Act
      const retry = await retryable(ctx);

      // Assert
      expect(retry).toEqual({ model: ctx.current.model, maxAttempts: 5 });
    });

    it(`should throw when maxAttempts is less than 2`, () => {
      // Arrange
      const cond = new Condition<MockLanguageModel>(() => true);

      // Act + Assert
      expect(() => cond.retry({ maxAttempts: 1 })).toThrow(/maxAttempts >= 2/);
      expect(() => cond.retry({ maxAttempts: 0 })).toThrow(/maxAttempts >= 2/);
    });
  });

  describe('integration with createRetryable', () => {
    it(`should switch to fallback when condition matches`, async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: apiError({ statusCode: 500 }),
      });
      const fallback = new MockLanguageModel({
        doGenerate: generateTextResult('hello'),
      });
      const cond = new Condition<MockLanguageModel>(() => true);

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [cond.switch({ model: fallback })],
        }),
        prompt: 'hi',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallback.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe('hello');
    });
  });
});
