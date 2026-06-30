import { describe, expect, it } from 'vitest';
import {
  Errors,
  MockLanguageModel,
  buildErrorContext,
  buildResultContext,
} from '../test-utils.js';
import { createErrorAPI } from './error.js';

const { error } = createErrorAPI<MockLanguageModel>();

describe('error', () => {
  it(`should run the predicate against the current error`, async () => {
    // Arrange
    const cond = error<MockLanguageModel, Error>(
      (e) => e.message === 'specific',
    );

    // Act
    const matched = await cond.evaluate(
      buildErrorContext(new Error('specific')),
    );
    const missed = await cond.evaluate(buildErrorContext(new Error('other')));

    // Assert
    expect(matched).toBe(true);
    expect(missed).toBe(false);
  });

  it(`should return false on result attempts`, async () => {
    // Arrange
    const cond = error<MockLanguageModel>(() => true);

    // Act
    const matched = await cond.evaluate(buildResultContext());

    // Assert
    expect(matched).toBe(false);
  });

  it(`should support async predicates`, async () => {
    // Arrange
    const cond = error<MockLanguageModel>(async () => Promise.resolve(true));

    // Act
    const matched = await cond.evaluate(buildErrorContext(new Error('x')));

    // Assert
    expect(matched).toBe(true);
  });

  describe('isRetryable', () => {
    it(`should match when isRetryable equals the requested flag`, async () => {
      // Arrange
      const cond = error.isRetryable<MockLanguageModel>(true);

      // Act
      const matched = await cond.evaluate(
        buildErrorContext(Errors.from({ isRetryable: true })),
      );
      const missed = await cond.evaluate(
        buildErrorContext(Errors.from({ isRetryable: false })),
      );

      // Assert
      expect(matched).toBe(true);
      expect(missed).toBe(false);
    });

    it(`should default to true`, async () => {
      // Arrange
      const cond = error.isRetryable<MockLanguageModel>();

      // Act
      const matched = await cond.evaluate(
        buildErrorContext(Errors.from({ isRetryable: true })),
      );

      // Assert
      expect(matched).toBe(true);
    });

    it(`should not match plain errors`, async () => {
      // Arrange
      const cond = error.isRetryable<MockLanguageModel>(true);

      // Act
      const matched = await cond.evaluate(
        buildErrorContext(new Error('plain')),
      );

      // Assert
      expect(matched).toBe(false);
    });
  });

  describe('statusCode', () => {
    it(`should match exact numeric codes`, async () => {
      // Arrange
      const cond = error.statusCode<MockLanguageModel>(429, 503);

      // Act
      const matched429 = await cond.evaluate(
        buildErrorContext(Errors.from({ statusCode: 429 })),
      );
      const matched503 = await cond.evaluate(
        buildErrorContext(Errors.from({ statusCode: 503 })),
      );
      const missed = await cond.evaluate(
        buildErrorContext(Errors.from({ statusCode: 404 })),
      );

      // Assert
      expect(matched429).toBe(true);
      expect(matched503).toBe(true);
      expect(missed).toBe(false);
    });

    it(`should accept a regex for ranges`, async () => {
      // Arrange
      const cond = error.statusCode<MockLanguageModel>(/^5\d\d$/);

      // Act
      const matched = await cond.evaluate(
        buildErrorContext(Errors.from({ statusCode: 503 })),
      );
      const missed = await cond.evaluate(
        buildErrorContext(Errors.from({ statusCode: 404 })),
      );

      // Assert
      expect(matched).toBe(true);
      expect(missed).toBe(false);
    });

    it(`should not match plain errors`, async () => {
      // Arrange
      const cond = error.statusCode<MockLanguageModel>(500);

      // Act
      const matched = await cond.evaluate(
        buildErrorContext(new Error('plain')),
      );

      // Assert
      expect(matched).toBe(false);
    });
  });

  describe('message', () => {
    it(`should match a substring`, async () => {
      // Arrange
      const cond = error.message<MockLanguageModel>('overloaded');

      // Act
      const matched = await cond.evaluate(
        buildErrorContext(new Error('Service is overloaded right now')),
      );
      const missed = await cond.evaluate(
        buildErrorContext(new Error('Bad request')),
      );

      // Assert
      expect(matched).toBe(true);
      expect(missed).toBe(false);
    });

    it(`should match substrings case-insensitively`, async () => {
      // Arrange
      const cond = error.message<MockLanguageModel>('OVERLOADED');

      // Act
      const matchedLower = await cond.evaluate(
        buildErrorContext(new Error('service overloaded')),
      );
      const matchedMixed = await cond.evaluate(
        buildErrorContext(new Error('Service OverLoaded')),
      );

      // Assert
      expect(matchedLower).toBe(true);
      expect(matchedMixed).toBe(true);
    });

    it(`should match a regex`, async () => {
      // Arrange
      const cond = error.message<MockLanguageModel>(/rate.?limit/i);

      // Act
      const matched = await cond.evaluate(
        buildErrorContext(new Error('Rate Limit exceeded')),
      );
      const missed = await cond.evaluate(
        buildErrorContext(new Error('Bad request')),
      );

      // Assert
      expect(matched).toBe(true);
      expect(missed).toBe(false);
    });

    it(`should accept a mix of string and regex patterns`, async () => {
      // Arrange
      const cond = error.message<MockLanguageModel>(
        'overloaded',
        /rate.?limit/i,
      );

      // Act
      const overloaded = await cond.evaluate(
        buildErrorContext(new Error('Service overloaded')),
      );
      const rateLimit = await cond.evaluate(
        buildErrorContext(new Error('Rate-limit hit')),
      );

      // Assert
      expect(overloaded).toBe(true);
      expect(rateLimit).toBe(true);
    });
  });

  describe('isTimeout', () => {
    it(`should match a TimeoutError`, async () => {
      // Arrange
      const cond = error.isTimeout<MockLanguageModel>();

      // Act
      const matched = await cond.evaluate(buildErrorContext(Errors.timeout()));

      // Assert
      expect(matched).toBe(true);
    });

    it(`should not match an AbortError or a plain error`, async () => {
      // Arrange
      const cond = error.isTimeout<MockLanguageModel>();

      // Act
      const aborted = await cond.evaluate(buildErrorContext(Errors.abort()));
      const plain = await cond.evaluate(buildErrorContext(new Error('boom')));

      // Assert
      expect(aborted).toBe(false);
      expect(plain).toBe(false);
    });
  });

  describe('isAbort', () => {
    it(`should match an AbortError`, async () => {
      // Arrange
      const cond = error.isAbort<MockLanguageModel>();

      // Act
      const matched = await cond.evaluate(buildErrorContext(Errors.abort()));

      // Assert
      expect(matched).toBe(true);
    });

    it(`should not match a TimeoutError or a plain error`, async () => {
      // Arrange
      const cond = error.isAbort<MockLanguageModel>();

      // Act
      const timedOut = await cond.evaluate(buildErrorContext(Errors.timeout()));
      const plain = await cond.evaluate(buildErrorContext(new Error('boom')));

      // Assert
      expect(timedOut).toBe(false);
      expect(plain).toBe(false);
    });
  });
});
