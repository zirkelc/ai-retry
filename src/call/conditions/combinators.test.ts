import { describe, expect, it } from 'vitest';
import { and } from '../../internal/conditions/and.js';
import { createErrorAPI } from '../../internal/conditions/error.js';
import { not } from '../../internal/conditions/not.js';
import { or } from '../../internal/conditions/or.js';
import {
  apiError,
  buildCallErrorContext,
  MockLanguageModel,
} from '../../internal/test-utils.js';

const { error, httpStatus } = createErrorAPI<MockLanguageModel, 'call'>();

const boom = buildCallErrorContext(new Error('service overloaded'));

describe('combinators (call layer)', () => {
  it('should evaluate and() against a call context', async () => {
    // Arrange
    const both = and(
      error<MockLanguageModel>(() => true),
      httpStatus<MockLanguageModel>('overloaded'),
    );
    const one = and(
      error<MockLanguageModel>(() => true),
      httpStatus<MockLanguageModel>('unrelated'),
    );

    // Act & Assert
    expect(await both.evaluate(boom)).toBe(true);
    expect(await one.evaluate(boom)).toBe(false);
  });

  it('should evaluate or() against a call context', async () => {
    // Arrange
    const cond = or(
      httpStatus<MockLanguageModel>('unrelated'),
      httpStatus<MockLanguageModel>('overloaded'),
    );

    // Act & Assert
    expect(await cond.evaluate(boom)).toBe(true);
  });

  it('should evaluate not() against a call context', async () => {
    // Arrange
    const cond = not(httpStatus<MockLanguageModel>('overloaded'));

    // Act & Assert
    expect(await cond.evaluate(boom)).toBe(false);
  });
});

describe('actions (call layer)', () => {
  const fallback = MockLanguageModel.from();

  it('should produce a retryable that fires only on a match', async () => {
    // Arrange
    const retryable = httpStatus<MockLanguageModel>('overloaded').switch({
      model: fallback,
    });

    // Act
    const fired = await retryable(boom);
    const skipped = await retryable(
      buildCallErrorContext(new Error('unrelated')),
    );

    // Assert
    expect(fired?.model).toBe(fallback);
    expect(fired?.maxAttempts).toBe(1);
    expect(skipped).toBe(undefined);
  });

  it('should reuse the current model for retry()', async () => {
    // Arrange
    const current = MockLanguageModel.from();
    const retryable = httpStatus<MockLanguageModel>('overloaded').retry();

    // Act
    const fired = await retryable(
      buildCallErrorContext(new Error('service overloaded'), current),
    );

    // Assert
    expect(fired?.model).toBe(current);
    expect(fired?.maxAttempts).toBe(2);
  });

  it('should honor a Retry-After header over a configured delay', async () => {
    // Arrange — the header path runs off the attempt's error, which the call
    // layer records the same way the model layer does.
    const retryable = error<MockLanguageModel>(() => true).retry({
      delay: 5_000,
      backoffFactor: 3,
    });

    // Act
    const fired = await retryable(
      buildCallErrorContext(
        apiError({
          statusCode: 429,
          responseHeaders: { 'retry-after': '2' },
        }),
      ),
    );

    // Assert
    expect(fired?.delay).toBe(2_000);
    expect(fired?.backoffFactor).toBe(1);
  });

  it('should cap a Retry-After header at 60 seconds', async () => {
    // Arrange
    const retryable = error<MockLanguageModel>(() => true).retry();

    // Act
    const fired = await retryable(
      buildCallErrorContext(
        apiError({
          statusCode: 429,
          responseHeaders: { 'retry-after': '600' },
        }),
      ),
    );

    // Assert
    expect(fired?.delay).toBe(60_000);
  });

  it('should reject maxAttempts below 2 on retry()', () => {
    // Arrange
    const cond = httpStatus<MockLanguageModel>(429);

    // Act
    const build = () => cond.retry({ maxAttempts: 1 });

    // Assert
    expect(build).toThrow();
  });
});
