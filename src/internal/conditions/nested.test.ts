import { Errors } from 'ai-test-kit';
import { describe, expect, it } from 'vitest';
import { buildErrorContext, MockLanguageModel } from '../test-utils.js';
import { createErrorAPI } from './error.js';
import { and } from './and.js';
import { not } from './not.js';
import { or } from './or.js';

const { error, httpStatus } = createErrorAPI<MockLanguageModel>();

/**
 * Retry transient server errors on the same model with backoff, but
 * carve out rate-limit/overload (those switch providers elsewhere).
 * Reads like a Drizzle `where` clause: and(isRetryable, not(or(...))).
 */
const retryable = and(
  error.isRetryable(true),
  not(or(httpStatus(429), httpStatus(529))),
).retry({ delay: 1_000, backoffFactor: 2 });

describe('nested and/or/not composition', () => {
  it(`should match a retryable server error outside the carve-out (503)`, async () => {
    // Arrange
    const ctx = buildErrorContext(Errors.serviceUnavailable());

    // Act
    const retry = await retryable(ctx);

    // Assert
    expect(retry?.model).toBe(ctx.current.model);
  });

  it(`should match a retryable server error outside the carve-out (500)`, async () => {
    // Arrange
    const ctx = buildErrorContext(Errors.internalServerError());

    // Act
    const retry = await retryable(ctx);

    // Assert
    expect(retry?.model).toBe(ctx.current.model);
  });

  it(`should not match a rate-limit error (carved out by not/or)`, async () => {
    // Arrange
    const ctx = buildErrorContext(Errors.rateLimited());

    // Act
    const retry = await retryable(ctx);

    // Assert
    expect(retry).toBeUndefined();
  });

  it(`should not match an overloaded error (carved out by not/or)`, async () => {
    // Arrange
    const ctx = buildErrorContext(Errors.serviceOverloaded());

    // Act
    const retry = await retryable(ctx);

    // Assert
    expect(retry).toBeUndefined();
  });

  it(`should not match a non-retryable error (excluded by and)`, async () => {
    // Arrange
    const ctx = buildErrorContext(Errors.badRequest());

    // Act
    const retry = await retryable(ctx);

    // Assert
    expect(retry).toBeUndefined();
  });
});
