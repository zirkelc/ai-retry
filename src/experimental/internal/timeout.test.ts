import { describe, expect, it } from 'vitest';
import {
  abortError,
  buildErrorContext,
  MockLanguageModel,
  timeoutError,
} from '../../internal/test-utils.js';
import { createErrorAPI } from './error.js';

const { timeout } = createErrorAPI<MockLanguageModel>();

describe('timeout', () => {
  it(`should match TimeoutError`, async () => {
    // Arrange
    const cond = timeout<MockLanguageModel>();

    // Act
    const matched = await cond.evaluate(buildErrorContext(timeoutError()));

    // Assert
    expect(matched).toBe(true);
  });

  it(`should not match AbortError`, async () => {
    // Arrange
    const cond = timeout<MockLanguageModel>();

    // Act
    const matched = await cond.evaluate(buildErrorContext(abortError()));

    // Assert
    expect(matched).toBe(false);
  });

  it(`should not match a plain Error`, async () => {
    // Arrange
    const cond = timeout<MockLanguageModel>();

    // Act
    const matched = await cond.evaluate(buildErrorContext(new Error('boom')));

    // Assert
    expect(matched).toBe(false);
  });
});
