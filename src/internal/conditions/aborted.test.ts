import { describe, expect, it } from 'vitest';
import { Errors, MockLanguageModel, buildErrorContext } from '../test-utils.js';
import { createErrorAPI } from './error.js';

const { aborted } = createErrorAPI<MockLanguageModel>();

describe('aborted', () => {
  it(`should match AbortError`, async () => {
    // Arrange
    const cond = aborted<MockLanguageModel>();

    // Act
    const matched = await cond.evaluate(buildErrorContext(Errors.abort()));

    // Assert
    expect(matched).toBe(true);
  });

  it(`should not match TimeoutError`, async () => {
    // Arrange
    const cond = aborted<MockLanguageModel>();

    // Act
    const matched = await cond.evaluate(buildErrorContext(Errors.timeout()));

    // Assert
    expect(matched).toBe(false);
  });

  it(`should not match a plain Error`, async () => {
    // Arrange
    const cond = aborted<MockLanguageModel>();

    // Act
    const matched = await cond.evaluate(buildErrorContext(new Error('boom')));

    // Assert
    expect(matched).toBe(false);
  });
});
