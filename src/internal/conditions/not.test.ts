import { describe, expect, it } from 'vitest';
import {
  buildErrorContext,
  falsy,
  MockLanguageModel,
  truthy,
} from '../test-utils.js';
import { Condition } from './condition.js';
import { not } from './not.js';

const ctx = buildErrorContext(new Error('boom'));

describe('not', () => {
  it(`should invert a matching condition`, async () => {
    // Act
    const matched = await not(truthy).evaluate(ctx);

    // Assert
    expect(matched).toBe(false);
  });

  it(`should invert a non-matching condition`, async () => {
    // Act
    const matched = await not(falsy).evaluate(ctx);

    // Assert
    expect(matched).toBe(true);
  });

  it(`should be involutive (not(not(c)) === c)`, async () => {
    // Act
    const matched = await not(not(truthy)).evaluate(ctx);

    // Assert
    expect(matched).toBe(true);
  });
});
