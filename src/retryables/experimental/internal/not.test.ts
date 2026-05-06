import { describe, expect, it } from 'vitest';
import { buildErrorContext, MockLanguageModel } from '../../../test-utils.js';
import { Condition } from './condition.js';
import { not } from './not.js';

const ctx = buildErrorContext(new Error('boom'));

const truthy = new Condition<MockLanguageModel>(() => true);
const falsy = new Condition<MockLanguageModel>(() => false);

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
