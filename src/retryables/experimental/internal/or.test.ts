import { describe, expect, it } from 'vitest';
import { buildErrorContext, MockLanguageModel } from '../../../test-utils.js';
import { Condition } from './condition.js';
import { or } from './or.js';

const ctx = buildErrorContext(new Error('boom'));

const truthy = new Condition<MockLanguageModel>(() => true);
const falsy = new Condition<MockLanguageModel>(() => false);

describe('or', () => {
  it(`should return true when any condition matches`, async () => {
    // Act
    const matched = await or(falsy, truthy, falsy).evaluate(ctx);

    // Assert
    expect(matched).toBe(true);
  });

  it(`should return false when no condition matches`, async () => {
    // Act
    const matched = await or(falsy, falsy).evaluate(ctx);

    // Assert
    expect(matched).toBe(false);
  });

  it(`should return false when given no conditions`, async () => {
    // Act
    const matched = await or<MockLanguageModel>().evaluate(ctx);

    // Assert
    expect(matched).toBe(false);
  });

  it(`should short-circuit on the first match`, async () => {
    // Arrange
    let visited = 0;
    const tracking = new Condition<MockLanguageModel>(() => {
      visited += 1;
      return false;
    });

    // Act
    await or(truthy, tracking).evaluate(ctx);

    // Assert
    expect(visited).toBe(0);
  });
});
