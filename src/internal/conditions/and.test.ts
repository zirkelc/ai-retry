import { describe, expect, it } from 'vitest';
import { buildErrorContext, MockLanguageModel } from '../test-utils.js';
import { and } from './and.js';
import { Condition } from './condition.js';

const ctx = buildErrorContext(new Error('boom'));

const truthy = new Condition<MockLanguageModel>(() => true);
const falsy = new Condition<MockLanguageModel>(() => false);

describe('and', () => {
  it(`should return true when all conditions match`, async () => {
    // Act
    const matched = await and(truthy, truthy).evaluate(ctx);

    // Assert
    expect(matched).toBe(true);
  });

  it(`should return false when any condition fails`, async () => {
    // Act
    const matched = await and(truthy, falsy, truthy).evaluate(ctx);

    // Assert
    expect(matched).toBe(false);
  });

  it(`should return true when given no conditions (vacuous truth)`, async () => {
    // Act
    const matched = await and<MockLanguageModel>().evaluate(ctx);

    // Assert
    expect(matched).toBe(true);
  });

  it(`should short-circuit on the first miss`, async () => {
    // Arrange
    let visited = 0;
    const tracking = new Condition<MockLanguageModel>(() => {
      visited += 1;
      return true;
    });

    // Act
    await and(falsy, tracking).evaluate(ctx);

    // Assert
    expect(visited).toBe(0);
  });
});
