import { describe, expect, it } from 'vitest';
import {
  buildErrorContext,
  buildResultContext,
  generateEmptyResult,
  generateTextResult,
  MockLanguageModel,
} from '../../internal/test-utils.js';
import type { LanguageModelResult } from '../../types.js';
import { result } from './result.js';

const contentFilterResult: LanguageModelResult = {
  ...generateEmptyResult,
  finishReason: { unified: 'content-filter', raw: undefined },
};

describe('result', () => {
  it(`should run the predicate against the current result`, async () => {
    // Arrange
    const cond = result<MockLanguageModel>((res) =>
      res.content.some((part) => part.type === 'text' && part.text === 'hi'),
    );

    // Act
    const matched = await cond.evaluate(
      buildResultContext(generateTextResult('hi')),
    );
    const missed = await cond.evaluate(
      buildResultContext(generateTextResult('bye')),
    );

    // Assert
    expect(matched).toBe(true);
    expect(missed).toBe(false);
  });

  it(`should return false on error attempts`, async () => {
    // Arrange
    const cond = result<MockLanguageModel>(() => true);

    // Act
    const matched = await cond.evaluate(buildErrorContext(new Error('boom')));

    // Assert
    expect(matched).toBe(false);
  });

  it(`should support async predicates`, async () => {
    // Arrange
    const cond = result<MockLanguageModel>(async () => Promise.resolve(true));

    // Act
    const matched = await cond.evaluate(buildResultContext());

    // Assert
    expect(matched).toBe(true);
  });

  describe('finishReason', () => {
    it(`should match a single reason`, async () => {
      // Arrange
      const cond = result.finishReason<MockLanguageModel>('content-filter');

      // Act
      const matched = await cond.evaluate(
        buildResultContext(contentFilterResult),
      );
      const missed = await cond.evaluate(
        buildResultContext(generateTextResult('hi')),
      );

      // Assert
      expect(matched).toBe(true);
      expect(missed).toBe(false);
    });

    it(`should match any of several reasons`, async () => {
      // Arrange
      const cond = result.finishReason<MockLanguageModel>(
        'content-filter',
        'length',
      );

      // Act
      const matched = await cond.evaluate(
        buildResultContext(contentFilterResult),
      );

      // Assert
      expect(matched).toBe(true);
    });

    it(`should return false on error attempts`, async () => {
      // Arrange
      const cond = result.finishReason<MockLanguageModel>('content-filter');

      // Act
      const matched = await cond.evaluate(buildErrorContext(new Error('boom')));

      // Assert
      expect(matched).toBe(false);
    });
  });
});
