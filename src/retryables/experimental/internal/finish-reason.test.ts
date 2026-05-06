import { generateText } from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../../../create-retryable-model.js';
import {
  buildResultContext,
  generateEmptyResult,
  generateTextResult,
  MockLanguageModel,
} from '../../../test-utils.js';
import type { LanguageModelGenerate } from '../../../types.js';
import { finishReason } from './finish-reason.js';

const contentFilterResult: LanguageModelGenerate = {
  ...generateEmptyResult,
  finishReason: { unified: 'content-filter', raw: undefined },
};

describe('finishReason', () => {
  it(`should match a single reason`, async () => {
    // Arrange
    const cond = finishReason<MockLanguageModel>('content-filter');

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
    const cond = finishReason<MockLanguageModel>('content-filter', 'length');

    // Act
    const matched = await cond.evaluate(
      buildResultContext(contentFilterResult),
    );

    // Assert
    expect(matched).toBe(true);
  });

  it(`should switch in createRetryable when finishReason hits`, async () => {
    // Arrange
    const baseModel = new MockLanguageModel({
      doGenerate: contentFilterResult,
    });
    const fallback = new MockLanguageModel({
      doGenerate: generateTextResult('hello'),
    });

    // Act
    const result = await generateText({
      model: createRetryable({
        model: baseModel,
        retries: [finishReason('content-filter').switch({ model: fallback })],
      }),
      prompt: 'hi',
      maxRetries: 0,
    });

    // Assert
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(fallback.doGenerate).toHaveBeenCalledTimes(1);
    expect(result.text).toBe('hello');
  });
});
