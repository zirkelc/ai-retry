import { generateText } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  Language,
  MockLanguageModel,
  buildResultContext,
  contentFilterResult,
  createRetryableModel,
} from '../test-utils.js';
import { createResultAPI } from './result.js';

const { finishReason } = createResultAPI<MockLanguageModel>();

describe('finishReason', () => {
  it(`should match a single reason`, async () => {
    // Arrange
    const cond = finishReason<MockLanguageModel>('content-filter');

    // Act
    const matched = await cond.evaluate(
      buildResultContext(contentFilterResult),
    );
    const missed = await cond.evaluate(
      buildResultContext(Language.result('hi')),
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

  it(`should switch in createRetryableModel when finishReason hits`, async () => {
    // Arrange
    const baseModel = MockLanguageModel.from({
      doGenerate: contentFilterResult,
    });
    const fallback = MockLanguageModel.from('hello');

    // Act
    const result = await generateText({
      model: createRetryableModel({
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
