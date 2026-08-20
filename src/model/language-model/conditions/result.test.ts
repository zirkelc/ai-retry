import { Iterables } from 'ai-test-kit';
import { generateText, streamText } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  chunksToText,
  createRetryableModel,
  Language,
  MockLanguageModel,
  mockResultText,
} from '../../../internal/test-utils.js';
import type {
  LanguageModelResult,
  LanguageModelStreamPart,
} from '../../../types.js';
import { result } from './index.js';

const flaggedText = 'flagged content';

const containsFlagged = (res: LanguageModelResult): boolean =>
  res.content.some(
    (part) => part.type === 'text' && part.text.includes('flagged'),
  );

describe('result', () => {
  describe('generateText', () => {
    it('should switch when predicate matches the result content', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from(flaggedText);
      const retryModel = MockLanguageModel.from(mockResultText);

      // Act
      const out = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [result(containsFlagged).switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(out.text).toBe(mockResultText);
    });

    it('should not switch when predicate misses', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from(mockResultText);
      const retryModel = MockLanguageModel.from(mockResultText);

      // Act
      const out = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [result(containsFlagged).switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(out.text).toBe(mockResultText);
    });
  });

  describe('streamText', () => {
    it('should pass through stream when result conditions cannot fire', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: Language.streamParts(mockResultText),
      });
      const retryModel = MockLanguageModel.from({
        doStream: Language.streamParts(mockResultText),
      });

      // Act
      const out = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [result(containsFlagged).switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });
      const chunks = await Iterables.toArray(out.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(chunksToText(chunks)).toBe(mockResultText);
    });
  });
});
