import { generateText } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  createRetryable,
  generateTextResult,
  MockLanguageModel,
} from '../../internal/test-utils.js';
import type { LanguageModelResult } from '../../types.js';
import { finishReason } from './index.js';

const okText = 'Hello, world!';

const contentFilteredResult: LanguageModelResult = {
  finishReason: { unified: 'content-filter', raw: undefined },
  usage: {
    inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 20, text: 0, reasoning: 0 },
  },
  content: [],
  warnings: [],
};

describe('finishReason', () => {
  describe('generateText', () => {
    it('should switch when finish reason matches', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: contentFilteredResult,
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

      // Act
      const out = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [
            finishReason('content-filter').switch({ model: retryModel }),
          ],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(out.text).toBe(okText);
    });

    it('should not switch when finish reason does not match', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

      // Act
      const out = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [
            finishReason('content-filter').switch({ model: retryModel }),
          ],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(out.text).toBe(okText);
    });

    it('should accept multiple reasons', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: contentFilteredResult,
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

      // Act
      const out = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [
            finishReason('content-filter', 'length').switch({
              model: retryModel,
            }),
          ],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(out.text).toBe(okText);
    });
  });
});
