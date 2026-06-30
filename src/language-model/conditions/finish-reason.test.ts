import { generateText } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  MockLanguageModel,
  createRetryableModel,
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
      const baseModel = MockLanguageModel.from({
        doGenerate: contentFilteredResult,
      });
      const retryModel = MockLanguageModel.from(okText);

      // Act
      const out = await generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from(okText);
      const retryModel = MockLanguageModel.from(okText);

      // Act
      const out = await generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doGenerate: contentFilteredResult,
      });
      const retryModel = MockLanguageModel.from(okText);

      // Act
      const out = await generateText({
        model: createRetryableModel({
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
