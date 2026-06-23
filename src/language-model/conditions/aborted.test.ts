import { Iterables } from 'ai-test-kit';
import { generateText, streamText } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  Errors,
  MockLanguageModel,
  chunksToText,
  createRetryableModel,
  mockResultText,
  mockStreamChunks,
} from '../../internal/test-utils.js';
import { aborted } from './index.js';

describe('aborted', () => {
  describe('generateText', () => {
    it('should switch on AbortError with a fresh deadline', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: Errors.abort() });
      const retryModel = MockLanguageModel.from(mockResultText);

      // Act
      const result = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [aborted().switch({ model: retryModel, timeout: 60_000 })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should not switch on TimeoutError', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: Errors.timeout(),
      });
      const retryModel = MockLanguageModel.from(mockResultText);

      // Act
      const result = generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [aborted().switch({ model: retryModel, timeout: 60_000 })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrow(/timed out/);
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    });
  });

  describe('streamText', () => {
    it('should switch on AbortError with a fresh deadline', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doStream: Errors.abort() });
      const retryModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      let streamError: unknown;

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [aborted().switch({ model: retryModel, timeout: 60_000 })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          streamError = data.error;
        },
      });
      const chunks = await Iterables.toArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
      expect(streamError).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });
  });
});
