import { Iterables } from 'ai-test-kit';
import { APICallError, generateText, streamText } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  Errors,
  Language,
  MockLanguageModel,
  chunksToText,
  createRetryableModel,
  mockResultText,
  mockStreamChunks,
} from '../../internal/test-utils.js';
import { error } from './index.js';

describe('error', () => {
  describe('generateText', () => {
    it('should switch when predicate matches the error', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: Errors.from({ statusCode: 418, message: 'teapot' }),
      });
      const retryModel = MockLanguageModel.from(mockResultText);

      // Act
      const result = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [
            error(
              (e) => APICallError.isInstance(e) && e.statusCode === 418,
            ).switch({ model: retryModel }),
          ],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should not switch when predicate misses', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: Errors.from({ statusCode: 500 }),
      });
      const retryModel = MockLanguageModel.from(mockResultText);

      // Act
      const result = generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [
            error(
              (e) => APICallError.isInstance(e) && e.statusCode === 418,
            ).switch({ model: retryModel }),
          ],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrow(APICallError);
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    });
  });

  describe('streamText', () => {
    it('should switch when predicate matches the error', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: Errors.from({ statusCode: 418 }),
      });
      const retryModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      let streamError: unknown;

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [
            error(
              (e) => APICallError.isInstance(e) && e.statusCode === 418,
            ).switch({ model: retryModel }),
          ],
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

  describe('error.isRetryable', () => {
    it('should retry the same model when isRetryable=true', async () => {
      // Arrange
      let attempt = 0;
      const baseModel = MockLanguageModel.from({
        doGenerate: async () => {
          attempt++;
          if (attempt === 1) {
            throw Errors.from({ statusCode: 503, isRetryable: true });
          }
          return Language.result(mockResultText);
        },
      });

      // Act
      const result = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [error.isRetryable(true).retry()],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(2);
      expect(result.text).toBe(mockResultText);
    });

    it('should switch when isRetryable=false', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: Errors.from({ statusCode: 400, isRetryable: false }),
      });
      const retryModel = MockLanguageModel.from(mockResultText);

      // Act
      const result = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [error.isRetryable(false).switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });
  });

  describe('error.statusCode', () => {
    it('should switch on matching numeric status', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: Errors.from({ statusCode: 503 }),
      });
      const retryModel = MockLanguageModel.from(mockResultText);

      // Act
      const result = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [error.statusCode(503).switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should switch on regex matching the status', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: Errors.from({ statusCode: 502 }),
      });
      const retryModel = MockLanguageModel.from(mockResultText);

      // Act
      const result = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [error.statusCode(/^5\d\d$/).switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });
  });

  describe('error.message', () => {
    it('should switch on case-insensitive substring match', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: Errors.from({ message: 'Service Overloaded' }),
      });
      const retryModel = MockLanguageModel.from(mockResultText);

      // Act
      const result = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [error.message('overloaded').switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });
  });
});
