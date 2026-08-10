import { Iterables } from 'ai-test-kit';
import { APICallError, generateText, streamText } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  chunksToText,
  createRetryableModel,
  Errors,
  Language,
  MockLanguageModel,
  mockResultText,
} from '../../../internal/test-utils.js';
import type { LanguageModelStreamPart } from '../../../types.js';
import { httpStatus } from './index.js';

describe('httpStatus', () => {
  describe('generateText', () => {
    it('should switch on numeric status match', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: Errors.from({ statusCode: 529 }),
      });
      const retryModel = MockLanguageModel.from(mockResultText);

      // Act
      const result = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [httpStatus(529).switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should switch on message substring match', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: Errors.from({
          statusCode: 200,
          message: 'Service Overloaded',
        }),
      });
      const retryModel = MockLanguageModel.from(mockResultText);

      // Act
      const result = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [httpStatus('overloaded').switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should switch on regex match', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: Errors.from({ statusCode: 502 }),
      });
      const retryModel = MockLanguageModel.from(mockResultText);

      // Act
      const result = await generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [httpStatus(/^5\d\d$/).switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should not switch when no pattern matches', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doGenerate: Errors.from({ statusCode: 400, message: 'bad request' }),
      });
      const retryModel = MockLanguageModel.from(mockResultText);

      // Act
      const result = generateText({
        model: createRetryableModel({
          model: baseModel,
          retries: [
            httpStatus(529, 'overloaded').switch({ model: retryModel }),
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
    it('should switch on numeric status match', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: Errors.from({ statusCode: 529 }),
      });
      const retryModel = MockLanguageModel.from({
        doStream: Language.streamParts(mockResultText),
      });
      let streamError: unknown;

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [httpStatus(529).switch({ model: retryModel })],
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
