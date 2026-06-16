import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import { APICallError, generateText, streamText } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  apiError,
  chunksToText,
  createRetryableModel,
  generateTextResult,
  MockLanguageModel,
} from '../../internal/test-utils.js';
import type { LanguageModelStreamPart } from '../../types.js';
import { httpStatus } from './index.js';

const okText = 'Hello, world!';

const okStream: LanguageModelStreamPart[] = [
  { type: 'stream-start', warnings: [] },
  { type: 'text-start', id: '1' },
  { type: 'text-delta', id: '1', delta: okText },
  { type: 'text-end', id: '1' },
  {
    type: 'finish',
    finishReason: { unified: 'stop', raw: undefined },
    usage: {
      inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 20, text: 0, reasoning: 0 },
    },
  },
];

describe('httpStatus', () => {
  describe('generateText', () => {
    it('should switch on numeric status match', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: apiError({ statusCode: 529 }),
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

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
      expect(result.text).toBe(okText);
    });

    it('should switch on message substring match', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: apiError({
          statusCode: 200,
          message: 'Service Overloaded',
        }),
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

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
      expect(result.text).toBe(okText);
    });

    it('should switch on regex match', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: apiError({ statusCode: 502 }),
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

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
      expect(result.text).toBe(okText);
    });

    it('should not switch when no pattern matches', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: apiError({ statusCode: 400, message: 'bad request' }),
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

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
      const baseModel = new MockLanguageModel({
        doStream: apiError({ statusCode: 529 }),
      });
      const retryModel = new MockLanguageModel({
        doStream: { stream: convertArrayToReadableStream(okStream) },
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
      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
      expect(streamError).toBeUndefined();
      expect(chunksToText(chunks)).toBe(okText);
    });
  });
});
