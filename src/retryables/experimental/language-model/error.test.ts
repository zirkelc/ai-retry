import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import { APICallError, generateText, streamText } from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../../../create-retryable-model.js';
import {
  apiError,
  chunksToText,
  generateTextResult,
  MockLanguageModel,
} from '../../../test-utils.js';
import type { LanguageModelStreamPart } from '../../../types.js';
import { error } from './index.js';

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

describe('error', () => {
  describe('generateText', () => {
    it('should switch when predicate matches the error', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: apiError({ statusCode: 418, message: 'teapot' }),
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

      // Act
      const result = await generateText({
        model: createRetryable({
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
      expect(result.text).toBe(okText);
    });

    it('should not switch when predicate misses', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: apiError({ statusCode: 500 }),
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

      // Act
      const result = generateText({
        model: createRetryable({
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
      const baseModel = new MockLanguageModel({
        doStream: apiError({ statusCode: 418 }),
      });
      const retryModel = new MockLanguageModel({
        doStream: { stream: convertArrayToReadableStream(okStream) },
      });
      let streamError: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
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
      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
      expect(streamError).toBeUndefined();
      expect(chunksToText(chunks)).toBe(okText);
    });
  });

  describe('error.isRetryable', () => {
    it('should retry the same model when isRetryable=true', async () => {
      // Arrange
      let attempt = 0;
      const baseModel = new MockLanguageModel({
        doGenerate: async () => {
          attempt++;
          if (attempt === 1) {
            throw apiError({ statusCode: 503, isRetryable: true });
          }
          return generateTextResult(okText);
        },
      });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [error.isRetryable(true).retry()],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(2);
      expect(result.text).toBe(okText);
    });

    it('should switch when isRetryable=false', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: apiError({ statusCode: 400, isRetryable: false }),
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [error.isRetryable(false).switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(okText);
    });
  });

  describe('error.statusCode', () => {
    it('should switch on matching numeric status', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: apiError({ statusCode: 503 }),
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [error.statusCode(503).switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(okText);
    });

    it('should switch on regex matching the status', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: apiError({ statusCode: 502 }),
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [error.statusCode(/^5\d\d$/).switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(okText);
    });
  });

  describe('error.message', () => {
    it('should switch on case-insensitive substring match', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: apiError({ message: 'Service Overloaded' }),
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [error.message('overloaded').switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(okText);
    });
  });
});
