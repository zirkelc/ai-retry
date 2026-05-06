import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import { generateText, streamText } from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../../../create-retryable-model.js';
import {
  abortError,
  chunksToText,
  generateTextResult,
  MockLanguageModel,
  timeoutError,
} from '../../../test-utils.js';
import type { LanguageModelStreamPart } from '../../../types.js';
import { timeout } from './index.js';

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

describe('timeout', () => {
  describe('generateText', () => {
    it('should switch on TimeoutError', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: timeoutError() });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [timeout().switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(okText);
    });

    it('should not switch on AbortError', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: abortError() });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

      // Act
      const result = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [timeout().switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrow(/aborted/);
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    });
  });

  describe('streamText', () => {
    it('should switch on TimeoutError', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doStream: timeoutError() });
      const retryModel = new MockLanguageModel({
        doStream: { stream: convertArrayToReadableStream(okStream) },
      });
      let streamError: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [timeout().switch({ model: retryModel })],
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
