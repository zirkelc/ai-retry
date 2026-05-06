import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import { generateText, streamText } from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../../../create-retryable-model.js';
import {
  chunksToText,
  generateTextResult,
  MockLanguageModel,
} from '../../../test-utils.js';
import type {
  LanguageModelGenerate,
  LanguageModelStreamPart,
} from '../../../types.js';
import { result } from './index.js';

const okText = 'Hello, world!';
const flaggedText = 'flagged content';

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

const containsFlagged = (res: LanguageModelGenerate): boolean =>
  res.content.some(
    (part) => part.type === 'text' && part.text.includes('flagged'),
  );

describe('result', () => {
  describe('generateText', () => {
    it('should switch when predicate matches the result content', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: generateTextResult(flaggedText),
      });
      const retryModel = new MockLanguageModel({
        doGenerate: generateTextResult(okText),
      });

      // Act
      const out = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [result(containsFlagged).switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(out.text).toBe(okText);
    });

    it('should not switch when predicate misses', async () => {
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
          retries: [result(containsFlagged).switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(out.text).toBe(okText);
    });
  });

  describe('streamText', () => {
    it('should pass through stream when result conditions cannot fire', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: { stream: convertArrayToReadableStream(okStream) },
      });
      const retryModel = new MockLanguageModel({
        doStream: { stream: convertArrayToReadableStream(okStream) },
      });

      // Act
      const out = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [result(containsFlagged).switch({ model: retryModel })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });
      const chunks = await convertAsyncIterableToArray(out.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(chunksToText(chunks)).toBe(okText);
    });
  });
});
