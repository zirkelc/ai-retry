import type { LanguageModelV2StreamPart } from '@ai-sdk/provider';
import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import { generateText, streamText } from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import { chunksToText, MockLanguageModel } from '../test-utils.js';
import type { LanguageModelV2Generate } from '../types.js';
import { finishReason } from './finish-reason.js';

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelV2Generate = {
  finishReason: 'stop',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const unknownFinishResult: LanguageModelV2Generate = {
  finishReason: 'unknown',
  usage: { inputTokens: 10, outputTokens: 0, totalTokens: 10 },
  content: [],
  warnings: [],
};

const errorFinishResult: LanguageModelV2Generate = {
  finishReason: 'error',
  usage: { inputTokens: 10, outputTokens: 0, totalTokens: 10 },
  content: [],
  warnings: [],
};

const lengthFinishResult: LanguageModelV2Generate = {
  finishReason: 'length',
  usage: { inputTokens: 10, outputTokens: 100, totalTokens: 110 },
  content: [{ type: 'text', text: 'Some truncated content...' }],
  warnings: [],
};

const mockStreamChunks: LanguageModelV2StreamPart[] = [
  {
    type: 'stream-start',
    warnings: [],
  },
  {
    type: 'response-metadata',
    id: 'id-0',
    modelId: 'mock-model-id',
    timestamp: new Date(0),
  },
  { type: 'text-start', id: '1' },
  { type: 'text-delta', id: '1', delta: 'Hello' },
  { type: 'text-delta', id: '1', delta: ', ' },
  { type: 'text-delta', id: '1', delta: 'world!' },
  { type: 'text-end', id: '1' },
  {
    type: 'finish',
    finishReason: 'stop',
    usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  },
];

const unknownFinishChunks: LanguageModelV2StreamPart[] = [
  {
    type: 'stream-start',
    warnings: [],
  },
  {
    type: 'finish',
    finishReason: 'unknown',
    usage: { inputTokens: 10, outputTokens: 0, totalTokens: 10 },
  },
];

const errorFinishChunks: LanguageModelV2StreamPart[] = [
  {
    type: 'stream-start',
    warnings: [],
  },
  {
    type: 'finish',
    finishReason: 'error',
    usage: { inputTokens: 10, outputTokens: 0, totalTokens: 10 },
  },
];

describe('finishReason', () => {
  describe('generateText', () => {
    it('should succeed without retry on normal stop finish reason', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: mockResult });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [finishReason(retryModel, { reasons: 'unknown' })],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(result.text).toBe(mockResultText);
      expect(result.finishReason).toBe('stop');
    });

    it('should retry on unknown finish reason', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: unknownFinishResult,
      });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [finishReason(retryModel, { reasons: 'unknown' })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
      expect(result.finishReason).toBe('stop');
    });

    it('should retry on error finish reason', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: errorFinishResult,
      });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [finishReason(retryModel, { reasons: 'error' })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should retry on multiple finish reasons', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: errorFinishResult,
      });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [
            finishReason(retryModel, {
              reasons: ['unknown', 'error', 'other'],
            }),
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

    it('should not retry if finish reason does not match', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: lengthFinishResult,
      });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [finishReason(retryModel, { reasons: 'unknown' })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(result.finishReason).toBe('length');
    });

    it('should respect maxAttempts option', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: unknownFinishResult,
      });
      const retryModel = new MockLanguageModel({
        doGenerate: unknownFinishResult,
      });
      const fallbackModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [
            finishReason(retryModel, { reasons: 'unknown', maxAttempts: 1 }),
            finishReason(fallbackModel, { reasons: 'unknown', maxAttempts: 1 }),
          ],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should work with single reason as string', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: unknownFinishResult,
      });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [finishReason(retryModel, { reasons: 'unknown' })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });
  });

  describe('streamText', () => {
    it('should succeed without retry on normal stop finish reason', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      const retryModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [finishReason(retryModel, { reasons: 'unknown' })],
        }),
        prompt: 'Hello!',
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should retry on unknown finish reason when no content streamed', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(unknownFinishChunks),
        },
      });
      const retryModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [finishReason(retryModel, { reasons: 'unknown' })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should retry on error finish reason when no content streamed', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(errorFinishChunks),
        },
      });
      const retryModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [finishReason(retryModel, { reasons: 'error' })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should retry on multiple finish reasons', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(unknownFinishChunks),
        },
      });
      const retryModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [
            finishReason(retryModel, {
              reasons: ['unknown', 'error', 'other'],
            }),
          ],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should not retry if content was already streamed', async () => {
      // Arrange - stream with content followed by unknown finish
      const streamWithContentThenUnknown: LanguageModelV2StreamPart[] = [
        {
          type: 'stream-start',
          warnings: [],
        },
        { type: 'text-start', id: '1' },
        { type: 'text-delta', id: '1', delta: 'Some content' },
        { type: 'text-end', id: '1' },
        {
          type: 'finish',
          finishReason: 'unknown',
          usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
        },
      ];

      const baseModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(streamWithContentThenUnknown),
        },
      });
      const retryModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [finishReason(retryModel, { reasons: 'unknown' })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert - should NOT retry because content was already streamed
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe('Some content');
    });

    it('should not retry if finish reason does not match', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(unknownFinishChunks),
        },
      });
      const retryModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [finishReason(retryModel, { reasons: 'error' })], // Looking for 'error', but gets 'unknown'
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          error = data.error;
        },
      });

      await convertAsyncIterableToArray(result.fullStream);

      // Assert - should not retry because finish reason doesn't match
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(error).toBeUndefined();
    });
  });
});
