import type {
  LanguageModelV2CallOptions,
  LanguageModelV2StreamPart,
} from '@ai-sdk/provider';
import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import { APICallError, generateText, streamText } from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import {
  chunksToText,
  createMockModel,
  createMockStreamingModel,
  type LanguageModelV2GenerateFn,
  type LanguageModelV2StreamFn,
} from '../test-utils.js';
import type { LanguageModelV2Generate } from '../types.js';
import { requestTimeout } from './request-timeout.js';

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelV2Generate = {
  finishReason: 'stop',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const timeoutError = async (opts: LanguageModelV2CallOptions) => {
  // Check if abortSignal is aborted and throw appropriate error
  if (opts.abortSignal?.aborted) {
    throw new DOMException('The operation was aborted', 'AbortError');
  }

  // Listen for abort event during the async operation
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      resolve(mockResult);
    }, 1_000);

    opts.abortSignal?.addEventListener('abort', () => {
      clearTimeout(timeout);
      reject(new DOMException('The operation was aborted', 'AbortError'));
    });
  });
};

const genericError = new APICallError({
  message: 'Some other error',
  url: '',
  requestBodyValues: {},
  statusCode: 500,
  responseHeaders: {},
  responseBody: '{"error": {"message": "Internal server error"}}',
  isRetryable: true,
  data: {
    error: {
      message: 'Internal server error',
    },
  },
});

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

describe('requestTimeout', () => {
  describe('generateText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = createMockModel(mockResult);
      const retryModel = createMockModel(mockResult);

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [requestTimeout(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should fallback in case of timeout error', async () => {
      // Arrange
      const baseModel = createMockModel(
        timeoutError as LanguageModelV2GenerateFn,
      );
      const retryModel = createMockModel(mockResult);

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [requestTimeout(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        abortSignal: AbortSignal.timeout(100), // Very short timeout to trigger timeout
      });

      // Assert
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(retryModel.doGenerateCalls.length).toBe(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should not fallback for non-timeout errors', async () => {
      // Arrange
      const baseModel = createMockModel(genericError);
      const retryModel = createMockModel(mockResult);

      // Act
      const result = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [requestTimeout(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrowError(APICallError);
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(retryModel.doGenerateCalls.length).toBe(0);
    });
  });

  describe('streamText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
      });
      const retryModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [requestTimeout(retryModel)],
        }),
        prompt: 'Hello!',
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStreamCalls.length).toBe(1);
      expect(retryModel.doStreamCalls.length).toBe(0);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    // TODO needs to read the first chunk to get the abort error
    it.todo('should fallback in case of timeout error', async () => {
      // Arrange
      const baseModel = createMockStreamingModel(
        timeoutError as LanguageModelV2StreamFn,
      );
      const retryModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [requestTimeout(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        abortSignal: AbortSignal.timeout(100), // Very short timeout to trigger timeout
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStreamCalls.length).toBe(1);
      expect(retryModel.doStreamCalls.length).toBe(1);
      expect(error).toBeUndefined();
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "type": "abort",
          },
        ]
      `);
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should not fallback for non-timeout errors', async () => {
      // Arrange
      const baseModel = createMockStreamingModel(genericError);
      const retryModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [requestTimeout(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      // Assert
      expect(baseModel.doStreamCalls.length).toBe(1);
      expect(retryModel.doStreamCalls.length).toBe(0);
      expect(error).toBeDefined();
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "error": [AI_APICallError: Some other error],
            "type": "error",
          },
        ]
      `);
    });
  });
});
