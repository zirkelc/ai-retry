import type { LanguageModelV2StreamPart } from '@ai-sdk/provider';
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
} from '../test-utils.js';
import type { LanguageModelV2Generate } from '../types.js';
import { requestNotRetryable } from './request-not-retryable.js';

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelV2Generate = {
  finishReason: 'stop',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const nonRetryableError = new APICallError({
  message: 'Invalid API key',
  url: '',
  requestBodyValues: {},
  statusCode: 401,
  responseHeaders: {},
  responseBody:
    '{"error": {"message": "Invalid API key", "code": "invalid_api_key"}}',
  isRetryable: false,
  data: {
    error: {
      message: 'Invalid API key',
      code: 'invalid_api_key',
    },
  },
});

const retryableError = new APICallError({
  message: 'Rate limit exceeded',
  url: '',
  requestBodyValues: {},
  statusCode: 429,
  responseHeaders: {},
  responseBody:
    '{"error": {"message": "Rate limit exceeded", "code": "rate_limit_exceeded"}}',
  isRetryable: true,
  data: {
    error: {
      message: 'Rate limit exceeded',
      code: 'rate_limit_exceeded',
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

describe('requestNotRetryable', () => {
  describe('generateText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = createMockModel(mockResult);
      const retryModel = createMockModel(mockResult);

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should fallback in case of non-retryable error', async () => {
      // Arrange
      const baseModel = createMockModel(nonRetryableError);
      const retryModel = createMockModel(mockResult);

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(retryModel.doGenerateCalls.length).toBe(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should not fallback for retryable error', async () => {
      // Arrange
      const baseModel = createMockModel(retryableError);
      const retryModel = createMockModel(mockResult);

      // Act
      const result = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrowError(APICallError);
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(retryModel.doGenerateCalls.length).toBe(0);
    });

    it('should not fallback for non AI SDK errors', async () => {
      // Arrange
      const genericError = new Error('Some generic error');
      const baseModel = createMockModel(genericError);
      const retryModel = createMockModel(mockResult);

      // Act
      const result = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrowError(Error);
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
          retries: [requestNotRetryable(retryModel)],
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

    it('should fallback in case of non-retryable error', async () => {
      // Arrange
      const baseModel = createMockStreamingModel(nonRetryableError);
      const retryModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
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
      expect(retryModel.doStreamCalls.length).toBe(1);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should not fallback for retryable error', async () => {
      // Arrange
      const baseModel = createMockStreamingModel(retryableError);
      const retryModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
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
            "error": [AI_APICallError: Rate limit exceeded],
            "type": "error",
          },
        ]
      `);
    });

    it('should not fallback for non AI SDK errors', async () => {
      // Arrange
      const genericError = new Error('Some generic error');
      const baseModel = createMockStreamingModel(genericError);
      const retryModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [requestNotRetryable(retryModel)],
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
            "error": [Error: Some generic error],
            "type": "error",
          },
        ]
      `);
    });
  });
});
