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
import { serviceOverloaded } from './service-overloaded.js';

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelV2Generate = {
  finishReason: 'stop',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const overloadedError = new APICallError({
  message:
    'Overloaded: The server is currently overloaded. Please try again later.',
  url: '',
  requestBodyValues: {},
  statusCode: 529,
  responseHeaders: {},
  responseBody: '',
  isRetryable: false,
  data: {},
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

describe('serviceOverloaded', () => {
  describe('generateText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = createMockModel(mockResult);
      const retryModel = createMockModel(mockResult);

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should retry for status 529', async () => {
      // Arrange
      const baseModel = createMockModel(overloadedError);
      const retryModel = createMockModel(mockResult);

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(retryModel.doGenerateCalls.length).toBe(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should not retry for status 200', async () => {
      // Arrange
      const baseModel = createMockModel(mockResult);
      const retryModel = createMockModel(mockResult);

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(retryModel.doGenerateCalls.length).toBe(0);
      expect(result.text).toBe(mockResultText);
    });

    it('should not retry if no matches', async () => {
      // Arrange
      const baseModel = createMockModel(
        new APICallError({
          message: 'Some other error',
          url: '',
          requestBodyValues: {},
          statusCode: 400,
          responseHeaders: {},
          responseBody: '{}',
          isRetryable: false,
          data: {
            error: {
              message: 'Some other error',
              type: null,
              param: 'prompt',
              code: 'other_error',
            },
          },
        }),
      );

      const retryModel = createMockModel(mockResult);

      // Act
      const result = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
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
          retries: [serviceOverloaded(retryModel)],
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

    it('should retry for status 529', async () => {
      // Arrange
      const baseModel = createMockStreamingModel(overloadedError);
      const retryModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
        }),
        prompt: 'Hello!',
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

    it('should not retry if no matches', async () => {
      // Arrange
      const baseModel = createMockStreamingModel(
        new APICallError({
          message: 'Some other error',
          url: '',
          requestBodyValues: {},
          statusCode: 400,
          responseHeaders: {},
          responseBody: '{}',
          isRetryable: false,
          data: {
            error: {
              message: 'Some other error',
              type: null,
              param: 'prompt',
              code: 'other_error',
            },
          },
        }),
      );

      const retryModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [serviceOverloaded(retryModel)],
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
