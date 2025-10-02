import { anthropic } from '@ai-sdk/anthropic';
import type { LanguageModelV2StreamPart } from '@ai-sdk/provider';
import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
  createTestServer,
} from '@ai-sdk/provider-utils/test';
import { APICallError, generateText, streamText } from 'ai';
import { describe, expect, it, vi } from 'vitest';
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
  message: 'Overloaded',
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

    describe('anthropic', () => {
      process.env.ANTHROPIC_API_KEY = 'test';

      const server = createTestServer({
        'https://api.anthropic.com/v1/messages': {},
      });

      it('should retry when overloaded error occurs at stream creation', async () => {
        // Arrange
        server.urls['https://api.anthropic.com/v1/messages'].response = {
          type: 'error',
          status: 529,
          body: '{"type":"error","error":{"details":null,"type":"overloaded_error","message":"Overloaded"}}',
        };

        const baseModel = anthropic('claude-sonnet-4-20250514');
        const retryModel = createMockStreamingModel({
          stream: convertArrayToReadableStream(mockStreamChunks),
        });

        const baseModelSpy = vi.spyOn(baseModel, 'doStream');

        // Act
        const result = streamText({
          model: createRetryable({
            model: baseModel,
            retries: [serviceOverloaded(retryModel)],
          }),
          prompt: 'Hello!',
        });

        const chunks = await convertAsyncIterableToArray(result.fullStream);

        // Assert
        expect(baseModelSpy).toHaveBeenCalledTimes(1);
        expect(retryModel.doStreamCalls.length).toBe(1);
        expect(chunks).toMatchInlineSnapshot(`
          [
            {
              "type": "start",
            },
            {
              "request": {},
              "type": "start-step",
              "warnings": [],
            },
            {
              "id": "1",
              "type": "text-start",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "Hello",
              "type": "text-delta",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": ", ",
              "type": "text-delta",
            },
            {
              "id": "1",
              "providerMetadata": undefined,
              "text": "world!",
              "type": "text-delta",
            },
            {
              "id": "1",
              "type": "text-end",
            },
            {
              "finishReason": "stop",
              "providerMetadata": undefined,
              "response": {
                "headers": undefined,
                "id": "id-0",
                "modelId": "mock-model-id",
                "timestamp": 1970-01-01T00:00:00.000Z,
              },
              "type": "finish-step",
              "usage": {
                "inputTokens": 10,
                "outputTokens": 20,
                "totalTokens": 30,
              },
            },
            {
              "finishReason": "stop",
              "totalUsage": {
                "cachedInputTokens": undefined,
                "inputTokens": 10,
                "outputTokens": 20,
                "reasoningTokens": undefined,
                "totalTokens": 30,
              },
              "type": "finish",
            },
          ]
        `);
      });

      it('should retry when overloaded error occurs at the stream start', async () => {
        // Arrange
        server.urls['https://api.anthropic.com/v1/messages'].response = {
          type: 'stream-chunks',
          chunks: [
            `data: {"type":"error","error":{"details":null,"type":"overloaded_error","message":"Overloaded"}}\n\n`,
          ],
        };

        const baseModel = anthropic('claude-sonnet-4-20250514');
        const retryModel = createMockStreamingModel({
          stream: convertArrayToReadableStream(mockStreamChunks),
        });

        const baseModelSpy = vi.spyOn(baseModel, 'doStream');

        // Act
        const result = streamText({
          model: createRetryable({
            model: baseModel,
            retries: [serviceOverloaded(retryModel)],
          }),
          prompt: 'Hello!',
        });

        const chunks = await convertAsyncIterableToArray(result.textStream);

        // Assert
        expect(baseModelSpy).toHaveBeenCalledTimes(1);
        expect(retryModel.doStreamCalls.length).toBe(1);
        expect(chunks).toMatchInlineSnapshot(`
          [
            "Hello",
            ", ",
            "world!",
          ]
        `);
      });

      it('should NOT retry when overloaded error occurs during streaming', async () => {
        // Arrange
        server.urls['https://api.anthropic.com/v1/messages'].response = {
          type: 'stream-chunks',
          chunks: [
            `data: {"type":"message_start","message":{"id":"msg_01KfpJoAEabmH2iHRRFjQMAG","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":17,"output_tokens":1}}}\n\n`,
            `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n`,
            `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n`,
            `data: {"type":"error","error":{"details":null,"type":"overloaded_error","message":"Overloaded"}}\n\n`,
          ],
        };

        const baseModel = anthropic('claude-sonnet-4-20250514');
        const retryModel = createMockStreamingModel({
          stream: convertArrayToReadableStream(mockStreamChunks),
        });

        const baseModelSpy = vi.spyOn(baseModel, 'doStream');

        // Act
        const result = streamText({
          model: createRetryable({
            model: baseModel,
            retries: [serviceOverloaded(retryModel)],
          }),
          prompt: 'Hello!',
        });

        const chunks = await convertAsyncIterableToArray(result.textStream);

        // Assert
        expect(baseModelSpy).toHaveBeenCalledTimes(1);
        expect(retryModel.doStreamCalls.length).toBe(0);
        expect(chunks).toMatchInlineSnapshot(`
          [
            "Hello",
          ]
        `);
      });
    });
  });
});
