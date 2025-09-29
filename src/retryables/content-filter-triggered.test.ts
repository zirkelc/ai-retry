import type { LanguageModelV2StreamPart } from '@ai-sdk/provider';
import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import {
  APICallError,
  generateText,
  NoObjectGeneratedError,
  streamText,
} from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import {
  chunksToText,
  createMockModel,
  createMockStreamingModel,
} from '../test-utils.js';
import type { LanguageModelV2Generate } from '../types.js';
import { contentFilterTriggered } from './content-filter-triggered.js';

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelV2Generate = {
  finishReason: 'stop',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const contentFilterResult: LanguageModelV2Generate = {
  finishReason: 'content-filter',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [],
  warnings: [],
};

const apiCallError = new APICallError({
  message:
    "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766",
  url: '',
  requestBodyValues: {},
  statusCode: 400,
  responseHeaders: {},
  responseBody:
    '{"error":{"message":"The response was filtered due to the prompt triggering Azure OpenAI\'s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766","type":null,"param":"prompt","code":"content_filter","status":400,"innererror":{"code":"ResponsibleAIPolicyViolation","content_filter_result":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":true,"severity":"high"},"violence":{"filtered":false,"severity":"safe"}}}}}',
  isRetryable: false,
  data: {
    error: {
      message:
        "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766",
      type: null,
      param: 'prompt',
      code: 'content_filter',
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

describe('contentFilterTriggered', () => {
  describe('generateText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = createMockModel(mockResult);
      const retryModel = createMockModel(mockResult);

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [contentFilterTriggered(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should retry in case of content filter result', async () => {
      // Arrange
      const baseModel = createMockModel(contentFilterResult);
      const retryModel = createMockModel(mockResult);

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [contentFilterTriggered(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(retryModel.doGenerateCalls.length).toBe(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should retry in case of content filter error', async () => {
      // Arrange
      const baseModel = createMockModel(apiCallError);
      const retryModel = createMockModel(mockResult);

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [contentFilterTriggered(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerateCalls.length).toBe(1);
      expect(retryModel.doGenerateCalls.length).toBe(1);
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
          retries: [contentFilterTriggered(retryModel)],
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
          retries: [contentFilterTriggered(retryModel)],
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

    it('should retry in case of content filter error', async () => {
      // Arrange
      const baseModel = createMockStreamingModel(apiCallError);
      const retryModel = createMockStreamingModel({
        stream: convertArrayToReadableStream(mockStreamChunks),
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [contentFilterTriggered(retryModel)],
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
          retries: [contentFilterTriggered(retryModel)],
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
      expect(error).toBeInstanceOf(APICallError);
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
