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
import {
  chunksToText,
  contentFilterResult,
  createRetryable,
  MockLanguageModel,
  mockResult,
  mockResultText,
  mockStreamChunks,
} from '../internal/test-utils.js';
import type { LanguageModelStreamPart } from '../types.js';
import { contentFilterTriggered } from './content-filter-triggered.js';

const contentFilterChunks: LanguageModelStreamPart[] = [
  {
    type: 'stream-start',
    warnings: [],
  },
  {
    type: 'text-start',
    id: '0',
  },
  {
    type: 'text-end',
    id: '0',
  },
  {
    type: 'finish',
    finishReason: { unified: 'content-filter', raw: undefined },
    usage: {
      inputTokens: { total: 10, noCache: 0, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 20, text: 0, reasoning: 0 },
    },
  },
];

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

describe('contentFilterTriggered', () => {
  describe('generateText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doGenerate: mockResult });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [contentFilterTriggered(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should retry in case of content filter result', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: contentFilterResult,
      });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

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
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should retry in case of content filter error', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: apiCallError,
      });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

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
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should not retry if no matches', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: new APICallError({
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
      });

      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

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
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    });
  });

  describe('streamText', () => {
    it('should succeed without errors', async () => {
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
          retries: [contentFilterTriggered(retryModel)],
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

    it('should retry in case of content filter error', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({ doStream: apiCallError });
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
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should retry when finish part reports content-filter before any content', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(contentFilterChunks),
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
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should not retry if no matches', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: new APICallError({
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
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
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
