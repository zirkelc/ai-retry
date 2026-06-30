import { Errors, Iterables } from 'ai-test-kit';
import {
  APICallError,
  generateText,
  NoObjectGeneratedError,
  streamText,
} from 'ai';
import { describe, expect, it } from 'vitest';
import {
  MockLanguageModel,
  chunksToText,
  contentFilterResult,
  createRetryableModel,
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

const apiCallError = Errors.from({
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
      const baseModel = MockLanguageModel.from({ doGenerate: mockResult });
      const retryModel = MockLanguageModel.from({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doGenerate: contentFilterResult,
      });
      const retryModel = MockLanguageModel.from({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({ doGenerate: apiCallError });
      const retryModel = MockLanguageModel.from({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doGenerate: Errors.badRequest(),
      });

      const retryModel = MockLanguageModel.from({ doGenerate: mockResult });

      // Act
      const result = generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const retryModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [contentFilterTriggered(retryModel)],
        }),
        prompt: 'Hello!',
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await Iterables.toArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should retry in case of content filter error', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doStream: apiCallError });
      const retryModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [contentFilterTriggered(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await Iterables.toArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should retry when finish part reports content-filter before any content', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: contentFilterChunks,
      });
      const retryModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [contentFilterTriggered(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await Iterables.toArray(result.fullStream);

      // Assert
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
      expect(error).toBeUndefined();
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should not retry if no matches', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: Errors.badRequest(),
      });

      const retryModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [contentFilterTriggered(retryModel)],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        onError(data) {
          error = data.error;
        },
      });

      const chunks = await Iterables.toArray(result.fullStream);

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
            "error": [AI_APICallError: Bad request],
            "type": "error",
          },
        ]
      `);
    });
  });
});
