import type { LanguageModelV2StreamPart } from '@ai-sdk/provider';
import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import { APICallError, embed, generateText, streamText } from 'ai';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import {
  chunksToText,
  type EmbeddingModelV2Embed,
  MockEmbeddingModel,
  MockLanguageModel,
} from '../test-utils.js';
import type { LanguageModelV2Generate } from '../types.js';
import { retryAfterDelay } from './retry-after-delay.js';

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelV2Generate = {
  finishReason: 'stop',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const mockEmbeddings: EmbeddingModelV2Embed = {
  embeddings: [[0.1, 0.2, 0.3]],
  usage: { tokens: 5 },
};

const rateLimitError = new APICallError({
  message: 'Rate limit exceeded',
  url: '',
  requestBodyValues: {},
  statusCode: 429,
  responseHeaders: {},
  responseBody: '{"error": {"message": "Rate limit exceeded"}}',
  isRetryable: true,
  data: {
    error: {
      message: 'Rate limit exceeded',
    },
  },
});

const rateLimitErrorWithRetryAfter = new APICallError({
  message: 'Rate limit exceeded',
  url: '',
  requestBodyValues: {},
  statusCode: 429,
  responseHeaders: {
    'retry-after': '5',
  },
  responseBody: '{"error": {"message": "Rate limit exceeded"}}',
  isRetryable: true,
  data: {
    error: {
      message: 'Rate limit exceeded',
    },
  },
});

const rateLimitErrorWithRetryAfterMs = new APICallError({
  message: 'Rate limit exceeded',
  url: '',
  requestBodyValues: {},
  statusCode: 429,
  responseHeaders: {
    'retry-after-ms': '3000',
  },
  responseBody: '{"error": {"message": "Rate limit exceeded"}}',
  isRetryable: true,
  data: {
    error: {
      message: 'Rate limit exceeded',
    },
  },
});

const rateLimitErrorWithRetryAfterDate = new APICallError({
  message: 'Rate limit exceeded',
  url: '',
  requestBodyValues: {},
  statusCode: 429,
  responseHeaders: {
    'retry-after': new Date(Date.now() + 4000).toUTCString(),
  },
  responseBody: '{"error": {"message": "Rate limit exceeded"}}',
  isRetryable: true,
  data: {
    error: {
      message: 'Rate limit exceeded',
    },
  },
});

const nonRetryableError = new APICallError({
  message: 'Invalid API key',
  url: '',
  requestBodyValues: {},
  statusCode: 401,
  responseHeaders: {},
  responseBody: '{"error": {"message": "Invalid API key"}}',
  isRetryable: false,
  data: {
    error: {
      message: 'Invalid API key',
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

describe('retryAfterDelay', () => {
  beforeEach(() => {
    vi.clearAllTimers();
    vi.useFakeTimers();
  });

  describe('generateText', () => {
    it('should succeed without errors', async () => {
      const baseModel = new MockLanguageModel({ doGenerate: mockResult });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      const promise = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay(retryModel, { delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(result.text).toBe(mockResultText);
    });

    it('should retry with delay', async () => {
      const baseModel = new MockLanguageModel({ doGenerate: rateLimitError });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      const promise = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay(retryModel, { delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should respect retry-after header in seconds', async () => {
      const baseModel = new MockLanguageModel({
        doGenerate: rateLimitErrorWithRetryAfter,
      });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      const promise = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay(retryModel, { delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should respect retry-after-ms header', async () => {
      const baseModel = new MockLanguageModel({
        doGenerate: rateLimitErrorWithRetryAfterMs,
      });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      const promise = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay(retryModel, { delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should respect retry-after header with HTTP date', async () => {
      const baseModel = new MockLanguageModel({
        doGenerate: rateLimitErrorWithRetryAfterDate,
      });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      const promise = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay(retryModel, { delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should not retry on non-retryable errors', async () => {
      const baseModel = new MockLanguageModel({
        doGenerate: nonRetryableError,
      });

      const result = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay(baseModel, { delay: 1000 })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      });

      await expect(result).rejects.toThrowError(APICallError);
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    });

    it('should use exponential backoff', async () => {
      let attempt = 0;
      const baseModel = new MockLanguageModel({
        doGenerate: async () => {
          attempt++;
          if (attempt < 3) {
            throw rateLimitError;
          }
          return mockResult;
        },
      });

      const promise = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [
            retryAfterDelay(baseModel, {
              delay: 100,
              backoffFactor: 3,
              maxAttempts: 5,
            }),
          ],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doGenerate).toHaveBeenCalledTimes(3);
      expect(result.text).toBe(mockResultText);
    });

    it('should respect maxAttempts', async () => {
      const baseModel = new MockLanguageModel({ doGenerate: rateLimitError });

      const result = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [
            retryAfterDelay(baseModel, { delay: 1000, maxAttempts: 2 }),
          ],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      }).catch((error) => error);

      await vi.runAllTimersAsync();

      const error = await result;
      expect(error).toBeInstanceOf(Error);
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(2);
    });

    it('should retry on same model when no model parameter provided', async () => {
      let attempt = 0;
      const baseModel = new MockLanguageModel({
        doGenerate: async () => {
          attempt++;
          if (attempt < 3) {
            throw rateLimitError;
          }
          return mockResult;
        },
      });

      const promise = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay({ delay: 100, maxAttempts: 5 })],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doGenerate).toHaveBeenCalledTimes(3);
      expect(result.text).toBe(mockResultText);
    });
  });

  describe('streamText', () => {
    it('should succeed without errors', async () => {
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

      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay(retryModel, { delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should retry with delay', async () => {
      const baseModel = new MockLanguageModel({ doStream: rateLimitError });
      const retryModel = new MockLanguageModel({
        doStream: {
          stream: convertArrayToReadableStream(mockStreamChunks),
        },
      });

      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay(retryModel, { delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const chunks = await convertAsyncIterableToArray(result.fullStream);

      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
      expect(chunksToText(chunks)).toBe(mockResultText);
    });
  });

  describe('embed', () => {
    it('should succeed without errors', async () => {
      const baseModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });
      const retryModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      const promise = embed({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay(retryModel, { delay: 1000 })],
        }),
        value: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toBe(mockEmbeddings.embeddings[0]);
    });

    it('should retry with delay', async () => {
      const baseModel = new MockEmbeddingModel({ doEmbed: rateLimitError });
      const retryModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      const promise = embed({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay(retryModel, { delay: 1000 })],
        }),
        value: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toBe(mockEmbeddings.embeddings[0]);
    });

    it('should respect retry-after header', async () => {
      const baseModel = new MockEmbeddingModel({
        doEmbed: rateLimitErrorWithRetryAfter,
      });
      const retryModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      const promise = embed({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay(retryModel, { delay: 1000 })],
        }),
        value: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toBe(mockEmbeddings.embeddings[0]);
    });
  });
});
