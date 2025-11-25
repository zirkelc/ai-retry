import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import { APICallError, embed, generateText, streamText } from 'ai';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import {
  chunksToText,
  type EmbeddingModelEmbed,
  MockEmbeddingModel,
  MockLanguageModel,
} from '../test-utils.js';
import type {
  EmbeddingModel,
  LanguageModel,
  LanguageModelGenerate,
  LanguageModelStreamPart,
  Retryable,
} from '../types.js';
import { isErrorAttempt } from '../utils.js';
import { retryAfterDelay } from './retry-after-delay.js';

const mockResultText = 'Hello, world!';

const mockResult: LanguageModelGenerate = {
  finishReason: 'stop',
  usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
  content: [{ type: 'text', text: mockResultText }],
  warnings: [],
};

const mockEmbeddings: EmbeddingModelEmbed = {
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

const genericRetryableError = new APICallError({
  message: 'Service temporarily unavailable',
  url: '',
  requestBodyValues: {},
  statusCode: 503,
  responseHeaders: {},
  responseBody: '{"error": {"message": "Service temporarily unavailable"}}',
  isRetryable: true,
  data: {
    error: {
      message: 'Service temporarily unavailable',
    },
  },
});

const mockStreamChunks: LanguageModelStreamPart[] = [
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
          retries: [retryAfterDelay({ delay: 1000 })],
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
      let attempt = 0;
      const baseModel = new MockLanguageModel({
        doGenerate: async () => {
          attempt++;
          if (attempt === 1) {
            throw rateLimitError;
          }
          return mockResult;
        },
      });

      const promise = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay({ delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doGenerate).toHaveBeenCalledTimes(2);
      expect(result.text).toBe(mockResultText);
    });

    it('should respect retry-after header in seconds', async () => {
      let attempt = 0;
      const baseModel = new MockLanguageModel({
        doGenerate: async () => {
          attempt++;
          if (attempt === 1) {
            throw rateLimitErrorWithRetryAfter;
          }
          return mockResult;
        },
      });

      const promise = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay({ delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doGenerate).toHaveBeenCalledTimes(2);
      expect(result.text).toBe(mockResultText);
    });

    it('should respect retry-after-ms header', async () => {
      let attempt = 0;
      const baseModel = new MockLanguageModel({
        doGenerate: async () => {
          attempt++;
          if (attempt === 1) {
            throw rateLimitErrorWithRetryAfterMs;
          }
          return mockResult;
        },
      });

      const promise = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay({ delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doGenerate).toHaveBeenCalledTimes(2);
      expect(result.text).toBe(mockResultText);
    });

    it('should respect retry-after header with HTTP date', async () => {
      let attempt = 0;
      const baseModel = new MockLanguageModel({
        doGenerate: async () => {
          attempt++;
          if (attempt === 1) {
            throw rateLimitErrorWithRetryAfterDate;
          }
          return mockResult;
        },
      });

      const promise = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay({ delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doGenerate).toHaveBeenCalledTimes(2);
      expect(result.text).toBe(mockResultText);
    });

    it('should not retry on non-retryable errors', async () => {
      const baseModel = new MockLanguageModel({
        doGenerate: nonRetryableError,
      });

      const result = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay({ delay: 1000 })],
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
            retryAfterDelay({
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
          retries: [retryAfterDelay({ delay: 1000, maxAttempts: 2 })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
      }).catch((error) => error);

      await vi.runAllTimersAsync();

      const error = await result;
      expect(error).toBeInstanceOf(Error);
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(2);
    });

    it('should retry with base model if no model is provided', async () => {
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

    it('should retry with fallback model when fallback fails with retryable error', async () => {
      const baseModel = new MockLanguageModel({
        doGenerate: genericRetryableError,
      });

      let fallbackAttempt = 0;
      const fallbackModel = new MockLanguageModel({
        doGenerate: async () => {
          fallbackAttempt++;
          if (fallbackAttempt === 1) {
            throw rateLimitError;
          }
          return mockResult;
        },
      });

      const fallbackOnError: Retryable<LanguageModel> = (context) => {
        if (
          isErrorAttempt(context.current) &&
          APICallError.isInstance(context.current.error)
        ) {
          return { model: fallbackModel, maxAttempts: 1 };
        }
        return undefined;
      };

      const promise = generateText({
        model: createRetryable({
          model: baseModel,
          retries: [
            fallbackOnError,
            retryAfterDelay({ delay: 100, maxAttempts: 3 }),
          ],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(2);
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
          retries: [retryAfterDelay({ delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      const chunks = await convertAsyncIterableToArray(result.fullStream);

      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should retry with delay', async () => {
      let attempt = 0;
      const baseModel = new MockLanguageModel({
        doStream: async () => {
          attempt++;
          if (attempt === 1) {
            throw rateLimitError;
          }
          return {
            stream: convertArrayToReadableStream(mockStreamChunks),
          };
        },
      });

      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay({ delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const chunks = await convertAsyncIterableToArray(result.fullStream);

      expect(baseModel.doStream).toHaveBeenCalledTimes(2);
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should retry with fallback model when fallback fails with retryable error', async () => {
      const baseModel = new MockLanguageModel({
        doStream: genericRetryableError,
      });

      let fallbackAttempt = 0;
      const fallbackModel = new MockLanguageModel({
        doStream: async () => {
          fallbackAttempt++;
          if (fallbackAttempt === 1) {
            throw rateLimitError;
          }
          return {
            stream: convertArrayToReadableStream(mockStreamChunks),
          };
        },
      });

      const fallbackOnError: Retryable<LanguageModel> = (context) => {
        if (
          isErrorAttempt(context.current) &&
          APICallError.isInstance(context.current.error)
        ) {
          return { model: fallbackModel, maxAttempts: 1 };
        }
        return undefined;
      };

      const result = streamText({
        model: createRetryable({
          model: baseModel,
          retries: [
            fallbackOnError,
            retryAfterDelay({ delay: 100, maxAttempts: 3 }),
          ],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const chunks = await convertAsyncIterableToArray(result.fullStream);

      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doStream).toHaveBeenCalledTimes(2);
      expect(chunksToText(chunks)).toBe(mockResultText);
    });
  });

  describe('embed', () => {
    it('should succeed without errors', async () => {
      const baseModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      const promise = embed({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay({ delay: 1000 })],
        }),
        value: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toBe(mockEmbeddings.embeddings[0]);
    });

    it('should retry with delay', async () => {
      let attempt = 0;
      const baseModel = new MockEmbeddingModel({
        doEmbed: async () => {
          attempt++;
          if (attempt === 1) {
            throw rateLimitError;
          }
          return mockEmbeddings;
        },
      });

      const promise = embed({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay({ delay: 1000 })],
        }),
        value: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doEmbed).toHaveBeenCalledTimes(2);
      expect(result.embedding).toBe(mockEmbeddings.embeddings[0]);
    });

    it('should respect retry-after header', async () => {
      let attempt = 0;
      const baseModel = new MockEmbeddingModel({
        doEmbed: async () => {
          attempt++;
          if (attempt === 1) {
            throw rateLimitErrorWithRetryAfter;
          }
          return mockEmbeddings;
        },
      });

      const promise = embed({
        model: createRetryable({
          model: baseModel,
          retries: [retryAfterDelay({ delay: 1000 })],
        }),
        value: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doEmbed).toHaveBeenCalledTimes(2);
      expect(result.embedding).toBe(mockEmbeddings.embeddings[0]);
    });

    it('should retry with fallback model when fallback fails with retryable error', async () => {
      const baseModel = new MockEmbeddingModel({
        doEmbed: genericRetryableError,
      });

      let fallbackAttempt = 0;
      const fallbackModel = new MockEmbeddingModel({
        doEmbed: async () => {
          fallbackAttempt++;
          if (fallbackAttempt === 1) {
            throw rateLimitError;
          }
          return mockEmbeddings;
        },
      });

      const fallbackOnError: Retryable<EmbeddingModel> = (context) => {
        if (
          isErrorAttempt(context.current) &&
          APICallError.isInstance(context.current.error)
        ) {
          return { model: fallbackModel, maxAttempts: 1 };
        }
        return undefined;
      };

      const promise = embed({
        model: createRetryable({
          model: baseModel,
          retries: [
            fallbackOnError,
            retryAfterDelay({ delay: 100, maxAttempts: 3 }),
          ],
        }),
        value: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doEmbed).toHaveBeenCalledTimes(2);
      expect(result.embedding).toBe(mockEmbeddings.embeddings[0]);
    });
  });
});
