import type {
  LanguageModelV2CallOptions,
  LanguageModelV2StreamPart,
} from '@ai-sdk/provider';
import {
  convertArrayToReadableStream,
  convertAsyncIterableToArray,
} from '@ai-sdk/provider-utils/test';
import { APICallError, embed, generateText, streamText } from 'ai';
import { describe, expect, it } from 'vitest';
import { createRetryable } from '../create-retryable-model.js';
import {
  chunksToText,
  type EmbeddingModelV2Embed,
  type LanguageModelV2GenerateFn,
  type LanguageModelV2StreamFn,
  MockEmbeddingModel,
  MockLanguageModel,
} from '../test-utils.js';
import type {
  EmbeddingModelV2CallOptions,
  LanguageModelV2Generate,
} from '../types.js';
import { requestTimeout } from './request-timeout.js';

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

const embeddingTimeoutError = async (
  opts: EmbeddingModelV2CallOptions<unknown>,
) => {
  // Check if abortSignal is aborted and throw appropriate error
  if (opts.abortSignal?.aborted) {
    throw new DOMException('The operation was aborted', 'AbortError');
  }

  // Listen for abort event during the async operation
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      resolve(mockEmbeddings);
    }, 1_000);

    opts.abortSignal?.addEventListener('abort', () => {
      clearTimeout(timeout);
      reject(new DOMException('The operation was aborted', 'AbortError'));
    });
  });
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
      const baseModel = new MockLanguageModel({ doGenerate: mockResult });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [requestTimeout(retryModel)],
        }),
        prompt: 'Hello!',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should fallback in case of timeout error', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: timeoutError as LanguageModelV2GenerateFn,
      });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

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
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);
    });

    it('should not fallback for non-timeout errors', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doGenerate: genericError,
      });
      const retryModel = new MockLanguageModel({ doGenerate: mockResult });

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
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    });

    it('should use fresh abort signal on retry (fix for abort signal reuse bug)', async () => {
      // Arrange
      let baseModelSignal: AbortSignal | undefined;
      let retryModelSignal: AbortSignal | undefined;

      // Base model that captures the abort signal and respects it
      const baseModel = new MockLanguageModel({
        doGenerate: (async (opts: LanguageModelV2CallOptions) => {
          baseModelSignal = opts.abortSignal;
          if (opts.abortSignal?.aborted) {
            throw new DOMException('The operation was aborted', 'AbortError');
          }
          return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => resolve(mockResult), 5000);
            opts.abortSignal?.addEventListener('abort', () => {
              clearTimeout(timeout);
              reject(
                new DOMException('The operation was aborted', 'AbortError'),
              );
            });
          });
        }) as LanguageModelV2GenerateFn,
      });

      // Retry model that captures its signal and verifies it's not aborted
      const retryModel = new MockLanguageModel({
        doGenerate: (async (opts: LanguageModelV2CallOptions) => {
          retryModelSignal = opts.abortSignal;
          // This should NOT be aborted since we get a fresh signal with the fix
          if (opts.abortSignal?.aborted) {
            throw new DOMException(
              'Retry failed: signal already aborted',
              'AbortError',
            );
          }
          return mockResult;
        }) as LanguageModelV2GenerateFn,
      });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [requestTimeout(retryModel)], // Uses default 60s timeout
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        abortSignal: AbortSignal.timeout(100), // Original times out after 100ms
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);

      // Verify the signals are different
      expect(baseModelSignal).toBeDefined();
      expect(retryModelSignal).toBeDefined();
      expect(baseModelSignal).not.toBe(retryModelSignal);

      // The original signal should be aborted, the new one should not be
      expect(baseModelSignal?.aborted).toBe(true);
      expect(retryModelSignal?.aborted).toBe(false);
    });

    it('should use custom timeout on retry when specified', async () => {
      // Arrange
      const customTimeout = 30000; // 30 seconds
      let retryModelSignal: AbortSignal | undefined;

      const baseModel = new MockLanguageModel({
        doGenerate: timeoutError as LanguageModelV2GenerateFn,
      });

      const retryModel = new MockLanguageModel({
        doGenerate: (async (opts: LanguageModelV2CallOptions) => {
          retryModelSignal = opts.abortSignal;
          // Verify signal is not aborted
          if (opts.abortSignal?.aborted) {
            throw new DOMException(
              'Should not be aborted initially',
              'AbortError',
            );
          }
          return mockResult;
        }) as LanguageModelV2GenerateFn,
      });

      // Act
      const result = await generateText({
        model: createRetryable({
          model: baseModel,
          retries: [requestTimeout(retryModel, { timeout: customTimeout })],
        }),
        prompt: 'Hello!',
        maxRetries: 0,
        abortSignal: AbortSignal.timeout(100),
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.text).toBe(mockResultText);

      // Verify retry got a fresh signal
      expect(retryModelSignal).toBeDefined();
      expect(retryModelSignal?.aborted).toBe(false);
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
          retries: [requestTimeout(retryModel)],
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

    // TODO needs to read the first chunk to get the abort error
    it.todo('should fallback in case of timeout error', async () => {
      // Arrange
      const baseModel = new MockLanguageModel({
        doStream: timeoutError as LanguageModelV2StreamFn,
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
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(1);
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
      const baseModel = new MockLanguageModel({ doStream: genericError });
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
      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
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

  describe('embed', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });
      const retryModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      // Act
      const result = await embed({
        model: createRetryable({
          model: baseModel,
          retries: [requestTimeout(retryModel)],
        }),
        value: 'Hello!',
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual(mockEmbeddings.embeddings[0]);
    });

    it('should fallback in case of timeout error', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({
        doEmbed: embeddingTimeoutError as any,
      });
      const retryModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      // Act
      const result = await embed({
        model: createRetryable({
          model: baseModel,
          retries: [requestTimeout(retryModel)],
        }),
        value: 'Hello!',
        maxRetries: 0,
        abortSignal: AbortSignal.timeout(100), // Very short timeout to trigger timeout
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual(mockEmbeddings.embeddings[0]);
    });

    it('should not fallback for non-timeout errors', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({
        doEmbed: genericError,
      });
      const retryModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

      // Act
      const result = embed({
        model: createRetryable({
          model: baseModel,
          retries: [requestTimeout(retryModel)],
        }),
        value: 'Hello!',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrowError(APICallError);
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(retryModel.doEmbed).toHaveBeenCalledTimes(0);
    });
  });
});
