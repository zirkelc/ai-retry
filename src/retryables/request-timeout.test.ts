import { Errors, Iterables } from 'ai-test-kit';
import {
  APICallError,
  embed,
  generateImage,
  generateText,
  streamText,
} from 'ai';
import { describe, expect, it } from 'vitest';
import {
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
  chunksToText,
  createRetryableModel,
  mockEmbeddings,
  mockImageResult,
  mockResult,
  mockResultText,
  mockStreamChunks,
  type LanguageModelGenerateFn,
  type LanguageModelStreamFn,
} from '../internal/test-utils.js';
import type {
  EmbeddingModelCallOptions,
  LanguageModelCallOptions,
} from '../types.js';
import { requestTimeout } from './request-timeout.js';

const embeddingTimeoutError = async (opts: EmbeddingModelCallOptions) => {
  // Check if abortSignal is aborted and throw appropriate error
  // AbortSignal.timeout() throws TimeoutError, not AbortError
  if (opts.abortSignal?.aborted) {
    throw Errors.timeout();
  }

  // Listen for abort event during the async operation
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      resolve(mockEmbeddings);
    }, 1_000);

    opts.abortSignal?.addEventListener('abort', () => {
      clearTimeout(timeout);
      // AbortSignal.timeout() throws TimeoutError when it fires
      reject(Errors.timeout());
    });
  });
};

const timeoutError = async (opts: LanguageModelCallOptions) => {
  // Check if abortSignal is aborted and throw appropriate error
  // AbortSignal.timeout() throws TimeoutError, not AbortError
  if (opts.abortSignal?.aborted) {
    throw Errors.timeout();
  }

  // Listen for abort event during the async operation
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      resolve(mockResult);
    }, 1_000);

    opts.abortSignal?.addEventListener('abort', () => {
      clearTimeout(timeout);
      // AbortSignal.timeout() throws TimeoutError when it fires
      reject(Errors.timeout());
    });
  });
};

const genericError = Errors.internalServerError();

describe('requestTimeout', () => {
  describe('generateText', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({ doGenerate: mockResult });
      const retryModel = MockLanguageModel.from({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doGenerate: timeoutError as LanguageModelGenerateFn,
      });
      const retryModel = MockLanguageModel.from({ doGenerate: mockResult });

      // Act
      const result = await generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({ doGenerate: genericError });
      const retryModel = MockLanguageModel.from({ doGenerate: mockResult });

      // Act
      const result = generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doGenerate: (async (opts: LanguageModelCallOptions) => {
          baseModelSignal = opts.abortSignal;
          // AbortSignal.timeout() throws TimeoutError, not AbortError
          if (opts.abortSignal?.aborted) {
            throw Errors.timeout();
          }
          return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => resolve(mockResult), 5000);
            opts.abortSignal?.addEventListener('abort', () => {
              clearTimeout(timeout);
              reject(Errors.timeout());
            });
          });
        }) as LanguageModelGenerateFn,
      });

      // Retry model that captures its signal and verifies it's not aborted
      const retryModel = MockLanguageModel.from({
        doGenerate: (async (opts: LanguageModelCallOptions) => {
          retryModelSignal = opts.abortSignal;
          // This should NOT be aborted since we get a fresh signal with the fix
          if (opts.abortSignal?.aborted) {
            throw Errors.timeout({
              message: 'Retry failed: signal already aborted',
            });
          }
          return mockResult;
        }) as LanguageModelGenerateFn,
      });

      // Act
      const result = await generateText({
        model: createRetryableModel({
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

      const baseModel = MockLanguageModel.from({
        doGenerate: timeoutError as LanguageModelGenerateFn,
      });

      const retryModel = MockLanguageModel.from({
        doGenerate: (async (opts: LanguageModelCallOptions) => {
          retryModelSignal = opts.abortSignal;
          // Verify signal is not aborted
          if (opts.abortSignal?.aborted) {
            throw Errors.timeout({
              message: 'Should not be aborted initially',
            });
          }
          return mockResult;
        }) as LanguageModelGenerateFn,
      });

      // Act
      const result = await generateText({
        model: createRetryableModel({
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
          retries: [requestTimeout(retryModel)],
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

    // TODO needs to read the first chunk to get the abort error
    it.todo('should fallback in case of timeout error', async () => {
      // Arrange
      const baseModel = MockLanguageModel.from({
        doStream: timeoutError as LanguageModelStreamFn,
      });
      const retryModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryableModel({
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

      const chunks = await Iterables.toArray(result.fullStream);

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
      const baseModel = MockLanguageModel.from({ doStream: genericError });
      const retryModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      let error: unknown;

      // Act
      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [requestTimeout(retryModel)],
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
      expect(error).toBeDefined();
      expect(chunks).toMatchInlineSnapshot(`
        [
          {
            "type": "start",
          },
          {
            "error": [AI_APICallError: Internal server error],
            "type": "error",
          },
        ]
      `);
    });
  });

  describe('embed', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = MockEmbeddingModel.from(mockEmbeddings);
      const retryModel = MockEmbeddingModel.from(mockEmbeddings);

      // Act
      const result = await embed({
        model: createRetryableModel({
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
      const baseModel = MockEmbeddingModel.from(embeddingTimeoutError as any);
      const retryModel = MockEmbeddingModel.from(mockEmbeddings);

      // Act
      const result = await embed({
        model: createRetryableModel({
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
      const baseModel = MockEmbeddingModel.from(genericError);
      const retryModel = MockEmbeddingModel.from(mockEmbeddings);

      // Act
      const result = embed({
        model: createRetryableModel({
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

  describe('generateImage', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = MockImageModel.from(mockImageResult);
      const retryModel = MockImageModel.from(mockImageResult);

      // Act
      const result = await generateImage({
        model: createRetryableModel({
          model: baseModel,
          retries: [requestTimeout(retryModel)],
        }),
        prompt: 'test',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(result.images.length).toBe(1);
    });

    it('should fallback on timeout error', async () => {
      // Arrange
      const timeoutError = new Error('Request timed out');
      timeoutError.name = 'TimeoutError';

      const baseModel = MockImageModel.from(timeoutError);
      const retryModel = MockImageModel.from(mockImageResult);

      // Act
      const result = await generateImage({
        model: createRetryableModel({
          model: baseModel,
          retries: [requestTimeout(retryModel)],
        }),
        prompt: 'test',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.images.length).toBe(1);
    });

    it('should not fallback for non-timeout errors', async () => {
      // Arrange
      const baseModel = MockImageModel.from(genericError);
      const retryModel = MockImageModel.from(mockImageResult);

      // Act
      const result = generateImage({
        model: createRetryableModel({
          model: baseModel,
          retries: [requestTimeout(retryModel)],
        }),
        prompt: 'test',
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrowError(APICallError);
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    });
  });
});
