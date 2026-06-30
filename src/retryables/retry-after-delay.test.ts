import { Errors, Iterables, Streams } from 'ai-test-kit';
import {
  APICallError,
  embed,
  generateImage,
  generateText,
  streamText,
} from 'ai';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  chunksToText,
  createRetryableModel,
  MockEmbeddingModel,
  mockEmbeddings,
  MockImageModel,
  mockImageResult,
  MockLanguageModel,
  mockResult,
  mockResultText,
  mockStreamChunks,
  nonRetryableError,
} from '../internal/test-utils.js';
import type {
  EmbeddingModel,
  ImageModel,
  LanguageModel,
  Retryable,
} from '../types.js';
import { isErrorAttempt } from '../internal/guards.js';
import { retryAfterDelay } from './retry-after-delay.js';

const rateLimitError = Errors.rateLimited();

const rateLimitErrorWithRetryAfter = Errors.rateLimited({ retryAfter: 5 });

const rateLimitErrorWithRetryAfterMs = Errors.rateLimited({
  retryAfter: { ms: 3000 },
});

const rateLimitErrorWithRetryAfterDate = Errors.rateLimited({
  retryAfter: new Date(Date.now() + 4000),
});

const genericRetryableError = Errors.serviceUnavailable();

describe('retryAfterDelay', () => {
  beforeEach(() => {
    vi.clearAllTimers();
    vi.useFakeTimers();
  });

  describe('generateText', () => {
    it('should succeed without errors', async () => {
      const baseModel = MockLanguageModel.from({ doGenerate: mockResult });
      const retryModel = MockLanguageModel.from({ doGenerate: mockResult });

      const promise = generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doGenerate: async () => {
          attempt++;
          if (attempt === 1) {
            throw rateLimitError;
          }
          return mockResult;
        },
      });

      const promise = generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doGenerate: async () => {
          attempt++;
          if (attempt === 1) {
            throw rateLimitErrorWithRetryAfter;
          }
          return mockResult;
        },
      });

      const promise = generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doGenerate: async () => {
          attempt++;
          if (attempt === 1) {
            throw rateLimitErrorWithRetryAfterMs;
          }
          return mockResult;
        },
      });

      const promise = generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doGenerate: async () => {
          attempt++;
          if (attempt === 1) {
            throw rateLimitErrorWithRetryAfterDate;
          }
          return mockResult;
        },
      });

      const promise = generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doGenerate: nonRetryableError,
      });

      const result = generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doGenerate: async () => {
          attempt++;
          if (attempt < 3) {
            throw rateLimitError;
          }
          return mockResult;
        },
      });

      const promise = generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({ doGenerate: rateLimitError });

      const result = generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doGenerate: async () => {
          attempt++;
          if (attempt < 3) {
            throw rateLimitError;
          }
          return mockResult;
        },
      });

      const promise = generateText({
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doGenerate: genericRetryableError,
      });

      let fallbackAttempt = 0;
      const fallbackModel = MockLanguageModel.from({
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
        model: createRetryableModel({
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
      const baseModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });
      const retryModel = MockLanguageModel.from({
        doStream: mockStreamChunks,
      });

      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [retryAfterDelay({ delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      const chunks = await Iterables.toArray(result.fullStream);

      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(retryModel.doStream).toHaveBeenCalledTimes(0);
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should retry with delay', async () => {
      let attempt = 0;
      const baseModel = MockLanguageModel.from({
        doStream: async () => {
          attempt++;
          if (attempt === 1) {
            throw rateLimitError;
          }
          return {
            stream: Streams.from(mockStreamChunks),
          };
        },
      });

      const result = streamText({
        model: createRetryableModel({
          model: baseModel,
          retries: [retryAfterDelay({ delay: 1000 })],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const chunks = await Iterables.toArray(result.fullStream);

      expect(baseModel.doStream).toHaveBeenCalledTimes(2);
      expect(chunksToText(chunks)).toBe(mockResultText);
    });

    it('should retry with fallback model when fallback fails with retryable error', async () => {
      const baseModel = MockLanguageModel.from({
        doStream: genericRetryableError,
      });

      let fallbackAttempt = 0;
      const fallbackModel = MockLanguageModel.from({
        doStream: async () => {
          fallbackAttempt++;
          if (fallbackAttempt === 1) {
            throw rateLimitError;
          }
          return {
            stream: Streams.from(mockStreamChunks),
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
        model: createRetryableModel({
          model: baseModel,
          retries: [
            fallbackOnError,
            retryAfterDelay({ delay: 100, maxAttempts: 3 }),
          ],
        }),
        prompt: 'Hello!',
      });

      await vi.runAllTimersAsync();
      const chunks = await Iterables.toArray(result.fullStream);

      expect(baseModel.doStream).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doStream).toHaveBeenCalledTimes(2);
      expect(chunksToText(chunks)).toBe(mockResultText);
    });
  });

  describe('embed', () => {
    it('should succeed without errors', async () => {
      const baseModel = MockEmbeddingModel.from(mockEmbeddings);

      const promise = embed({
        model: createRetryableModel({
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
      const baseModel = MockEmbeddingModel.from(async () => {
        attempt++;
        if (attempt === 1) {
          throw rateLimitError;
        }
        return mockEmbeddings;
      });

      const promise = embed({
        model: createRetryableModel({
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
      const baseModel = MockEmbeddingModel.from(async () => {
        attempt++;
        if (attempt === 1) {
          throw rateLimitErrorWithRetryAfter;
        }
        return mockEmbeddings;
      });

      const promise = embed({
        model: createRetryableModel({
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
      const baseModel = MockEmbeddingModel.from(genericRetryableError);

      let fallbackAttempt = 0;
      const fallbackModel = MockEmbeddingModel.from(async () => {
        fallbackAttempt++;
        if (fallbackAttempt === 1) {
          throw rateLimitError;
        }
        return mockEmbeddings;
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
        model: createRetryableModel({
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

  describe('generateImage', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = MockImageModel.from(mockImageResult);

      // Act
      const promise = generateImage({
        model: createRetryableModel({
          model: baseModel,
          retries: [retryAfterDelay({ delay: 1000 })],
        }),
        prompt: 'test',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.images.length).toBe(1);
    });

    it('should retry with delay on retryable error', async () => {
      // Arrange
      let attempt = 0;
      const baseModel = MockImageModel.from(async () => {
        attempt++;
        if (attempt === 1) {
          throw rateLimitError;
        }
        return mockImageResult;
      });

      // Act
      const promise = generateImage({
        model: createRetryableModel({
          model: baseModel,
          retries: [retryAfterDelay({ delay: 1000 })],
        }),
        prompt: 'test',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(2);
      expect(result.images.length).toBe(1);
    });

    it('should retry with fallback model when fallback fails with retryable error', async () => {
      // Arrange
      const baseModel = MockImageModel.from(genericRetryableError);

      let fallbackAttempt = 0;
      const fallbackModel = MockImageModel.from(async () => {
        fallbackAttempt++;
        if (fallbackAttempt === 1) {
          throw rateLimitError;
        }
        return mockImageResult;
      });

      const fallbackOnError: Retryable<ImageModel> = (context) => {
        if (
          isErrorAttempt(context.current) &&
          APICallError.isInstance(context.current.error)
        ) {
          return { model: fallbackModel, maxAttempts: 1 };
        }
        return undefined;
      };

      // Act
      const promise = generateImage({
        model: createRetryableModel({
          model: baseModel,
          retries: [
            fallbackOnError,
            retryAfterDelay({ delay: 100, maxAttempts: 3 }),
          ],
        }),
        prompt: 'test',
      });

      await vi.runAllTimersAsync();
      const result = await promise;

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(2);
      expect(result.images.length).toBe(1);
    });
  });
});
