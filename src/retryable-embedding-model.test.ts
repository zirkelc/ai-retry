import { APICallError, embed, RetryError } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { createRetryable } from './create-retryable-model.js';
import { MockEmbeddingModel } from './test-utils.js';
import type {
  EmbeddingModel,
  EmbeddingModelEmbed,
  Retryable,
  RetryableModelOptions,
  RetryContext,
} from './types.js';
import { isErrorAttempt } from './utils.js';

type OnError = Required<RetryableModelOptions<EmbeddingModel>>['onError'];
type OnRetry = Required<RetryableModelOptions<EmbeddingModel>>['onRetry'];

const mockEmbeddings: EmbeddingModelEmbed<number> = {
  embeddings: [[0.1, 0.2, 0.3]],
  usage: { tokens: 5 },
};

const retryableError = new APICallError({
  message: 'Rate limit exceeded',
  url: '',
  requestBodyValues: {},
  statusCode: 429,
  responseHeaders: {},
  responseBody:
    '{"error": {"message": "Rate limit exceeded", "code": "rate_limit_exceeded"}}',
  isRetryable: true,
  data: {
    error: {
      message: 'Rate limit exceeded',
      code: 'rate_limit_exceeded',
    },
  },
});

const nonRetryableError = new APICallError({
  message: 'Invalid API key',
  url: '',
  requestBodyValues: {},
  statusCode: 401,
  responseHeaders: {},
  responseBody:
    '{"error": {"message": "Invalid API key", "code": "invalid_api_key"}}',
  isRetryable: false,
  data: {
    error: {
      message: 'Invalid API key',
      code: 'invalid_api_key',
    },
  },
});

describe('embed', () => {
  it('should embed successfully when no errors occur', async () => {
    // Arrange
    const baseModel = new MockEmbeddingModel({
      doEmbed: mockEmbeddings,
    });
    const retryableModel = createRetryable({
      model: baseModel,
      retries: [],
    });

    // Act
    const result = await embed({
      model: retryableModel,
      value: 'Hello!',
    });

    // Assert
    expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
    expect(result.embedding).toEqual(mockEmbeddings.embeddings[0]);
  });

  describe('retries', () => {
    describe('error-based retries', () => {
      it('should retry with errors', async () => {
        // Arrange
        const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
        const fallbackModel = new MockEmbeddingModel({
          doEmbed: mockEmbeddings,
        });

        const fallbackRetryable = (context: RetryContext<EmbeddingModel>) => {
          if (
            isErrorAttempt(context.current) &&
            APICallError.isInstance(context.current.error)
          ) {
            return { model: fallbackModel, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await embed({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackRetryable],
          }),
          value: 'Hello!',
          //
        });

        // Assert
        expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doEmbed).toHaveBeenCalledTimes(1);
        expect(result.embedding).toEqual(mockEmbeddings.embeddings[0]);
      });

      it('should not retry without errors', async () => {
        // Arrange
        const baseModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });
        const fallbackModel1 = new MockEmbeddingModel({
          doEmbed: mockEmbeddings,
        });
        const fallbackModel2 = new MockEmbeddingModel({
          doEmbed: mockEmbeddings,
        });

        // Act
        const result = await embed({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackModel1, fallbackModel2],
          }),
          value: 'Hello!',
        });

        // Assert
        expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doEmbed).toHaveBeenCalledTimes(0);
        expect(fallbackModel2.doEmbed).toHaveBeenCalledTimes(0);
        expect(result.embedding).toEqual(mockEmbeddings.embeddings[0]);
      });

      it('should use plain embedding models for error-based attempts', async () => {
        // Arrange
        const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
        const fallbackModel1 = new MockEmbeddingModel({
          doEmbed: mockEmbeddings,
        });
        const fallbackModel2 = new MockEmbeddingModel({
          doEmbed: mockEmbeddings,
        });

        const fallbackRetryable = (context: RetryContext<EmbeddingModel>) => {
          if (
            isErrorAttempt(context.current) &&
            APICallError.isInstance(context.current.error)
          ) {
            return { model: fallbackModel2, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await embed({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackModel1, fallbackRetryable],
          }),
          value: 'Hello!',
        });

        // Assert
        expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doEmbed).toHaveBeenCalledTimes(1); // Should be called
        expect(fallbackModel2.doEmbed).toHaveBeenCalledTimes(0);
        expect(result.embedding).toEqual(mockEmbeddings.embeddings[0]);
      });
    });
  });

  describe('disabled', () => {
    it('should not retry when disabled is true', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
      const fallbackModel = new MockEmbeddingModel({
        doEmbed: mockEmbeddings,
      });

      const fallbackRetryable: Retryable<EmbeddingModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: true,
      });

      // Act & Assert
      await expect(
        embed({
          model: retryableModel,
          value: 'Hello!',
          maxRetries: 0, // Disable AI SDK's own retry mechanism
        }),
      ).rejects.toThrow('Rate limit exceeded');

      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doEmbed).toHaveBeenCalledTimes(0);
    });

    it('should retry when disabled is false', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
      const fallbackModel = new MockEmbeddingModel({
        doEmbed: mockEmbeddings,
      });

      const fallbackRetryable: Retryable<EmbeddingModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: false,
      });

      // Act
      const result = await embed({
        model: retryableModel,
        value: 'Hello!',
      });

      // Assert
      expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doEmbed).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual([0.1, 0.2, 0.3]);
    });
  });

  describe('onError', () => {
    it('should call onError handler when an error occurs', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
      const fallbackModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      await embed({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackModel],
          onError: onErrorSpy,
        }),
        value: 'Hello!',
      });

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(1);

      const firstErrorCall = onErrorSpy.mock.calls[0]![0];
      expect(firstErrorCall.current.error).toBe(retryableError);
      expect(firstErrorCall.current.model).toBe(baseModel);
      expect(firstErrorCall.attempts.length).toBe(1);
      // expect(firstErrorCall.totalAttempts).toBe(1);
    });

    it('should call onError handler for each error in multiple attempts', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
      const fallbackModel = new MockEmbeddingModel({
        doEmbed: nonRetryableError,
      });
      const finalModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });
      const onErrorSpy = vi.fn<OnError>();

      // Act
      await embed({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackModel, finalModel],
          onError: onErrorSpy,
        }),
        value: 'Hello!',
      });

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(2);

      // Check that onError was called for each error
      const firstErrorCall = onErrorSpy.mock.calls[0]![0];
      const secondErrorCall = onErrorSpy.mock.calls[1]![0];

      expect(firstErrorCall.current.error).toBe(retryableError);
      expect(firstErrorCall.current.model).toBe(baseModel);
      expect(firstErrorCall.attempts.length).toBe(1);
      // expect(firstErrorCall.totalAttempts).toBe(1);

      expect(secondErrorCall.current.error).toBe(nonRetryableError);
      expect(secondErrorCall.current.model).toBe(fallbackModel);
      expect(secondErrorCall.attempts.length).toBe(2);
      // expect(secondErrorCall.totalAttempts).toBe(2);
    });

    it('should call onError handler before onRetry handler', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
      const fallbackModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });
      const onErrorSpy = vi.fn<OnError>();
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      await embed({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackModel],
          onError: onErrorSpy,
          onRetry: onRetrySpy,
        }),
        value: 'Hello!',
      });

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(1);
      expect(onRetrySpy).toHaveBeenCalledTimes(1);

      // Verify onError is called before onRetry by checking call order
      const errorCallTime = onErrorSpy.mock.invocationCallOrder[0] ?? 0;
      const retryCallTime = onRetrySpy.mock.invocationCallOrder[0] ?? 0;
      expect(errorCallTime).toBeLessThan(retryCallTime);

      // Verify the context passed to each handler
      const firstErrorCall = onErrorSpy.mock.calls[0]![0];
      const firstRetryCall = onRetrySpy.mock.calls[0]![0];
      expect(firstErrorCall.current.model).toBe(baseModel);
      expect(firstErrorCall.attempts.length).toBe(1);
      expect(firstRetryCall.current.model).toBe(fallbackModel);
      expect(firstRetryCall.attempts.length).toBe(1);
    });
  });

  describe('onRetry', () => {
    it('should call onRetry handler for error-based retries', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
      const fallbackModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      await embed({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackModel],
          onRetry: onRetrySpy,
        }),
        value: 'Hello!',
      });

      // Assert
      expect(onRetrySpy).toHaveBeenCalledTimes(1);

      const firstRetryCall = onRetrySpy.mock.calls[0]![0];
      expect(isErrorAttempt(firstRetryCall.current)).toBe(true);
      if (isErrorAttempt(firstRetryCall.current)) {
        expect(firstRetryCall.current.error).toBe(retryableError);
      }
      expect(firstRetryCall.current.model).toBe(fallbackModel);
      expect(firstRetryCall.attempts.length).toBe(1);
      // expect(firstRetryCall.totalAttempts).toBe(1);
    });

    it('should call onRetry handler for each retry attempt', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
      const fallbackModel1 = new MockEmbeddingModel({
        doEmbed: nonRetryableError,
      });
      const fallbackModel2 = new MockEmbeddingModel({
        doEmbed: mockEmbeddings,
      });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      await embed({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackModel1, fallbackModel2],
          onRetry: onRetrySpy,
        }),
        value: 'Hello!',
      });

      // Assert
      expect(onRetrySpy).toHaveBeenCalledTimes(2);

      // Check that onRetry was called for each retry
      const firstRetryCall = onRetrySpy.mock.calls[0]![0];
      const secondRetryCall = onRetrySpy.mock.calls[1]![0];

      expect(isErrorAttempt(firstRetryCall.current)).toBe(true);
      if (isErrorAttempt(firstRetryCall.current)) {
        expect(firstRetryCall.current.error).toBe(retryableError);
      }
      expect(firstRetryCall.current.model).toBe(fallbackModel1);
      expect(firstRetryCall.attempts.length).toBe(1);
      // expect(firstRetryCall.totalAttempts).toBe(1);

      expect(isErrorAttempt(secondRetryCall.current)).toBe(true);
      if (isErrorAttempt(secondRetryCall.current)) {
        expect(secondRetryCall.current.error).toBe(nonRetryableError);
      }
      expect(secondRetryCall.current.model).toBe(fallbackModel2);
      expect(secondRetryCall.attempts.length).toBe(2);
      // expect(secondRetryCall.totalAttempts).toBe(2);
    });

    it('should NOT call onRetry on first attempt', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });
      const onRetrySpy = vi.fn<OnRetry>();

      // Act
      await embed({
        model: createRetryable({
          model: baseModel,
          retries: [],
          onRetry: onRetrySpy,
        }),
        value: 'Hello!',
      });

      // Assert
      expect(onRetrySpy).not.toHaveBeenCalled();
    });
  });

  describe('RetryableOptions', () => {
    describe('maxAttempts', () => {
      it('should try each model only once by default', async () => {
        // Arrange
        const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
        const fallbackModel1 = new MockEmbeddingModel({
          doEmbed: retryableError,
        });
        const fallbackModel2 = new MockEmbeddingModel({
          doEmbed: retryableError,
        });
        const finalModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

        // Act
        await embed({
          model: createRetryable({
            model: baseModel,
            retries: [
              fallbackModel1,
              { model: fallbackModel2 },
              async () => ({ model: finalModel }),
            ],
          }),
          value: 'Hello!',
        });

        // Assert
        expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doEmbed).toHaveBeenCalledTimes(1);
        expect(fallbackModel2.doEmbed).toHaveBeenCalledTimes(1);
        expect(finalModel.doEmbed).toHaveBeenCalledTimes(1);
      });

      it('should try models multiple times if maxAttempts is set', async () => {
        // Arrange
        const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
        const fallbackModel1 = new MockEmbeddingModel({
          doEmbed: retryableError,
        });
        const fallbackModel2 = new MockEmbeddingModel({
          doEmbed: retryableError,
        });
        const finalModel = new MockEmbeddingModel({ doEmbed: mockEmbeddings });

        // Act
        await embed({
          model: createRetryable({
            model: baseModel,
            retries: [
              // Retryable  with different maxAttempts
              { model: fallbackModel1, maxAttempts: 2 },
              async () => ({ model: fallbackModel2, maxAttempts: 3 }),
              finalModel,
            ],
          }),
          value: 'Hello!',
        });

        // Assert
        expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doEmbed).toHaveBeenCalledTimes(2);
        expect(fallbackModel2.doEmbed).toHaveBeenCalledTimes(3);
        expect(finalModel.doEmbed).toHaveBeenCalledTimes(1);
      });
    });

    describe('maxRetries', () => {
      it('should ignore maxRetries setting when retryable matches', async () => {
        // Arrange
        const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
        const fallbackModel = new MockEmbeddingModel({
          doEmbed: nonRetryableError,
        });

        // Act & Assert
        try {
          await embed({
            model: createRetryable({
              model: baseModel,
              retries: [fallbackModel],
            }),
            value: 'Hello!',
            maxRetries: 1, // Should be ignored since RetryError is thrown
          });
          expect.unreachable('Should throw RetryError');
        } catch (error) {
          expect(error).toBeInstanceOf(RetryError);
          expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
          expect(fallbackModel.doEmbed).toHaveBeenCalledTimes(1);
        }
      });

      it('should respect maxRetries setting when no retryable matches', async () => {
        // Arrange
        const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });

        // Act & Assert
        try {
          await embed({
            model: createRetryable({
              model: baseModel,
              retries: [],
            }),
            value: 'Hello!',
            maxRetries: 1, // Should be ignored since RetryError is thrown
          });
          expect.unreachable('Should throw RetryError');
        } catch (error) {
          expect(error).toBeInstanceOf(RetryError);
          expect(baseModel.doEmbed).toHaveBeenCalledTimes(2); // 1 initial + 1 retry
        }
      });
    });

    describe('delay', () => {
      it('should apply delay before retrying', async () => {
        // Arrange
        vi.useFakeTimers();
        const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
        const fallbackModel = new MockEmbeddingModel({
          doEmbed: mockEmbeddings,
        });
        const delayMs = 100;

        // Act
        const promise = embed({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel, delay: delayMs }],
          }),
          value: 'Hello!',
        });

        await vi.runAllTimersAsync();
        await promise;

        // Assert
        expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doEmbed).toHaveBeenCalledTimes(1);

        vi.useRealTimers();
      });

      it('should apply different delays for multiple retries', async () => {
        // Arrange
        vi.useFakeTimers();
        const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
        const fallbackModel1 = new MockEmbeddingModel({
          doEmbed: retryableError,
        });
        const fallbackModel2 = new MockEmbeddingModel({
          doEmbed: mockEmbeddings,
        });
        const delay1 = 50;
        const delay2 = 50;

        // Act
        const promise = embed({
          model: createRetryable({
            model: baseModel,
            retries: [
              { model: fallbackModel1, delay: delay1 },
              () => ({ model: fallbackModel2, delay: delay2 }),
            ],
          }),
          value: 'Hello!',
        });

        await vi.runAllTimersAsync();
        await promise;

        // Assert
        expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doEmbed).toHaveBeenCalledTimes(1);
        expect(fallbackModel2.doEmbed).toHaveBeenCalledTimes(1);

        vi.useRealTimers();
      });

      it('should not delay when delay is not specified', async () => {
        // Arrange
        const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
        const fallbackModel = new MockEmbeddingModel({
          doEmbed: mockEmbeddings,
        });

        // Act
        await embed({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel }],
          }),
          value: 'Hello!',
        });

        // Assert
        expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doEmbed).toHaveBeenCalledTimes(1);
      });
    });

    describe('providerOptions', () => {
      it('should override base model providerOptions with retry model providerOptions', async () => {
        // Arrange
        const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
        const fallbackModel = new MockEmbeddingModel({
          doEmbed: mockEmbeddings,
        });
        const originalProviderOptions = { openai: { user: 'original-user' } };
        const retryProviderOptions = { openai: { user: 'retry-user' } };

        // Act
        await embed({
          model: createRetryable({
            model: baseModel,
            retries: [
              {
                model: fallbackModel,
                providerOptions: retryProviderOptions,
              },
            ],
          }),
          value: 'Hello!',
          providerOptions: originalProviderOptions,
        });

        // Assert
        expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doEmbed).toHaveBeenCalledTimes(1);
        expect(baseModel.doEmbed).toHaveBeenCalledWith(
          expect.objectContaining({
            providerOptions: originalProviderOptions,
          }),
        );
        expect(fallbackModel.doEmbed).toHaveBeenCalledWith(
          expect.objectContaining({
            providerOptions: retryProviderOptions,
          }),
        );
      });
    });

    describe('timeout', () => {
      it('should create fresh abort signal with specified timeout on retry', async () => {
        // Arrange
        let baseModelSignal: AbortSignal | undefined;
        let fallbackModelSignal: AbortSignal | undefined;
        const abortError = new DOMException(
          'The operation was aborted',
          'AbortError',
        );

        const baseModel = new MockEmbeddingModel({
          doEmbed: async (opts: any) => {
            baseModelSignal = opts.abortSignal;
            throw abortError;
          },
        });

        const fallbackModel = new MockEmbeddingModel({
          doEmbed: async (opts: any) => {
            fallbackModelSignal = opts.abortSignal;
            // Verify the new signal is not aborted
            if (opts.abortSignal?.aborted) {
              throw new Error('Should not be aborted with fresh signal');
            }
            return mockEmbeddings;
          },
        });

        // Create an already-aborted signal
        const controller = new AbortController();
        controller.abort();

        // Act
        await embed({
          model: createRetryable({
            model: baseModel,
            retries: [
              {
                model: fallbackModel,
                timeout: 30000, // 30 second timeout for retry
              },
            ],
          }),
          value: 'Hello!',
          abortSignal: controller.signal,
        });

        // Assert
        expect(baseModel.doEmbed).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doEmbed).toHaveBeenCalledTimes(1);

        // Base model should receive the original aborted signal
        expect(baseModelSignal?.aborted).toBe(true);

        // Fallback model should receive a fresh, non-aborted signal
        expect(fallbackModelSignal).toBeDefined();
        expect(fallbackModelSignal?.aborted).toBe(false);
        expect(baseModelSignal).not.toBe(fallbackModelSignal);
      });
    });
  });

  describe('RetryError', () => {
    it('should throw RetryError when all retry attempts are exhausted', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });
      const fallbackModel1 = new MockEmbeddingModel({
        doEmbed: nonRetryableError,
      });
      const fallbackModel2 = new MockEmbeddingModel({
        doEmbed: retryableError,
      });

      // Act & Assert
      try {
        await embed({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackModel1, fallbackModel2],
          }),
          value: 'Hello!',
        });
        expect.unreachable(
          'Should throw RetryError when all attempts are exhausted',
        );
      } catch (error) {
        expect(error).toBeInstanceOf(RetryError);

        const retryError = error as RetryError;
        expect(retryError.reason).toBe('maxRetriesExceeded');
        expect(retryError.errors).toHaveLength(3);
        expect(retryError.errors[0]).toBe(retryableError);
        expect(retryError.errors[1]).toBe(nonRetryableError);
        expect(retryError.errors[2]).toBe(retryableError);
      }
    });

    it('should throw original error directly on first attempt with no retryables', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({ doEmbed: retryableError });

      // Act & Assert
      try {
        await embed({
          model: createRetryable({
            model: baseModel,
            retries: [], // No retry models
          }),
          value: 'Hello!',
          maxRetries: 0, // No automatic retries
        });
        expect.unreachable(
          'Should throw original error on first attempt with no retries',
        );
      } catch (error) {
        expect(error).not.toBeInstanceOf(RetryError);
        expect(error).toBe(retryableError);
      }
    });

    it('should throw original error directly when retryable returns undefined', async () => {
      // Arrange
      const baseModel = new MockEmbeddingModel({
        doEmbed: nonRetryableError,
      });

      // Act & Assert
      try {
        await embed({
          model: createRetryable({
            model: baseModel,
            retries: [() => undefined],
          }),
          value: 'Hello!',
          maxRetries: 0, // No automatic retries
        });
        expect.unreachable(
          'Should throw original error when retry models return undefined',
        );
      } catch (error) {
        expect(error).not.toBeInstanceOf(RetryError);
        expect(error).toBe(nonRetryableError);
      }
    });
  });
});
