import {
  APICallError,
  generateImage,
  NoImageGeneratedError,
  RetryError,
} from 'ai';
import { describe, expect, it, vi } from 'vitest';
import {
  createRetryable,
  MockImageModel,
  mockImageResult,
  nonRetryableError,
  retryableError,
  serviceOverloadedError,
  serviceUnavailableError,
} from './test-utils.js';
import { noImageGenerated } from '../retryables/no-image-generated.js';
import { requestTimeout } from '../retryables/request-timeout.js';
import { serviceOverloaded } from '../retryables/service-overloaded.js';
import { serviceUnavailable } from '../retryables/service-unavailable.js';
import type {
  ImageModel,
  OnRetryOverrides,
  Retryable,
  RetryableModelOptions,
  RetryContext,
} from '../types.js';
import { isErrorAttempt } from './guards.js';

type OnError = Required<RetryableModelOptions<ImageModel>>['onError'];
type OnRetry = Required<RetryableModelOptions<ImageModel>>['onRetry'];
type OnSuccess = Required<RetryableModelOptions<ImageModel>>['onSuccess'];
type OnFailure = Required<RetryableModelOptions<ImageModel>>['onFailure'];

const noImageError = new NoImageGeneratedError({
  message: `No image generated`,
});

describe('generateImage', () => {
  it('should generate image successfully when no errors occur', async () => {
    // Arrange
    const baseModel = new MockImageModel({
      doGenerate: mockImageResult,
    });
    const retryableModel = createRetryable({
      model: baseModel,
      retries: [],
    });

    // Act
    const result = await generateImage({
      model: retryableModel,
      prompt: `A beautiful sunset`,
    });

    // Assert
    expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
    expect(result.images.length).toBe(1);
  });

  describe('retries', () => {
    describe('error-based retries', () => {
      it('should retry with errors', async () => {
        // Arrange
        const baseModel = new MockImageModel({ doGenerate: retryableError });
        const fallbackModel = new MockImageModel({
          doGenerate: mockImageResult,
        });

        const fallbackRetryable = (context: RetryContext<ImageModel>) => {
          if (
            isErrorAttempt(context.current) &&
            APICallError.isInstance(context.current.error)
          ) {
            return { model: fallbackModel, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackRetryable],
          }),
          prompt: `A beautiful sunset`,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(result.images.length).toBe(1);
      });

      it('should not retry without errors', async () => {
        // Arrange
        const baseModel = new MockImageModel({ doGenerate: mockImageResult });
        const fallbackModel1 = new MockImageModel({
          doGenerate: mockImageResult,
        });
        const fallbackModel2 = new MockImageModel({
          doGenerate: mockImageResult,
        });

        // Act
        const result = await generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackModel1, fallbackModel2],
          }),
          prompt: `A beautiful sunset`,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(0);
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(0);
        expect(result.images.length).toBe(1);
      });

      it('should use plain image models for error-based attempts', async () => {
        // Arrange
        const baseModel = new MockImageModel({ doGenerate: retryableError });
        const fallbackModel1 = new MockImageModel({
          doGenerate: mockImageResult,
        });
        const fallbackModel2 = new MockImageModel({
          doGenerate: mockImageResult,
        });

        const fallbackRetryable = (context: RetryContext<ImageModel>) => {
          if (
            isErrorAttempt(context.current) &&
            APICallError.isInstance(context.current.error)
          ) {
            return { model: fallbackModel2, maxAttempts: 1 };
          }
          return undefined;
        };

        // Act
        const result = await generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackModel1, fallbackRetryable],
          }),
          prompt: `A beautiful sunset`,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel1.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel2.doGenerate).toHaveBeenCalledTimes(0);
        expect(result.images.length).toBe(1);
      });
    });
  });

  describe('disabled', () => {
    it('should not retry when disabled is true', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: retryableError });
      const fallbackModel = new MockImageModel({
        doGenerate: mockImageResult,
      });

      const fallbackRetryable: Retryable<ImageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: true,
      });

      // Act & Assert
      const result = generateImage({
        model: retryableModel,
        prompt: `A beautiful sunset`,
        maxRetries: 0 /** Disable AI SDK's own retry mechanism */,
      });
      await expect(result).rejects.toThrow();
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(0);
    });

    it('should not retry when disabled function returns true', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: retryableError });
      const fallbackModel = new MockImageModel({
        doGenerate: mockImageResult,
      });

      const fallbackRetryable: Retryable<ImageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackRetryable],
        disabled: () => true,
      });

      // Act & Assert
      const result = generateImage({
        model: retryableModel,
        prompt: `A beautiful sunset`,
        maxRetries: 0 /** Disable AI SDK's own retry mechanism */,
      });
      await expect(result).rejects.toThrow();
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(0);
    });
  });

  describe('onError', () => {
    it('should call onError callback when error occurs', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: retryableError });
      const fallbackModel = new MockImageModel({
        doGenerate: mockImageResult,
      });

      const onErrorSpy = vi.fn<OnError>();

      const fallbackRetryable: Retryable<ImageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      // Act
      await generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackRetryable],
          onError: onErrorSpy,
        }),
        prompt: `A beautiful sunset`,
      });

      // Assert
      expect(onErrorSpy).toHaveBeenCalledTimes(1);
      expect(onErrorSpy.mock.calls[0]?.[0].current.type).toBe(`error`);
    });
  });

  describe('onRetry', () => {
    it('should call onRetry callback before retry', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: retryableError });
      const fallbackModel = new MockImageModel({
        doGenerate: mockImageResult,
      });

      const onRetrySpy = vi.fn<OnRetry>();

      const fallbackRetryable: Retryable<ImageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      // Act
      await generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackRetryable],
          onRetry: onRetrySpy,
        }),
        prompt: `A beautiful sunset`,
      });

      // Assert
      expect(onRetrySpy).toHaveBeenCalledTimes(1);
    });

    describe('overrides', () => {
      it('should override size and seed for the upcoming retry attempt', async () => {
        // Arrange
        const baseModel = new MockImageModel({ doGenerate: retryableError });
        const fallbackModel = new MockImageModel({
          doGenerate: mockImageResult,
        });
        const override: OnRetryOverrides<ImageModel> = {
          options: { size: `1024x1024`, seed: 42 },
        };

        // Act
        await generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackModel],
            onRetry: () => override,
          }),
          prompt: `A beautiful sunset`,
          size: `512x512`,
        });

        // Assert
        const fallbackCallOptions = (fallbackModel.doGenerate as any).mock
          .calls[0][0];
        expect(fallbackCallOptions.size).toBe(`1024x1024`);
        expect(fallbackCallOptions.seed).toBe(42);
      });

      it('should override providerOptions for the upcoming retry attempt', async () => {
        // Arrange
        const baseModel = new MockImageModel({ doGenerate: retryableError });
        const fallbackModel = new MockImageModel({
          doGenerate: mockImageResult,
        });
        const sanitizedProviderOptions = { openai: { quality: `low` } };

        // Act
        await generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackModel],
            onRetry: () => ({
              options: { providerOptions: sanitizedProviderOptions },
            }),
          }),
          prompt: `A beautiful sunset`,
          providerOptions: { openai: { quality: `high` } },
        });

        // Assert
        const fallbackCallOptions = (fallbackModel.doGenerate as any).mock
          .calls[0][0];
        expect(fallbackCallOptions.providerOptions).toEqual(
          sanitizedProviderOptions,
        );
      });

      it('should let onRetry overrides beat Retry.options', async () => {
        // Arrange
        const baseModel = new MockImageModel({ doGenerate: retryableError });
        const fallbackModel = new MockImageModel({
          doGenerate: mockImageResult,
        });
        const override: OnRetryOverrides<ImageModel> = {
          options: { size: `2048x2048` },
        };

        // Act
        await generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel, options: { size: `1024x1024` } }],
            onRetry: () => override,
          }),
          prompt: `A beautiful sunset`,
        });

        // Assert
        const fallbackCallOptions = (fallbackModel.doGenerate as any).mock
          .calls[0][0];
        expect(fallbackCallOptions.size).toBe(`2048x2048`);
      });

      it('should fall back to Retry.options when onRetry returns undefined', async () => {
        // Arrange
        const baseModel = new MockImageModel({ doGenerate: retryableError });
        const fallbackModel = new MockImageModel({
          doGenerate: mockImageResult,
        });

        // Act
        await generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel, options: { size: `1024x1024` } }],
            onRetry: () => undefined,
          }),
          prompt: `A beautiful sunset`,
        });

        // Assert
        const fallbackCallOptions = (fallbackModel.doGenerate as any).mock
          .calls[0][0];
        expect(fallbackCallOptions.size).toBe(`1024x1024`);
      });

      it('should support async onRetry returning overrides', async () => {
        // Arrange
        const baseModel = new MockImageModel({ doGenerate: retryableError });
        const fallbackModel = new MockImageModel({
          doGenerate: mockImageResult,
        });

        // Act
        await generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackModel],
            onRetry: async () => {
              await Promise.resolve();
              return { options: { seed: 99 } };
            },
          }),
          prompt: `A beautiful sunset`,
        });

        // Assert
        const fallbackCallOptions = (fallbackModel.doGenerate as any).mock
          .calls[0][0];
        expect(fallbackCallOptions.seed).toBe(99);
      });
    });
  });

  describe('onSuccess', () => {
    it('should call onSuccess with base model when no retry occurs', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: mockImageResult });
      const onSuccessSpy = vi.fn<OnSuccess>();

      // Act
      await generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [],
          onSuccess: onSuccessSpy,
        }),
        prompt: `A beautiful sunset`,
      });

      // Assert
      expect(onSuccessSpy).toHaveBeenCalledTimes(1);

      const successCall = onSuccessSpy.mock.calls[0]![0];
      expect(successCall.current.type).toBe('success');
      expect(successCall.current.model).toBe(baseModel);
      expect(successCall.current.result).toBeDefined();
      expect(successCall.attempts.length).toBe(0);
    });

    it('should call onSuccess with fallback model after retry', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: retryableError });
      const fallbackModel = new MockImageModel({
        doGenerate: mockImageResult,
      });

      const onSuccessSpy = vi.fn<OnSuccess>();

      // Act
      await generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackModel],
          onSuccess: onSuccessSpy,
        }),
        prompt: `A beautiful sunset`,
      });

      // Assert
      expect(onSuccessSpy).toHaveBeenCalledTimes(1);

      const successCall = onSuccessSpy.mock.calls[0]![0];
      expect(successCall.current.type).toBe('success');
      expect(successCall.current.model).toBe(fallbackModel);
      expect(successCall.current.result).toBeDefined();
      expect(successCall.attempts.length).toBe(1);
    });

    it('should NOT call onSuccess when all retries fail', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: retryableError });
      const fallbackModel = new MockImageModel({
        doGenerate: nonRetryableError,
      });
      const onSuccessSpy = vi.fn<OnSuccess>();

      // Act & Assert
      const result = generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackModel],
          onSuccess: onSuccessSpy,
        }),
        prompt: `A beautiful sunset`,
      });
      await expect(result).rejects.toThrow();

      expect(onSuccessSpy).not.toHaveBeenCalled();
    });
  });

  describe('onFailure', () => {
    it('should call onFailure with raw error when no retry is available', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: nonRetryableError });
      const onFailureSpy = vi.fn<OnFailure>();

      // Act
      const result = generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [],
          onFailure: onFailureSpy,
        }),
        prompt: `A beautiful sunset`,
      });
      await expect(result).rejects.toThrow();

      // Assert
      expect(onFailureSpy).toHaveBeenCalledTimes(1);

      const failureCall = onFailureSpy.mock.calls[0]![0];
      expect(failureCall.current.type).toBe('error');
      expect(failureCall.current.model).toBe(baseModel);
      expect(failureCall.attempts.length).toBe(1);
      expect(failureCall.error).toBe(nonRetryableError);
    });

    it('should call onFailure with RetryError when retries are exhausted', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: retryableError });
      const fallbackModel = new MockImageModel({
        doGenerate: nonRetryableError,
      });
      const onFailureSpy = vi.fn<OnFailure>();

      // Act
      const result = generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackModel],
          onFailure: onFailureSpy,
        }),
        prompt: `A beautiful sunset`,
      });
      await expect(result).rejects.toThrow();

      // Assert
      expect(onFailureSpy).toHaveBeenCalledTimes(1);

      const failureCall = onFailureSpy.mock.calls[0]![0];
      expect(failureCall.current.type).toBe('error');
      expect(failureCall.current.model).toBe(fallbackModel);
      expect(failureCall.attempts.length).toBe(2);
      expect(failureCall.error).toBeInstanceOf(RetryError);
    });

    it('should NOT call onFailure on success', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: mockImageResult });
      const onFailureSpy = vi.fn<OnFailure>();
      const onSuccessSpy = vi.fn<OnSuccess>();

      // Act
      await generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [],
          onFailure: onFailureSpy,
          onSuccess: onSuccessSpy,
        }),
        prompt: `A beautiful sunset`,
      });

      // Assert
      expect(onSuccessSpy).toHaveBeenCalledTimes(1);
      expect(onFailureSpy).not.toHaveBeenCalled();
    });
  });

  describe('RetryError', () => {
    it('should throw RetryError when all retries fail', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: retryableError });
      const fallbackModel = new MockImageModel({
        doGenerate: nonRetryableError,
      });

      const fallbackRetryable = (context: RetryContext<ImageModel>) => {
        if (
          isErrorAttempt(context.current) &&
          APICallError.isInstance(context.current.error) &&
          context.current.error.isRetryable
        ) {
          return { model: fallbackModel, maxAttempts: 1 };
        }
        return undefined;
      };

      const retryableModel = createRetryable({
        model: baseModel,
        retries: [fallbackRetryable],
      });

      // Act & Assert
      const result = generateImage({
        model: retryableModel,
        prompt: `A beautiful sunset`,
      });
      await expect(result).rejects.toThrow(RetryError);
    });
  });

  describe('RetryableOptions', () => {
    describe('maxAttempts', () => {
      it('should respect maxAttempts per model', async () => {
        // Arrange
        const baseModel = new MockImageModel({ doGenerate: retryableError });
        const fallbackModel = new MockImageModel({
          doGenerate: retryableError,
        });

        const fallbackRetryable = (context: RetryContext<ImageModel>) => {
          if (isErrorAttempt(context.current)) {
            return { model: fallbackModel, maxAttempts: 2 };
          }
          return undefined;
        };

        const retryableModel = createRetryable({
          model: baseModel,
          retries: [fallbackRetryable],
        });

        // Act & Assert
        const result = generateImage({
          model: retryableModel,
          prompt: `A beautiful sunset`,
        });
        await expect(result).rejects.toThrow(RetryError);
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(2);
      });
    });

    describe('delay', () => {
      it('should apply delay before retry', async () => {
        // Arrange
        vi.useFakeTimers();
        const baseModel = new MockImageModel({ doGenerate: retryableError });
        const fallbackModel = new MockImageModel({
          doGenerate: mockImageResult,
        });

        const fallbackRetryable: Retryable<ImageModel> = () => {
          return { model: fallbackModel, maxAttempts: 1, delay: 1000 };
        };

        // Act
        const resultPromise = generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [fallbackRetryable],
          }),
          prompt: `A beautiful sunset`,
        });

        await vi.advanceTimersByTimeAsync(1000);
        const result = await resultPromise;

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(result.images.length).toBe(1);

        vi.useRealTimers();
      });
    });

    describe('size', () => {
      it('should override size on retry', async () => {
        // Arrange
        const baseModel = new MockImageModel({ doGenerate: retryableError });
        const fallbackModel = new MockImageModel({
          doGenerate: mockImageResult,
        });

        // Act
        await generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel, options: { size: `1024x1024` } }],
          }),
          prompt: `A beautiful sunset`,
          size: `512x512`,
        });

        // Assert
        const fallbackCallOptions = (fallbackModel.doGenerate as any).mock
          .calls[0][0];
        expect(fallbackCallOptions.size).toBe(`1024x1024`);
      });
    });

    describe('seed', () => {
      it('should override seed on retry', async () => {
        // Arrange
        const baseModel = new MockImageModel({ doGenerate: retryableError });
        const fallbackModel = new MockImageModel({
          doGenerate: mockImageResult,
        });

        // Act
        await generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [{ model: fallbackModel, options: { seed: 42 } }],
          }),
          prompt: `A beautiful sunset`,
          seed: 1,
        });

        // Assert
        const fallbackCallOptions = (fallbackModel.doGenerate as any).mock
          .calls[0][0];
        expect(fallbackCallOptions.seed).toBe(42);
      });
    });
  });

  describe('retryables', () => {
    describe('noImageGenerated handler', () => {
      it('should retry with fallback model on NoImageGeneratedError', async () => {
        // Arrange
        const baseModel = new MockImageModel({ doGenerate: noImageError });
        const fallbackModel = new MockImageModel({
          doGenerate: mockImageResult,
        });

        // Act
        const result = await generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [noImageGenerated(fallbackModel)],
          }),
          prompt: `A beautiful sunset`,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(result.images.length).toBe(1);
      });
    });

    describe('serviceOverloaded handler', () => {
      it('should retry with fallback model on 529 error', async () => {
        // Arrange
        const baseModel = new MockImageModel({
          doGenerate: serviceOverloadedError,
        });
        const fallbackModel = new MockImageModel({
          doGenerate: mockImageResult,
        });

        // Act
        const result = await generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [serviceOverloaded(fallbackModel)],
          }),
          prompt: `A beautiful sunset`,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(result.images.length).toBe(1);
      });
    });

    describe('serviceUnavailable handler', () => {
      it('should retry with fallback model on 503 error', async () => {
        // Arrange
        const baseModel = new MockImageModel({
          doGenerate: serviceUnavailableError,
        });
        const fallbackModel = new MockImageModel({
          doGenerate: mockImageResult,
        });

        // Act
        const result = await generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [serviceUnavailable(fallbackModel)],
          }),
          prompt: `A beautiful sunset`,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(result.images.length).toBe(1);
      });
    });

    describe('requestTimeout handler', () => {
      it('should retry with fallback model on timeout error', async () => {
        // Arrange
        const timeoutError = new Error(`Request timed out`);
        timeoutError.name = `TimeoutError`;

        const baseModel = new MockImageModel({ doGenerate: timeoutError });
        const fallbackModel = new MockImageModel({
          doGenerate: mockImageResult,
        });

        // Act
        const result = await generateImage({
          model: createRetryable({
            model: baseModel,
            retries: [requestTimeout(fallbackModel)],
          }),
          prompt: `A beautiful sunset`,
        });

        // Assert
        expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(fallbackModel.doGenerate).toHaveBeenCalledTimes(1);
        expect(result.images.length).toBe(1);
      });
    });
  });
});
