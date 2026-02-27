import {
  APICallError,
  generateImage,
  NoImageGeneratedError,
  RetryError,
} from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { createRetryable } from './create-retryable-model.js';
import { noImageGenerated } from './retryables/no-image-generated.js';
import { requestTimeout } from './retryables/request-timeout.js';
import { serviceOverloaded } from './retryables/service-overloaded.js';
import { serviceUnavailable } from './retryables/service-unavailable.js';
import { MockImageModel } from './test-utils.js';
import type {
  ImageModel,
  ImageModelGenerate,
  Retryable,
  RetryableModelOptions,
  RetryContext,
} from './types.js';
import { isErrorAttempt } from './utils.js';

type OnError = Required<RetryableModelOptions<ImageModel>>['onError'];
type OnRetry = Required<RetryableModelOptions<ImageModel>>['onRetry'];

/** Valid base64 PNG image (1x1 transparent pixel) */
const validBase64Image = `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==`;

const mockImageResult: ImageModelGenerate = {
  images: [validBase64Image],
  warnings: [],
  response: {
    timestamp: new Date(),
    modelId: `mock-model`,
    headers: undefined,
  },
};

const retryableError = new APICallError({
  message: `Rate limit exceeded`,
  url: ``,
  requestBodyValues: {},
  statusCode: 429,
  responseHeaders: {},
  responseBody: `{"error": {"message": "Rate limit exceeded", "code": "rate_limit_exceeded"}}`,
  isRetryable: true,
  data: {
    error: {
      message: `Rate limit exceeded`,
      code: `rate_limit_exceeded`,
    },
  },
});

const nonRetryableError = new APICallError({
  message: `Invalid API key`,
  url: ``,
  requestBodyValues: {},
  statusCode: 401,
  responseHeaders: {},
  responseBody: `{"error": {"message": "Invalid API key", "code": "invalid_api_key"}}`,
  isRetryable: false,
  data: {
    error: {
      message: `Invalid API key`,
      code: `invalid_api_key`,
    },
  },
});

const serviceOverloadedError = new APICallError({
  message: `Service overloaded`,
  url: ``,
  requestBodyValues: {},
  statusCode: 529,
  responseHeaders: {},
  responseBody: `{"error": {"message": "Service overloaded"}}`,
  isRetryable: true,
  data: {
    error: {
      message: `Service overloaded`,
    },
  },
});

const serviceUnavailableError = new APICallError({
  message: `Service unavailable`,
  url: ``,
  requestBodyValues: {},
  statusCode: 503,
  responseHeaders: {},
  responseBody: `{"error": {"message": "Service unavailable"}}`,
  isRetryable: true,
  data: {
    error: {
      message: `Service unavailable`,
    },
  },
});

const noImageError = new NoImageGeneratedError({
  message: `No image generated`,
});

describe(`generateImage`, () => {
  it(`should generate image successfully when no errors occur`, async () => {
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

  describe(`retries`, () => {
    describe(`error-based retries`, () => {
      it(`should retry with errors`, async () => {
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

      it(`should not retry without errors`, async () => {
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

      it(`should use plain image models for error-based attempts`, async () => {
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

  describe(`disabled`, () => {
    it(`should not retry when disabled is true`, async () => {
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

    it(`should not retry when disabled function returns true`, async () => {
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

  describe(`callbacks`, () => {
    it(`should call onError callback when error occurs`, async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: retryableError });
      const fallbackModel = new MockImageModel({
        doGenerate: mockImageResult,
      });

      const onError = vi.fn<OnError>();

      const fallbackRetryable: Retryable<ImageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      // Act
      await generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackRetryable],
          onError,
        }),
        prompt: `A beautiful sunset`,
      });

      // Assert
      expect(onError).toHaveBeenCalledTimes(1);
      expect(onError.mock.calls[0]?.[0].current.type).toBe(`error`);
    });

    it(`should call onRetry callback before retry`, async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: retryableError });
      const fallbackModel = new MockImageModel({
        doGenerate: mockImageResult,
      });

      const onRetry = vi.fn<OnRetry>();

      const fallbackRetryable: Retryable<ImageModel> = () => {
        return { model: fallbackModel, maxAttempts: 1 };
      };

      // Act
      await generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackRetryable],
          onRetry,
        }),
        prompt: `A beautiful sunset`,
      });

      // Assert
      expect(onRetry).toHaveBeenCalledTimes(1);
    });
  });

  describe(`RetryError`, () => {
    it(`should throw RetryError when all retries fail`, async () => {
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

  describe(`noImageGenerated handler`, () => {
    it(`should retry with fallback model on NoImageGeneratedError`, async () => {
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

  describe(`serviceOverloaded handler`, () => {
    it(`should retry with fallback model on 529 error`, async () => {
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

  describe(`serviceUnavailable handler`, () => {
    it(`should retry with fallback model on 503 error`, async () => {
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

  describe(`requestTimeout handler`, () => {
    it(`should retry with fallback model on timeout error`, async () => {
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

  describe(`maxAttempts`, () => {
    it(`should respect maxAttempts per model`, async () => {
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

  describe(`delay and backoff`, () => {
    it(`should apply delay before retry`, async () => {
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

  describe(`call options override`, () => {
    it(`should override call options on retry`, async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: retryableError });
      const fallbackModel = new MockImageModel({
        doGenerate: mockImageResult,
      });

      const fallbackRetryable: Retryable<ImageModel> = () => {
        return {
          model: fallbackModel,
          maxAttempts: 1,
          options: {
            size: `1024x1024`,
            seed: 42,
          },
        };
      };

      // Act
      await generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [fallbackRetryable],
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
  });
});
