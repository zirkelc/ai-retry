import { APICallError, generateImage, NoImageGeneratedError } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import {
  createRetryable,
  MockImageModel,
  mockImageResult,
} from '../internal/test-utils.js';
import type { ImageModel, RetryableModelOptions } from '../types.js';
import { noImageGenerated } from './no-image-generated.js';

type OnError = Required<RetryableModelOptions<ImageModel>>['onError'];
type OnRetry = Required<RetryableModelOptions<ImageModel>>['onRetry'];

const noImageError = new NoImageGeneratedError({
  message: `No image generated`,
});

const otherError = new APICallError({
  message: `Some other error`,
  url: ``,
  requestBodyValues: {},
  statusCode: 400,
  responseHeaders: {},
  responseBody: `{}`,
  isRetryable: false,
  data: {
    error: {
      message: `Some other error`,
      type: null,
      param: `prompt`,
      code: `other_error`,
    },
  },
});

describe('noImageGenerated', () => {
  describe('generateImage', () => {
    it('should succeed without errors', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: mockImageResult });
      const retryModel = new MockImageModel({ doGenerate: mockImageResult });

      // Act
      const result = await generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [noImageGenerated(retryModel)],
        }),
        prompt: `test`,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
      expect(result.images.length).toBe(1);
    });

    it('should fallback on NoImageGeneratedError', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: noImageError });
      const retryModel = new MockImageModel({ doGenerate: mockImageResult });

      // Act
      const result = await generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [noImageGenerated(retryModel)],
        }),
        prompt: `test`,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(result.images.length).toBe(1);
    });

    it('should not fallback for non-matching errors', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: otherError });
      const retryModel = new MockImageModel({ doGenerate: mockImageResult });

      // Act
      const result = generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [noImageGenerated(retryModel)],
        }),
        prompt: `test`,
        maxRetries: 0,
      });

      // Assert
      await expect(result).rejects.toThrowError(APICallError);
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    });

    it('should call onError callback when error occurs', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: noImageError });
      const retryModel = new MockImageModel({ doGenerate: mockImageResult });
      const onError = vi.fn<OnError>();

      // Act
      await generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [noImageGenerated(retryModel)],
          onError,
        }),
        prompt: `test`,
      });

      // Assert
      expect(onError).toHaveBeenCalledTimes(1);
      expect(onError.mock.calls[0]?.[0].current.type).toBe(`error`);
    });

    it('should call onRetry callback before retry', async () => {
      // Arrange
      const baseModel = new MockImageModel({ doGenerate: noImageError });
      const retryModel = new MockImageModel({ doGenerate: mockImageResult });
      const onRetry = vi.fn<OnRetry>();

      // Act
      await generateImage({
        model: createRetryable({
          model: baseModel,
          retries: [noImageGenerated(retryModel)],
          onRetry,
        }),
        prompt: `test`,
      });

      // Assert
      expect(onRetry).toHaveBeenCalledTimes(1);
    });
  });
});
