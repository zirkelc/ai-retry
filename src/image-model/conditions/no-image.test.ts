import { Errors } from 'ai-test-kit';
import { APICallError, generateImage, NoImageGeneratedError } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  Image,
  MockImageModel,
  createRetryableModel,
} from '../../internal/test-utils.js';
import type { ImageModelGenerate } from '../../types.js';
import { noImage } from './index.js';

const okImage: ImageModelGenerate = {
  images: [Image.png()],
  warnings: [],
  response: {
    timestamp: new Date(0),
    modelId: 'mock-model',
    headers: undefined,
  },
};

describe('noImage', () => {
  describe('generateImage', () => {
    it('should switch on NoImageGeneratedError', async () => {
      // Arrange
      const baseModel = MockImageModel.from(
        new NoImageGeneratedError({ message: 'no image' }),
      );
      const retryModel = MockImageModel.from(okImage);

      // Act
      const out = await generateImage({
        model: createRetryableModel({
          model: baseModel,
          retries: [noImage().switch({ model: retryModel })],
        }),
        prompt: 'cat',
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(out.images.length).toBe(1);
    });

    it('should not switch on other errors', async () => {
      // Arrange
      const baseModel = MockImageModel.from(Errors.internalServerError());
      const retryModel = MockImageModel.from(okImage);

      // Act
      const out = generateImage({
        model: createRetryableModel({
          model: baseModel,
          retries: [noImage().switch({ model: retryModel })],
        }),
        prompt: 'cat',
        maxRetries: 0,
      });

      // Assert
      await expect(out).rejects.toThrow(APICallError);
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(0);
    });
  });
});
