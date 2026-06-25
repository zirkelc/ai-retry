import { APICallError, generateImage } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  Errors,
  Image,
  MockImageModel,
  createRetryableModel,
} from '../../internal/test-utils.js';
import type { ImageModelGenerate } from '../../types.js';
import { error, httpStatus } from './index.js';

const okImage: ImageModelGenerate = {
  images: [Image.png()],
  warnings: [],
  response: {
    timestamp: new Date(0),
    modelId: 'mock-model',
    headers: undefined,
  },
};

describe('error (image)', () => {
  describe('generateImage', () => {
    it('should switch when predicate matches', async () => {
      // Arrange
      const baseModel = MockImageModel.from(Errors.from({ statusCode: 503 }));
      const retryModel = MockImageModel.from(okImage);

      // Act
      const out = await generateImage({
        model: createRetryableModel({
          model: baseModel,
          retries: [
            error(
              (e) => APICallError.isInstance(e) && e.statusCode === 503,
            ).switch({ model: retryModel }),
          ],
        }),
        prompt: 'cat',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(out.images.length).toBe(1);
    });

    it('should switch on httpStatus match', async () => {
      // Arrange
      const baseModel = MockImageModel.from(Errors.from({ statusCode: 529 }));
      const retryModel = MockImageModel.from(okImage);

      // Act
      const out = await generateImage({
        model: createRetryableModel({
          model: baseModel,
          retries: [httpStatus(529).switch({ model: retryModel })],
        }),
        prompt: 'cat',
        maxRetries: 0,
      });

      // Assert
      expect(baseModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(retryModel.doGenerate).toHaveBeenCalledTimes(1);
      expect(out.images.length).toBe(1);
    });
  });
});
