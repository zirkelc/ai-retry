import { NoImageGeneratedError } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  MockImageModel,
  mockImageResult,
  mockImageResults,
  nonRetryableError,
  retryableError,
} from '../../../internal/test-utils.js';
import { retryableGenerateImage } from '../generate-image.js';
import { error, httpStatus, noImage, not, result } from './index.js';

const prompt = 'a cat';

describe('image-model call conditions', () => {
  describe('error', () => {
    it('should switch when the predicate matches', async () => {
      // Arrange
      const primary = MockImageModel.from(retryableError);
      const fallback = MockImageModel.from(mockImageResult);

      // Act
      const out = await retryableGenerateImage({
        model: primary,
        prompt,
        retry: [error(() => true).switch({ model: fallback })],
      });

      // Assert
      expect(out.images.length).toBe(1);
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });

    it('should not switch when the predicate misses', async () => {
      // Arrange
      const primary = MockImageModel.from(nonRetryableError);
      const fallback = MockImageModel.from(mockImageResult);

      // Act
      const out = retryableGenerateImage({
        model: primary,
        prompt,
        retry: [not(error(() => true)).switch({ model: fallback })],
      });

      // Assert
      await expect(out).rejects.toThrow();
      expect(fallback.doGenerate.mock.calls.length).toBe(0);
    });
  });

  describe('httpStatus', () => {
    it('should switch on a matching status code', async () => {
      // Arrange
      const primary = MockImageModel.from(retryableError);
      const fallback = MockImageModel.from(mockImageResult);

      // Act
      await retryableGenerateImage({
        model: primary,
        prompt,
        retry: [httpStatus(429).switch({ model: fallback })],
      });

      // Assert
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });
  });

  describe('noImage', () => {
    it('should switch when the model generated no images', async () => {
      // Arrange
      const primary = MockImageModel.from(
        new NoImageGeneratedError({ responses: [] }),
      );
      const fallback = MockImageModel.from(mockImageResult);

      // Act
      const out = await retryableGenerateImage({
        model: primary,
        prompt,
        retry: [noImage().switch({ model: fallback })],
      });

      // Assert
      expect(out.images.length).toBe(1);
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });

    it('should not switch on an unrelated error', async () => {
      // Arrange
      const primary = MockImageModel.from(nonRetryableError);
      const fallback = MockImageModel.from(mockImageResult);

      // Act
      const out = retryableGenerateImage({
        model: primary,
        prompt,
        retry: [noImage().switch({ model: fallback })],
      });

      // Assert
      await expect(out).rejects.toThrow();
      expect(fallback.doGenerate.mock.calls.length).toBe(0);
    });
  });

  describe('result', () => {
    it('should switch on too few images, with no guard needed', async () => {
      // Arrange — one entry point, one union member.
      const primary = MockImageModel.from(mockImageResult);
      const fallback = MockImageModel.from(mockImageResults(2));

      // Act
      const out = await retryableGenerateImage({
        model: primary,
        prompt,
        retry: [
          result((res) => res.images.length < 2).switch({ model: fallback }),
        ],
      });

      // Assert
      expect(out.images.length).toBe(2);
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });

    it('should keep the result when the predicate misses', async () => {
      // Arrange
      const primary = MockImageModel.from(mockImageResult);
      const fallback = MockImageModel.from(mockImageResults(2));

      // Act
      const out = await retryableGenerateImage({
        model: primary,
        prompt,
        retry: [result(() => false).switch({ model: fallback })],
      });

      // Assert
      expect(out.images.length).toBe(1);
      expect(fallback.doGenerate.mock.calls.length).toBe(0);
    });
  });
});
