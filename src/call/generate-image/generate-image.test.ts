import { generateImage } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  MockImageModel,
  mockImageResult,
  mockImageResults,
  nonRetryableError,
  retryableError,
} from '../../internal/test-utils.js';
import type { ImageModel } from '../../types.js';
import { result as imageResult } from './conditions/index.js';
import { retryableGenerateImage } from './generate-image.js';

const prompt = 'a cat';

describe('retryableGenerateImage', () => {
  describe('success', () => {
    it('should return the first attempt when nothing fails', async () => {
      // Arrange
      const model = MockImageModel.from(mockImageResult);

      // Act
      const result = await retryableGenerateImage({ model, prompt });

      // Assert
      expect(result.images.length).toBe(1);
      expect(model.doGenerate.mock.calls.length).toBe(1);
    });

    it('should hand back the SDK result untouched', async () => {
      // Arrange — the loop passes the result straight through, so what the
      // caller gets is the SDK's own object rather than a copy or a wrapper.
      const model = MockImageModel.from(mockImageResult);

      // Act
      const direct = await generateImage({ model, prompt });
      const wrapped = await retryableGenerateImage({ model, prompt });

      // Assert — the SDK exposes most of a result through prototype getters, so
      // sameness of prototype is what says nothing rebuilt it on the way out.
      expect(Object.getPrototypeOf(wrapped)).toBe(
        Object.getPrototypeOf(direct),
      );
      expect(wrapped.images.length).toBe(1);
    });
  });

  describe('error-based retries', () => {
    it('should fall over to the next model after an error', async () => {
      // Arrange
      const primary = MockImageModel.from(retryableError);
      const fallback = MockImageModel.from(mockImageResult);

      // Act
      const result = await retryableGenerateImage({
        model: primary,
        prompt,
        retry: [fallback],
      });

      // Assert
      expect(result.images.length).toBe(1);
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });

    it('should surface the error when no retry matched', async () => {
      // Arrange
      const primary = MockImageModel.from(nonRetryableError);

      // Act
      const result = retryableGenerateImage({
        model: primary,
        prompt,
        retry: [],
      });

      // Assert
      await expect(result).rejects.toThrow(nonRetryableError);
    });

    it('should override the prompt for the retry attempt', async () => {
      // Arrange
      const primary = MockImageModel.from(retryableError);
      const fallback = MockImageModel.from(mockImageResult);

      // Act
      await retryableGenerateImage({
        model: primary,
        prompt,
        retry: [{ model: fallback, options: { prompt: 'a dog' } }],
      });

      // Assert
      expect(fallback.doGenerate.mock.calls[0]![0].prompt).toBe('a dog');
    });

    describe('deadlines', () => {
      it('should compose a retry timeout into the abort signal', async () => {
        // Arrange — `generateImage` has no `timeout` argument of its own.
        const primary = MockImageModel.from(retryableError);
        const slow = MockImageModel.from({
          ...mockImageResult,
          delayInMs: 5_000,
        });
        const rescue = MockImageModel.from(mockImageResult);

        // Act
        const result = await retryableGenerateImage({
          model: primary,
          prompt,
          retry: [{ model: slow, timeout: 50 }, rescue],
        });

        // Assert
        expect(result.images.length).toBe(1);
        expect(
          primary.doGenerate.mock.calls[0]![0].abortSignal,
        ).toBeUndefined();
        expect(slow.doGenerate.mock.calls[0]![0].abortSignal).toBeDefined();
        expect(rescue.doGenerate.mock.calls.length).toBe(1);
      });
    });
  });

  describe('result-based retries', () => {
    it('should fall over on too few images', async () => {
      // Arrange — fewer images than wanted, which is not an error.
      const primary = MockImageModel.from(mockImageResult);
      const fallback = MockImageModel.from(mockImageResults(2));

      // Act
      const result = await retryableGenerateImage({
        model: primary,
        prompt,
        retry: [
          imageResult((res) => res.images.length < 2).switch({
            model: fallback,
          }),
        ],
      });

      // Assert — the images read directly, with no guard: one entry point,
      // one member.
      expect(result.images.length).toBe(2);
      expect(fallback.doGenerate.mock.calls.length).toBe(1);
    });

    it('should keep the result when no condition matches', async () => {
      // Arrange
      const primary = MockImageModel.from(mockImageResult);
      const fallback = MockImageModel.from(mockImageResults(2));

      // Act
      const result = await retryableGenerateImage({
        model: primary,
        prompt,
        retry: [imageResult(() => false).switch({ model: fallback })],
      });

      // Assert
      expect(result.images.length).toBe(1);
      expect(fallback.doGenerate.mock.calls.length).toBe(0);
    });
  });
});
