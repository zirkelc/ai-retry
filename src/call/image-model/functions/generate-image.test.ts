import { describe, expect, it } from 'vitest';
import {
  createSpanExporter,
  findSpan,
  Image,
  MockImageModel,
  nonRetryableError,
  retryableError,
} from '../../../internal/test-utils.js';
import type { ImageModel } from '../../../types.js';
import { result as imageResult } from '../conditions/index.js';
import { retryableGenerateImage } from './generate-image.js';

const prompt = 'a cat';
const oneImage = Image.result([Image.png()]);
const twoImages = Image.result([Image.png(), Image.png()]);

/** A `doGenerate` that takes `ms` to answer, and rejects if aborted first. */
const slowGenerate =
  (ms: number): ImageModel['doGenerate'] =>
  async ({ abortSignal }) => {
    await new Promise<void>((resolve, reject) => {
      const handle = setTimeout(resolve, ms);
      abortSignal?.addEventListener('abort', () => {
        clearTimeout(handle);
        reject(abortSignal.reason);
      });
    });
    return oneImage;
  };

describe('retryableGenerateImage', () => {
  describe('success', () => {
    it('should return the first attempt when nothing fails', async () => {
      // Arrange
      const model = MockImageModel.from(oneImage);

      // Act
      const result = await retryableGenerateImage({ model, prompt });

      // Assert
      expect(result.images.length).toBe(1);
      expect(model.doGenerate.mock.calls.length).toBe(1);
    });

    it('should hand back the SDK result untouched', async () => {
      // Arrange
      const model = MockImageModel.from(oneImage);

      // Act
      const result = await retryableGenerateImage({ model, prompt });

      // Assert
      expect('operation' in result).toBe(false);
      expect(result.image).toBe(result.images[0]);
    });
  });

  describe('retries', () => {
    it('should fall over to the next model after an error', async () => {
      // Arrange
      const primary = MockImageModel.from(retryableError);
      const fallback = MockImageModel.from(oneImage);

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
  });

  describe('deadlines', () => {
    it('should compose a retry timeout into the abort signal', async () => {
      // Arrange — `generateImage` has no `timeout` argument of its own.
      const primary = MockImageModel.from(retryableError);
      const slow = MockImageModel.from(slowGenerate(5_000));
      const rescue = MockImageModel.from(oneImage);

      // Act
      const result = await retryableGenerateImage({
        model: primary,
        prompt,
        retry: [{ model: slow, timeout: 50 }, rescue],
      });

      // Assert
      expect(result.images.length).toBe(1);
      expect(primary.doGenerate.mock.calls[0]![0].abortSignal).toBeUndefined();
      expect(slow.doGenerate.mock.calls[0]![0].abortSignal).toBeDefined();
      expect(rescue.doGenerate.mock.calls.length).toBe(1);
    });
  });

  describe('argument overrides', () => {
    it('should override the prompt for the retry attempt', async () => {
      // Arrange
      const primary = MockImageModel.from(retryableError);
      const fallback = MockImageModel.from(oneImage);

      // Act
      await retryableGenerateImage({
        model: primary,
        prompt,
        retry: [{ model: fallback, options: { prompt: 'a dog' } }],
      });

      // Assert
      expect(fallback.doGenerate.mock.calls[0]![0].prompt).toBe('a dog');
    });
  });

  describe('result-based retries', () => {
    it('should fall over on too few images', async () => {
      // Arrange — fewer images than wanted, which is not an error.
      const primary = MockImageModel.from(oneImage);
      const fallback = MockImageModel.from(twoImages);

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
      const primary = MockImageModel.from(oneImage);
      const fallback = MockImageModel.from(twoImages);

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

  describe('telemetry', () => {
    it('should emit an operation span named after the entry point', async () => {
      // Arrange
      const { exporter, tracer } = createSpanExporter();
      const model = MockImageModel.from(oneImage);

      // Act
      await retryableGenerateImage({
        model,
        prompt,
        retry: { retries: [], telemetry: { isEnabled: true, tracer } },
      });

      // Assert
      const operation = findSpan(exporter, 'ai_retry.generateImage');
      expect(operation.attributes['gen_ai.operation.name']).toBe(
        'generate_content',
      );
    });
  });
});
