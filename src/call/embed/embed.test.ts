import { embed } from 'ai';
import { describe, expect, it } from 'vitest';
import {
  Embedding,
  MockEmbeddingModel,
  nonRetryableError,
  retryableError,
  slowCall,
} from '../../internal/test-utils.js';
import type { EmbeddingModel } from '../../types.js';
import { result as embeddingResult } from './conditions/index.js';
import { retryableEmbed } from './embed.js';

const value = 'hi';

describe('retryableEmbed', () => {
  describe('success', () => {
    it('should return the first attempt when nothing fails', async () => {
      // Arrange
      const model = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const result = await retryableEmbed({ model, value });

      // Assert
      expect(result.embedding.length).toBe(3);
      expect(result.value).toBe(value);
      expect(model.doEmbed.mock.calls.length).toBe(1);
    });

    it('should hand back the SDK result untouched', async () => {
      // Arrange — the loop passes the result straight through, so what the
      // caller gets is the SDK's own object rather than a copy or a wrapper.
      const model = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const direct = await embed({ model, value });
      const wrapped = await retryableEmbed({ model, value });

      // Assert — the SDK exposes most of a result through prototype getters, so
      // sameness of prototype is what says nothing rebuilt it on the way out.
      expect(Object.getPrototypeOf(wrapped)).toBe(
        Object.getPrototypeOf(direct),
      );
      expect(wrapped.embedding.length).toBe(3);
    });
  });

  describe('error-based retries', () => {
    it('should fall over to the next model after an error', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from(retryableError);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const result = await retryableEmbed({
        model: primary,
        value,
        retry: [fallback],
      });

      // Assert
      expect(result.embedding.length).toBe(3);
      expect(fallback.doEmbed.mock.calls.length).toBe(1);
    });

    it('should surface the error when no retry matched', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from(nonRetryableError);

      // Act
      const result = retryableEmbed({ model: primary, value, retry: [] });

      // Assert
      await expect(result).rejects.toThrow(nonRetryableError);
    });

    it('should override the value for the retry attempt', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from(retryableError);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      await retryableEmbed({
        model: primary,
        value,
        retry: [{ model: fallback, options: { value: 'rephrased' } }],
      });

      // Assert
      expect(fallback.doEmbed.mock.calls[0]![0].values).toEqual(['rephrased']);
    });

    describe('deadlines', () => {
      it('should compose a retry timeout into the abort signal', async () => {
        // Arrange — `embed` has no `timeout` argument, so the deadline can only
        // be expressed as a signal.
        const primary = MockEmbeddingModel.from(retryableError);
        const slow = MockEmbeddingModel.from(
          slowCall(5_000, Embedding.result([Embedding.vector(3)])),
        );
        const rescue = MockEmbeddingModel.from([Embedding.vector(3)]);

        // Act
        const result = await retryableEmbed({
          model: primary,
          value,
          retry: [{ model: slow, timeout: 50 }, rescue],
        });

        // Assert
        expect(result.embedding.length).toBe(3);
        expect(primary.doEmbed.mock.calls[0]![0].abortSignal).toBeUndefined();
        expect(slow.doEmbed.mock.calls[0]![0].abortSignal).toBeDefined();
        expect(rescue.doEmbed.mock.calls.length).toBe(1);
      });

      it("should compose the caller's own signal in alongside the deadline", async () => {
        // Arrange — a genuine cancel still has to propagate mid-attempt.
        const controller = new AbortController();
        const primary = MockEmbeddingModel.from(retryableError);
        const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

        // Act
        await retryableEmbed({
          model: primary,
          value,
          abortSignal: controller.signal,
          retry: [{ model: fallback, timeout: 5_000 }],
        });

        // Assert
        const signal = fallback.doEmbed.mock.calls[0]![0].abortSignal;
        expect(signal).toBeDefined();
        expect(signal).not.toBe(controller.signal);
      });
    });
  });

  describe('result-based retries', () => {
    it('should fall over on a degenerate embedding', async () => {
      // Arrange — a result no error path would ever surface.
      const primary = MockEmbeddingModel.from([[0, 0, 0]]);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const result = await retryableEmbed({
        model: primary,
        value,
        retry: [
          embeddingResult((res) => res.embedding.every((n) => n === 0)).switch({
            model: fallback,
          }),
        ],
      });

      // Assert
      expect(result.embedding).toEqual([0.1, 0.2, 0.3]);
      expect(fallback.doEmbed.mock.calls.length).toBe(1);
    });

    it('should keep the result when no condition matches', async () => {
      // Arrange
      const primary = MockEmbeddingModel.from([[0, 0, 0]]);
      const fallback = MockEmbeddingModel.from([Embedding.vector(3)]);

      // Act
      const result = await retryableEmbed({
        model: primary,
        value,
        retry: [embeddingResult(() => false).switch({ model: fallback })],
      });

      // Assert
      expect(result.embedding).toEqual([0, 0, 0]);
      expect(fallback.doEmbed.mock.calls.length).toBe(0);
    });
  });
});
